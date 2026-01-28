import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from itertools import product
from config import *
from models import LateFusionModel

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, save_path=None):
        """
        :param patience: 连续多少轮验证集指标无提升则停止
        :param min_delta: 指标提升的最小阈值
        :param save_path: 最佳模型保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_f1 = 0.0
        self.save_path = save_path
        self.early_stop = False

    def __call__(self, current_val_f1, model):
        if current_val_f1 > self.best_val_f1 + self.min_delta:
            self.best_val_f1 = current_val_f1
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# evaluate_model 函数
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask, image)
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    
    return avg_val_loss, val_acc, val_f1

# train_model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path):
    best_val_f1 = 0.0
    # patience=3
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, save_path=save_path)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask, image)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print(f"\n早停触发！当前轮次：{epoch+1}，最佳验证F1：{early_stopping.best_val_f1:.4f}")
            break
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print("-" * 50)
    
    return early_stopping.best_val_f1

# hyperparameter_search 函数
def hyperparameter_search(train_dataset, val_dataset, device):
    param_grid = PARAM_GRID
    best_params = None
    best_f1 = 0.0
    results = []
    
    param_combinations = list(product(
        param_grid["batch_size"],
        param_grid["lr"],
        param_grid["dropout_rate"],
        param_grid["epochs"]
    ))
    
    print(f"开始超参数搜索，共 {len(param_combinations)} 组参数组合")
    print("=" * 60)
    
    for idx, (batch_size, lr, dropout_rate, epochs) in enumerate(param_combinations):
        print(f"\n第 {idx+1}/{len(param_combinations)} 组参数：")
        print(f"batch_size: {batch_size}, lr: {lr}, dropout_rate: {dropout_rate}, epochs: {epochs}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = LateFusionModel(dropout_rate=dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        temp_save_path = os.path.join(RESULT_DIR, f"temp_model_{idx}.pth")
        
        current_f1 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            save_path=temp_save_path
        )
        
        result = {
            "batch_size": batch_size,
            "lr": lr,
            "dropout_rate": dropout_rate,
            "epochs": epochs,
            "val_f1": current_f1
        }
        results.append(result)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_params = result
            best_model_path = os.path.join(RESULT_DIR, "best_model_hyper.pth")
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            if os.path.exists(temp_save_path):
                os.rename(temp_save_path, best_model_path)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "hyperparameter_results.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("超参数搜索完成！")
    print(f"最佳参数：{best_params}")
    print(f"最佳验证F1：{best_f1:.4f}")
    print(f"搜索结果已保存到 {os.path.join(RESULT_DIR, 'hyperparameter_results.csv')}")
    
    return best_params, best_f1

# predict_model 函数
def predict_model(model, test_loader, device, label_map):
    model.eval()
    predictions = []
    guids = []
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            guid_batch = batch["guid"]
            
            logits = model(input_ids, attention_mask, image)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend([reverse_label_map[p] for p in preds])
            guids.extend(guid_batch)
    
    result_path = os.path.join(RESULT_DIR, "submission.csv")
    result_df = pd.DataFrame({"guid": guids, "tag": predictions})
    result_df.to_csv(result_path, index=False, header=False)
    
    print(f"预测结果已保存到 {result_path}")
    return result_df

# train_single_model 函数
def train_single_model(model_name, model, train_dataset, val_dataset, device, fixed_hyperparams):
    batch_size = fixed_hyperparams["batch_size"]
    lr = fixed_hyperparams["lr"]
    dropout_rate = fixed_hyperparams["dropout_rate"]
    epochs = fixed_hyperparams["epochs"]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model_save_path = os.path.join(RESULT_DIR, f"best_model_{model_name}.pth")
    
    print(f"\n开始训练 {model_name} 模型")
    print(f"固定超参数：{fixed_hyperparams}")
    print("-" * 50)
    # 训练（早停已集成）
    best_val_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        save_path=model_save_path
    )
    
    model.load_state_dict(torch.load(model_save_path))

    final_val_loss, final_val_acc, final_val_f1 = evaluate_model(model, val_loader, criterion, device)
    
    print(f"\n{model_name} 模型训练完成！")
    print(f"最终验证F1：{final_val_f1:.4f} | 最终验证Acc：{final_val_acc:.4f} | 最终验证Loss：{final_val_loss:.4f}")
    print("=" * 60)
    
    return {
        "model_name": model_name,
        "val_f1": final_val_f1,
        "val_acc": final_val_acc,
        "val_loss": final_val_loss,
        "hyperparams": fixed_hyperparams
    }