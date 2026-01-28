import os
import json
import torch
import pandas as pd
from torch.utils.data import random_split
from config import *
from dataset import MultimodalDataset
from models import LateFusionModel, EarlyFusionModel, CrossAttentionFusionModel
from trainer import hyperparameter_search, predict_model, train_single_model

# --------------------------
# 步骤状态管理工具函数
# --------------------------
def load_step_status():
    """加载步骤完成状态，不存在则初始化"""
    status_path = os.path.join(RESULT_DIR, "step_status.json")
    default_status = {
        "step1": False,   # 数据集拆分
        "step2": False,   # 超参数搜索
        "step3_late": False,  # LateFusion训练
        "step3_early": False, # EarlyFusion训练
        "step3_cross": False  # CrossAttentionFusion训练
    }
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            for key in default_status.keys():
                if key not in status:
                    status[key] = default_status[key]
            return status
        except:
            return default_status
    return default_status

def save_step_status(status):
    """保存步骤完成状态"""
    status_path = os.path.join(RESULT_DIR, "step_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

def init_missing_cache():
    """初始化缺失的缓存文件（适配已完成的步骤）"""
    status = load_step_status()
    if status["step1"] and not os.path.exists(os.path.join(RESULT_DIR, "dataset_split.pth")):
        print("检测到步骤1标记为已完成，正在补全数据集缓存...")
        full_dataset = MultimodalDataset(is_test=False)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        torch.save(
            {"train": train_dataset, "val": val_dataset, "full": full_dataset},
            os.path.join(RESULT_DIR, "dataset_split.pth")
        )
        print("数据集缓存补全完成！")
    
    if status["step2"] and not os.path.exists(os.path.join(RESULT_DIR, "best_hyperparams.pth")):
        print("检测到步骤2标记为已完成，正在补全超参数缓存...")
        hyper_csv_path = os.path.join(RESULT_DIR, "hyperparameter_results.csv")
        if os.path.exists(hyper_csv_path):
            # 从超参数搜索结果中读取最佳参数
            df = pd.read_csv(hyper_csv_path)
            best_idx = df["val_f1"].idxmax()
            best_params = df.iloc[best_idx][["batch_size", "lr", "dropout_rate", "epochs"]].to_dict()
            # 转换数值类型（CSV读取可能为float）
            best_params["batch_size"] = int(best_params["batch_size"])
            best_params["epochs"] = int(best_params["epochs"])
            torch.save(best_params, os.path.join(RESULT_DIR, "best_hyperparams.pth"))
            print("超参数缓存补全完成！")
        else:
            print("警告：超参数搜索结果文件缺失，无法补全缓存！")

def main():
    # 初始化缺失的缓存文件
    init_missing_cache()
    
    # 加载步骤状态
    step_status = load_step_status()
    
    print("=" * 60)
    full_dataset = None
    train_dataset = None
    val_dataset = None
    
    if not step_status["step1"]:
        print("步骤1：加载并拆分数据集")
        full_dataset = MultimodalDataset(
            data_dir=DATA_DIR,
            label_path=TRAIN_LABEL_PATH,
            is_test=False
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        # 保存数据集缓存
        torch.save(
            {"train": train_dataset, "val": val_dataset, "full": full_dataset},
            os.path.join(RESULT_DIR, "dataset_split.pth")
        )
        # 更新状态
        step_status["step1"] = True
        save_step_status(step_status)
        print(f"数据集拆分完成：训练集 {train_size} 条，验证集 {val_size} 条")
    else:
        print("步骤1：已完成，跳过")
        # 加载缓存的数据集
        dataset_cache = torch.load(os.path.join(RESULT_DIR, "dataset_split.pth"))
        train_dataset = dataset_cache["train"]
        val_dataset = dataset_cache["val"]
        full_dataset = dataset_cache["full"]
    
    print("\n" + "=" * 60)
    best_params = None
    
    if not step_status["step2"]:
        print("步骤2：执行超参数网格搜索（基于Late Fusion模型）")
        best_params, best_f1 = hyperparameter_search(train_dataset, val_dataset, DEVICE)
        # 保存最佳超参数
        torch.save(best_params, os.path.join(RESULT_DIR, "best_hyperparams.pth"))
        # 更新状态
        step_status["step2"] = True
        save_step_status(step_status)
    else:
        print("步骤2：已完成，跳过")
        # 加载缓存的最佳超参数
        best_params = torch.load(os.path.join(RESULT_DIR, "best_hyperparams.pth"))
    fixed_hyperparams = best_params
    
    print("\n" + "=" * 60)
    print("步骤3：执行3种多模态融合模型对比实验")
    
    # 定义待对比的模型
    models_to_compare = {
        "LateFusion": (LateFusionModel, "step3_late"),
        "EarlyFusion": (EarlyFusionModel, "step3_early"),
        "CrossAttentionFusion": (CrossAttentionFusionModel, "step3_cross")
    }
    
    # 加载已有的对比结果
    compare_results = []
    compare_result_path = os.path.join(RESULT_DIR, "multimodal_fusion_compare.csv")
    if os.path.exists(compare_result_path):
        compare_results = pd.read_csv(compare_result_path).to_dict("records")
    
    # 遍历模型进行训练
    for model_name, (model_cls, status_key) in models_to_compare.items():
        if step_status[status_key]:
            print(f"\n{model_name}：已完成，跳过")
            continue
        
        # 初始化模型
        model_instance = model_cls(dropout_rate=fixed_hyperparams["dropout_rate"]).to(DEVICE)
        
        # 训练模型
        model_result = train_single_model(
            model_name=model_name,
            model=model_instance,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=DEVICE,
            fixed_hyperparams=fixed_hyperparams
        )
        
        compare_results.append(model_result)
        
        step_status[status_key] = True
        save_step_status(step_status)
        pd.DataFrame(compare_results).to_csv(compare_result_path, index=False, encoding="utf-8-sig")
    
    print("\n对比实验汇总：")
    for result in compare_results:
        print(f"{result['model_name']} | F1: {result['val_f1']:.4f} | Acc: {result['val_acc']:.4f}")
    print(f"对比结果已保存到 {compare_result_path}")
    
    print("\n" + "=" * 60)
    if compare_results:
        best_fusion_model = max(compare_results, key=lambda x: x["val_f1"])
        best_fusion_model_name = best_fusion_model["model_name"]
        print(f"最佳融合模型：{best_fusion_model_name}（F1：{best_fusion_model['val_f1']:.4f}）")
    else:
        print("警告：无模型训练结果，默认选择LateFusion作为最佳模型")
        best_fusion_model_name = "LateFusion"
    
    print("\n" + "=" * 60)
    print("步骤4：加载测试集")
    test_dataset = MultimodalDataset(
        data_dir=DATA_DIR,
        label_path=TEST_LABEL_PATH,
        is_test=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=fixed_hyperparams["batch_size"],
        shuffle=False
    )
    print(f"测试集加载完成：共 {len(test_dataset)} 条数据")
    
    print("\n" + "=" * 60)
    print(f"步骤5：使用最佳融合模型（{best_fusion_model_name}）进行预测")
    if best_fusion_model_name == "LateFusion":
        best_predict_model = LateFusionModel(dropout_rate=fixed_hyperparams["dropout_rate"]).to(DEVICE)
    elif best_fusion_model_name == "EarlyFusion":
        best_predict_model = EarlyFusionModel(dropout_rate=fixed_hyperparams["dropout_rate"]).to(DEVICE)
    else:
        best_predict_model = CrossAttentionFusionModel(dropout_rate=fixed_hyperparams["dropout_rate"]).to(DEVICE)
    
    best_model_path = os.path.join(RESULT_DIR, f"best_model_{best_fusion_model_name}.pth")
    if os.path.exists(best_model_path):
        best_predict_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        predict_model(best_predict_model, test_loader, DEVICE, full_dataset.label_map)
    else:
        print(f"警告：{best_fusion_model_name}模型权重文件不存在，无法进行预测！")
    
    print("\n" + "=" * 60)
    print("所有流程执行完成！")

if __name__ == "__main__":
    main()