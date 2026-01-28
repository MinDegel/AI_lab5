# -*- coding: utf-8 -*-
"""
多模态实验 - 历史结果报告生成器
功能：独立打印所有步骤的历史记录，不影响主程序
运行方式：直接运行此文件，无需依赖主程序执行
"""
import os
import json
import torch
import pandas as pd
from config import * 
from dataset import MultimodalDataset

def load_step_status():
    """加载步骤状态（用于辅助判断）"""
    status_path = os.path.join(RESULT_DIR, "step_status.json")
    default_status = {
        "step1": False, "step2": False, 
        "step3_late": False, "step3_early": False, "step3_cross": False,
        "step4_predict": False
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

def print_history_report():
    """核心：打印完整历史结果报告"""
    print("=" * 80)
    print("多模态情感分类实验 - 结果报告")
    print(f"报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据来源：{RESULT_DIR} 目录下的缓存文件")
    print("=" * 80)
    print("\n【一、实验基础信息】")
    print("-" * 50)
    print(f"使用设备：{'GPU (CUDA)' if DEVICE.type == 'cuda' else 'CPU'}")
    print(f"模型类型：3种多模态融合模型（Late Fusion/Early Fusion/Cross-Attention Fusion）")
    print(f"数据目录：{DATA_DIR}")
    print(f"结果目录：{RESULT_DIR}")

    print("\n【二、步骤1 - 数据集拆分结果】")
    print("-" * 50)
    dataset_cache_path = os.path.join(RESULT_DIR, "dataset_split.pth")
    if os.path.exists(dataset_cache_path):
        try:
            dataset_cache = torch.load(dataset_cache_path)
            full_size = len(dataset_cache["full"])
            train_size = len(dataset_cache["train"])
            val_size = len(dataset_cache["val"])
            test_size = len(MultimodalDataset(is_test=True))  # 读取测试集大小

            print(f"数据规模统计：")
            print(f"- 完整训练集（含标签）：{full_size:>6} 条")
            print(f"- 训练集（80%）：{train_size:>10} 条（{train_size/full_size*100:>4.1f}%）")
            print(f"- 验证集（20%）：{val_size:>10} 条（{val_size/full_size*100:>4.1f}%）")
            print(f"- 测试集（无标签）：{test_size:>8} 条")
            print(f"缓存文件：dataset_split.pth（存在）")
        except Exception as e:
            print(f"读取数据集缓存失败：{str(e)}")
            print(f"缓存文件：dataset_split.pth（损坏或格式错误）")
    else:
        print(f"未找到数据集拆分缓存文件（dataset_split.pth）")
        print(f"提示：需先运行主程序步骤1，生成数据集拆分结果")


    print("\n【三、步骤2 - 超参数搜索结果】")
    print("-" * 50)
    hyper_csv_path = os.path.join(RESULT_DIR, "hyperparameter_results.csv")
    best_hyper_path = os.path.join(RESULT_DIR, "best_hyperparams.pth")

    if os.path.exists(hyper_csv_path) and os.path.exists(best_hyper_path):
        try:
            hyper_df = pd.read_csv(hyper_csv_path)
            best_params = torch.load(best_hyper_path)
            best_idx = hyper_df["val_f1"].idxmax()
            best_row = hyper_df.iloc[best_idx]

            print(f"搜索概况：")
            print(f"- 总搜索组合数：{len(hyper_df):>3} 组")
            print(f"- 最佳验证集F1：{best_row['val_f1']:>6.4f}")
            print(f"- 平均验证集F1：{hyper_df['val_f1'].mean():>6.4f}")

            print(f"\n最佳超参数组合：")
            print(f"- 批次大小（batch_size）：{int(best_params['batch_size']):>4}")
            print(f"- 学习率（lr）：{best_params['lr']:>10.6f}")
            print(f"- Dropout率：{best_params['dropout_rate']:>8.2f}")
            print(f"- 训练轮数（epochs）：{int(best_params['epochs']):>4}")

            top3_df = hyper_df.nlargest(3, "val_f1").reset_index(drop=True)
            print(f"\n前3组最优参数（按F1排序）：")
            print(f"   {'排名':<4} {'batch_size':<10} {'lr':<12} {'dropout':<8} {'F1分数':<8}")
            print(f"   {'-'*4:<4} {'-'*10:<10} {'-'*12:<12} {'-'*8:<8} {'-'*8:<8}")
            for i, row in top3_df.iterrows():
                print(f"   {i+1:<4} {int(row['batch_size']):<10} {row['lr']:.6f:<12} {row['dropout_rate']:.2f:<8} {row['val_f1']:.4f:<8}")

            print(f"\n相关文件：")
            print(f"   - hyperparameter_results.csv（超参数搜索完整记录）")
            print(f"   - best_hyperparams.pth（最佳超参数缓存）")
        except Exception as e:
            print(f"读取超参数结果失败：{str(e)}")
            print(f"提示：检查文件格式是否正确（如CSV是否被修改）")
    else:
        print(f"缺失超参数相关文件：")
        if not os.path.exists(hyper_csv_path):
            print(f"- hyperparameter_results.csv（超参数搜索记录，主程序步骤2生成）")
        if not os.path.exists(best_hyper_path):
            print(f"- best_hyperparams.pth（最佳超参数缓存，主程序步骤2生成）")

    print("\n【四、步骤3 - 多模态融合模型对比结果】")
    print("-" * 50)
    compare_csv_path = os.path.join(RESULT_DIR, "multimodal_fusion_compare.csv")

    if os.path.exists(compare_csv_path):
        try:
            compare_df = pd.read_csv(compare_csv_path)
            # 按F1降序排序
            compare_df_sorted = compare_df.sort_values("val_f1", ascending=False).reset_index(drop=True)

            print(f"模型性能汇总（按验证集F1排序）：")
            print(f"{'模型名称':<20} {'F1分数':<10} {'准确率':<10} {'验证损失':<10}")
            print(f"{'-'*20:<20} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10}")
            for _, row in compare_df_sorted.iterrows():
                model_name = row["model_name"]
                f1 = row["val_f1"]
                acc = row["val_acc"]
                loss = row["val_loss"]
                print(f"   {model_name:<20} {f1:<10.4f} {acc:<10.4f} {loss:<10.4f}")

            # 标注最佳模型
            best_model = compare_df_sorted.iloc[0]
            print(f"\n最佳融合模型：{best_model['model_name']}")
            print(f"- 最佳F1分数：{best_model['val_f1']:.4f}（高于其他模型 {best_model['val_f1'] - compare_df_sorted['val_f1'].iloc[1]:.4f}）")
            print(f"- 对应准确率：{best_model['val_acc']:.4f}")
            print(f"- 使用超参数：{eval(best_model['hyperparams'])}" if isinstance(best_model['hyperparams'], str) else best_model['hyperparams'])

            # 检查模型权重文件是否存在
            print(f"\n模型权重文件状态：")
            for _, row in compare_df_sorted.iterrows():
                model_name = row["model_name"]
                weight_path = os.path.join(RESULT_DIR, f"best_model_{model_name}.pth")
                if os.path.exists(weight_path):
                    file_size = os.path.getsize(weight_path) / 1024 / 1024  # 转MB
                    print(f"{model_name}：best_model_{model_name}.pth（{file_size:.2f} MB）")
                else:
                    print(f"{model_name}：best_model_{model_name}.pth（缺失）")

            print(f"\n对比结果文件：multimodal_fusion_compare.csv（存在）")
        except Exception as e:
            print(f"读取模型对比结果失败：{str(e)}")
            print(f"提示：检查multimodal_fusion_compare.csv是否被修改或损坏")
    else:
        print(f"未找到模型对比结果文件（multimodal_fusion_compare.csv）")
        print(f"提示：需先运行主程序步骤3，完成至少1个模型的训练")

    print("\n【五、步骤4 - 测试集预测结果】")
    print("-" * 50)
    submission_path = os.path.join(RESULT_DIR, "submission.csv")
    step_status = load_step_status()

    if os.path.exists(submission_path):
        try:
            submission_df = pd.read_csv(submission_path, header=None, names=["guid", "tag"])
            tag_count = submission_df["tag"].value_counts().to_dict()
            total_pred = len(submission_df)

            print(f"预测完成概况：")
            print(f"- 总预测样本数：{total_pred:>6} 条")
            print(f"- 预测结果文件：submission.csv（存在，{os.path.getsize(submission_path)/1024:.2f} KB）")

            print(f"\n预测标签分布：")
            print(f"{'情感标签':<10} {'数量':<8} {'占比':<8}")
            print(f"{'-'*10:<10} {'-'*8:<8} {'-'*8:<8}")
            for tag in ["positive", "neutral", "negative"]:
                count = tag_count.get(tag, 0)
                ratio = count / total_pred * 100 if total_pred > 0 else 0
                print(f"{tag:<10} {count:<8} {ratio:<8.1f}%")

            # 尝试关联最佳模型（如果有对比结果）
            if os.path.exists(compare_csv_path):
                compare_df = pd.read_csv(compare_csv_path)
                best_model_name = compare_df.sort_values("val_f1", ascending=False).iloc[0]["model_name"]
                print(f"\n预测关联模型：{best_model_name}（推测，基于最佳验证集F1）")
        except Exception as e:
            print(f"读取预测结果失败：{str(e)}")
            print(f"提示：检查submission.csv格式是否正确（如是否被手动修改）")
    else:
        print(f"未找到测试集预测结果文件（submission.csv）")
        if step_status["step4_predict"]:
            print(f"步骤状态标记为'已完成'，但预测文件缺失（可能被删除）")
        else:
            print(f"提示：需先运行主程序步骤4，生成测试集预测结果")

    # 报告总结
    print("\n【六、报告总结】")
    print("-" * 50)

    completed_steps = 0
    step_names = [
        ("步骤1：数据集拆分", os.path.exists(dataset_cache_path)),
        ("步骤2：超参数搜索", os.path.exists(hyper_csv_path) and os.path.exists(best_hyper_path)),
        ("步骤3：多模型训练", os.path.exists(compare_csv_path)),
        ("步骤4：测试集预测", os.path.exists(submission_path))
    ]

    print(f"已完成的步骤：")
    for step_name, is_completed in step_names:
        if is_completed:
            completed_steps += 1
            print(f"   - {step_name}")

    print(f"\n未完成的步骤：")
    has_uncompleted = False
    for step_name, is_completed in step_names:
        if not is_completed:
            has_uncompleted = True
            print(f"   - {step_name}")
    if not has_uncompleted:
        print(f"- 无（所有步骤均已完成）")

    print(f"\n报告生成完成！")
    print(f"提示：可复制此报告到实验文档，或截图存档")
    print("=" * 80)

if __name__ == "__main__":
    print_history_report()