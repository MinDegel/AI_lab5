import os
import torch

PROJECT_ROOT = r"D:\111son\uni3.1\AI\lab5\project5"
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
TRAIN_LABEL_PATH = os.path.join(PROJECT_ROOT, "train.txt")
TEST_LABEL_PATH = os.path.join(PROJECT_ROOT, "test_without_label.txt")

os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL_NAME = "bert-base-uncased"
IMAGE_MODEL_NAME = "resnet50"
NUM_CLASSES = 3  # positive/neutral/negative
MAX_LEN = 32

BASE_BATCH_SIZE = 16
BASE_EPOCHS = 6
BASE_LEARNING_RATE = 1e-5
BASE_DROPOUT_RATE = 0.4

# 超参数搜索网格
PARAM_GRID = {
    "batch_size": [16],
    "lr": [1e-5, 1.5e-5], 
    "dropout_rate": [0.4, 0.5],
    "epochs": [6]
}