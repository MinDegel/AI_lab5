import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import pandas as pd
from config import *

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_bert_tokenizer():
    return BertTokenizer.from_pretrained(TEXT_MODEL_NAME)

class MultimodalDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, label_path=TRAIN_LABEL_PATH, 
                 transform=None, tokenizer=None, max_len=MAX_LEN, is_test=False):
        self.data_dir = data_dir
        self.transform = transform or get_image_transform()
        self.tokenizer = tokenizer or get_bert_tokenizer()
        self.max_len = max_len
        self.is_test = is_test
        
        self.df = pd.read_csv(label_path, header=0, names=["guid", "tag"])
        self.label_map = {"positive": 0, "neutral": 1, "negative": 2}
        if not is_test:
            self.df["tag"] = self.df["tag"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = self.df.iloc[idx]["guid"]
        guid = str(int(guid)).strip()
        guid = ''.join(filter(str.isdigit, str(guid)))
        
        img_path_lower = os.path.join(self.data_dir, f"{guid}.jpg")
        img_path_upper = os.path.join(self.data_dir, f"{guid}.JPG")
        img_path = img_path_lower if os.path.exists(img_path_lower) else img_path_upper
        
        text_path_lower = os.path.join(self.data_dir, f"{guid}.txt")
        text_path_upper = os.path.join(self.data_dir, f"{guid}.TXT")
        text_path = text_path_lower if os.path.exists(text_path_lower) else text_path_upper
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件不存在：{img_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"文本文件不存在：{text_path}")
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        with open(text_path, "r", encoding="latin-1") as f:
            text = f.read().strip()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return_dict = {
            "image": image,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "guid": guid
        }
        
        if not self.is_test:
            label = torch.tensor(self.df.iloc[idx]["tag"], dtype=torch.long)
            return_dict["label"] = label
        
        return return_dict