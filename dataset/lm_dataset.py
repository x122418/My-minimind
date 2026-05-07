from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 全局预处理 / 后处理工具函数
# ──────────────────────────────────────────────────────────────────────────────

class PretrainDateset(Dataset):
    # init
    def __init__(self, data_path, tokenizer, max_length = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
    # __len__
    def __len__(self):
        return len(self.samples)
    # __getitem__
    def __getitem__(self, index:int):
        
    # 拿到json中的每一行
        sample = self.samples[index]
    # tokenizer转化字符串为token_id
        tokens = self.tokenizer((sample['text']), 
                                add_special_tokens = False,
                                max_length = self.max_length - 2,
                                truncation = True).input_ids #假设每一行都有text字段 表示文本
    # 加入PAD，EOS，BOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.Tensor(input_ids, dtype = torch.long)
    
    # 自行编写labels, prevent pad being computed in loss
        labels = input_ids.clone()
        labels = tokens[tokens == self.tokenizer.pad_token_id] = -100
    # 编写attn_mask, 标记有效位置和PAD  (非PAD为1 PAD为0)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()
    # 返回input_id, attn_mask, labels
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': labels
        }