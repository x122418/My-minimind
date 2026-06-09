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

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


class PretrainDataset(Dataset):
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
        input_ids = torch.tensor(input_ids, dtype = torch.long)
    
    # 自行编写labels, prevent pad being computed in loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
    # 编写attn_mask, 标记有效位置和PAD  (非PAD为1 PAD为0)
        # attn_mask = (input_ids != self.tokenizer.pad_token_id).long()
    # 返回input_id, attn_mask, labels

        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 1024): 
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split = "train")
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens = False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens = False).input_ids
    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        result = []
        for msg in conversations:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        result.append("<|im_start|>assistant\n")
        return "\n".join(result)


    def generate_labels(self, input_ids):
        # 让所有input_id 都为 -100
        labels = [-100] * len(input_ids)
        i = 0
        while i<len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 向后扫描 找到eos位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将中间label改为有效
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            # 如果没有找到bos 则继续向后寻找
            else:
                i += 1
        return labels
    def __getitem__(self, index):
        sample = self.samples[index]
        # 是否需要添加随机syetem_prompt
        conversations = pre_processing_chat(sample['conversations'])
        # 用chat_template 把对话转为文本
        prompt = self.create_chat_prompt(conversations)
        # 清理think块
        prompt = post_processing_chat(prompt)
        # tokenizer 截断 补充pad
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id]*(self.max_length - len(input_ids))
        # 生成label 只让assistant加入loss计算
        labels=self.generate_labels(input_ids=input_ids)
        attention_mask = torch.tensor(
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), attention_mask

class RLAIDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length = 1024):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = load_dataset("json", data_files = json_path, split = "train")
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens = False
        ).input_ids
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens = False
        ).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i%2 ==0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"]
