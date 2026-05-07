import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 日志
def Logger(content):
    if is_main_process():
        print(content)


# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi + current_step / total_steps)))

# 初始化分布式
def init_distributed_mode():
    if int(os.environ.get('RANK', -1)) == -1:
        return 0
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# 设置种子
def setup_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False