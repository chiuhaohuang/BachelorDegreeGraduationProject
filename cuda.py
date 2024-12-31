# 此文件用于测试PyTorch CUDA是否可用
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
print(torch.cuda.is_available())
