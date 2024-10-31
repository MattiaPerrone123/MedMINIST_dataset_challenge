import os
import torch

def setup_device(gpu_ids):
    """
    Configures GPU devices based on given IDs, defaults to CPU if no valid GPU is available
    """
    str_ids = gpu_ids.split(',')
    gpu_list = [int(id) for id in str_ids if int(id) >= 0]
    if gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[0])
    return torch.device(f'cuda:{gpu_list[0]}' if gpu_list else 'cpu')
