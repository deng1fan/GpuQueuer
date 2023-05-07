from queuer import set_config_gpus

class Config:

    use_gpu = True
    wait_gpus = True  # 是否愿意接受排队等待
    cuda_max_memory_utilization = 0.2  # nvitop的gpu最大内存使用阈值
    cuda_min_free_memory = "35GiB"  # nvitop的gpu最大内存使用量
    visible_cuda = 'auto_select_1'  # 使用“auto_select_[想要使用的GPU数量]”前缀自动选择可用GPU，或者使用列表指定GPU
    # visible_cuda = [1, 2, 3, 6]  # 使用“auto_select_[想要使用的GPU数量]”前缀自动选择可用GPU，或者使用列表指定GPU

    # 以下为自动调整参数，无需手动改
    default_device = "cuda:0"  # 程序自动调整，默认的设备
    task_id = None  # 程序自动调整，如果选择等待GPU，那么这将是排队的号，此处无需填写，由程序自动生成
    confirm_gpu_free = False  # 程序自动调整，用于标识当前训练任务是否已经确认了GPU出于空闲，如果两次都等到了相同的GPU那么就认为该GPU空闲
    last_confirm_gpus = None  # 程序自动调整，记录第一次确认空闲的gpus

if __name__ == '__main__':
    config = Config()
    config = set_config_gpus(config)
    print()