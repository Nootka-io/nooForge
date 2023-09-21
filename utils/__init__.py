from pynvml import *


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


def print_gpu_utilization():
    '''prints how much gpu mem is utilized by nvidia-smi'''
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_training_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
