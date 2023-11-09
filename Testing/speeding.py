import time
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from models.bisenet.bisenet import BiSeNet
from models.erfnet.erfnet import ERFNet
from models.fanet.fanet import FANet
from models.icnet.icnet import ICNet
from models.segnet.segnet import SegNet
from models.shelfnet.shelfnet import ShelfNet18
from models.swiftnet.semseg import SwiftNet


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def latency_test(model: nn.Module, size: Tuple[int, ...] = (1, 3, 1024, 2048), dtype: str = 'fp32', nwarmup: int = 50,
                 nruns: int = 1000, verbose: bool = True):
    x = torch.randn(size, device='cuda')
    if dtype == 'fp16':
        x = x.half()

    if verbose:
        print('Warm up')
    with torch.no_grad():
        for _ in range(nwarmup):
            _ = model(x)
    torch.cuda.synchronize()
    if verbose:
        print('Benchmarking')
    timings = []
    with torch.no_grad():
        for _ in range(nruns):
            start_time = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

    return np.mean(timings)


def calculate_flops(model: torch.nn.Module, size: Tuple[int, ...] = (1, 3, 1024, 2048)):
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model.eval()
    x = torch.randn(size).cuda()
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops, max_depth=1))


def run(model, size, name):
    model.cuda()
    model.eval()
    t_cnt = 0.0
    with torch.no_grad():
        input = torch.rand(size).cuda()
        torch.cuda.synchronize()
        x = model(input)
        x = model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_ts = time.time()

        for i in range(100):
            x = model(input)
        torch.cuda.synchronize()
        end_ts = time.time()

        t_cnt = end_ts - start_ts  # t_cnt + (end_ts-start_ts)
    print("=======================================")
    print("Model Name: " + name)
    print("FPS: %f" % (100 / t_cnt))
    # print("=======================================")


if __name__ == "__main__":
    # segnet = SegNet()
    # run(segnet, size=(1, 3, 360, 640), name='SegNet')
    #
    # icnet = ICNet()
    # run(icnet, size=(1, 3, 1025, 2049), name='ICNet')
    #
    # erfnet = ERFNet()
    # run(erfnet, size=(1, 3, 512, 1024), name='ERFNet')
    #
    # bisenet = BiSeNet()
    # run(bisenet, size=(1, 3, 768, 1536), name='BiSeNet')
    #
    # shelfnet = ShelfNet18()
    # run(shelfnet, size=(1, 3, 1024, 2048), name='ShelfNet')
    #
    # swiftnet = SwiftNet()
    # run(swiftnet, size=(1, 3, 1024, 2048), name='SwiftNet')

    fanet18 = FANet(backbone='resnet18').cuda().eval()
    fanet34 = FANet(backbone='resnet34').cuda().eval()
    for s in [(256, 512), (384, 768), (512, 1024), (768, 1536), (1024, 2048)]:
        print(s)
        t1, t2 = 0, 0
        for _ in range(5):
            t1 += latency_test(fanet18, (1, 3, s[0], s[1]), verbose=False) / 5
            t2 += latency_test(fanet34, (1, 3, s[0], s[1]), verbose=False) / 5
        calculate_flops(fanet18, (1, 3, s[0], s[1]))
        print(t1)
        calculate_flops(fanet34, (1, 3, s[0], s[1]))
        print(t2)
