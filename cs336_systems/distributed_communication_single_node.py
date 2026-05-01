import os
import time

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_one_config(rank, world_size, backend, device, size_bytes, n_warmup, n_iter, return_q):
    # setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if device == "cuda":
        torch.cuda.set_device(rank)
    data = torch.empty((size_bytes // 4), dtype=torch.float32, device=device)

    # warmup
    for _ in range(n_warmup):
        dist.all_reduce(data)
        if device=="cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        if device=="cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(data)
        if device=="cuda":
            torch.cuda.synchronize() # 该函数有“等待所有cuda操作完成”的作用，使得cuda代码在同步时不会出错。
        times.append(time.perf_counter() - t0)

    # gather各rank的均值到rank 0
    gathered = [None] * world_size # len(gathered)必须等于world_size
    # 每个 rank 把自己的均值贡献出去，all_gather_object 让所有 rank 都拿到一份按rank 顺序排列的完整列表
    dist.all_gather_object(gathered, sum(times) / len(times))
    if rank == 0:
        return_q.put((backend, world_size, size_bytes, gathered))

    dist.destroy_process_group()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    return_q = ctx.Queue()

    sizes = [1, 10, 100, 1000] # MB
    scale = 1024 ** 2
    world_sizes = [2, 4, 6]
    # configs = [("gloo", "cpu"), ("nccl", "cuda")]
    configs = [("gloo", "cpu")]

    for backend, device in configs:
        for ws in world_sizes:
            for mb in sizes:
                size = mb * scale
                mp.spawn(
                    fn=benchmark_one_config,
                    args=(ws, backend, device, size, 5, 10, return_q),
                    nprocs=ws,
                    join=True
                )

    results = []
    for _ in range(len(configs) * len(world_sizes) * len(sizes)):
        results.append(return_q.get())

    df = pd.DataFrame(results, columns=["backend", "world_size", "size_bytes", "rank_times"])     
    df["mean_ms"]  = df["rank_times"].apply(lambda xs: 1000 * sum(xs) / len(xs))                  
    df["max_ms"]   = df["rank_times"].apply(lambda xs: 1000 * max(xs))   
    # 短板时间，更接近真实通信耗时                                                                  
    print(df.to_string(index=False))                                                              
    df.to_csv("results.csv", index=False)  