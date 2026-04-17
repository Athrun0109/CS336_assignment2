import math

import torch

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 注意Nq与Nk可能不相同！
        batch_size, Nq, d = Q.shape
        batch_size, Nk, d = K.shape
        Bq = 16
        Bk = 16
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        # 初始化全局存储变量O, L
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, Nq), dtype=torch.float32, device=Q.device)
        for b in range(batch_size):
            for i in range(Tq):
                # 这里有个陷阱，Nq不一定能被Bq整除，因此最后一个tile可能并不是Bq的长度，构建Oi、li、mi不能直接用Bq！
                start = Bq * i
                end = min(start + Bq, Nq)
                Qi = Q[b, start:end, :]
                Oi = torch.zeros((end-start, d), dtype=torch.float32, device=Q.device)
                li = torch.zeros((end-start,), dtype=torch.float32, device=Q.device)
                mi = torch.full((end-start,), -math.inf, dtype=torch.float32, device=Q.device)
                for j in range(Tk):
                    idx = Bk * j
                    Kj = K[b, idx:idx+Bk, :]
                    Vj = V[b, idx:idx+Bk, :]
                    Si = Qi @ Kj.T / math.sqrt(d) # shape=(Bq, Bk)
                    # 计算到当前位置Si每列的最大值
                    # 关于torch.max，输入dim参数后会返回values、indices这两项内容；如果不加dim参数，就返回单一最大值的tensor。
                    mi_new = torch.max(mi, torch.max(Si, dim=-1).values)
                    Pi = torch.exp(Si - mi_new.unsqueeze(-1)) # shape=(Bq, Bk)
                    li = torch.exp(mi - mi_new) * li + torch.sum(Pi, dim=-1)
                    temp = torch.exp(mi - mi_new) # shape=(Bq,)
                    Oi = temp.unsqueeze(-1) * Oi + Pi @ Vj # shape=(Bq, d)
                    mi = mi_new # 更新mi
                Oi = Oi / li.unsqueeze(-1) # shape=(Bq, d)
                Li = mi + torch.log(li) # shape=(Bq,)
                # 将O、L写回全局内存
                O[b, start:end] = Oi
                L[b, start:end] = Li
            
        # 保存变量用于反向传播
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        pass
        raise NotImplementedError("Backward pass is not required currently.")