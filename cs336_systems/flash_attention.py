import math

import torch
import triton
import triton.language as tl

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
                    # 计算到当前位置Si每行的最大值
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
    

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    Q_stride_b, Q_stride_row, Q_stride_dim,
    K_stride_b, K_stride_row, K_stride_dim,
    V_stride_b, V_stride_row, V_stride_dim,
    O_stride_b, O_stride_row, O_stride_dim,
    L_stride_b, L_stride_row,
    batch_size, Nq, Nk, 
    D: tl.constexpr,
    Bq: tl.constexpr, Bk: tl.constexpr
):
    # 注意：下面给出的不是i、j，而是以tile_size为单位的第x th个
    Q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_base = batch_idx * Q_stride_b
    K_base = batch_idx * K_stride_b
    L_base = batch_idx * L_stride_b
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + Q_base, # 当前grid的指针开始的物理位置
        shape=(Nq, D),
        strides=(Q_stride_row, Q_stride_dim),
        offsets=(Q_tile_idx * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + K_base,
        shape=(Nk, D),
        strides=(K_stride_row, K_stride_dim),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + K_base,
        shape=(Nk, D),
        strides=(V_stride_row, V_stride_dim),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + Q_base,
        shape=(Nq, D),
        strides=(O_stride_row, O_stride_dim),
        offsets=(Q_tile_idx * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + L_base,
        shape=(Nq,),
        strides=(L_stride_row,),
        offsets=(Q_tile_idx * Bq,),
        block_shape=(Bq,),
        order=(0,),
    )

    Oi = tl.zeros((Bq, D), dtype=tl.float32)
    li = tl.zeros((Bq,), dtype=tl.float32)
    mi = tl.full((Bq,), -float('inf'), dtype=tl.float32)
