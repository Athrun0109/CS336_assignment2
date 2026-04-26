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
                mi = torch.full((end-start,), -1e6, dtype=torch.float32, device=Q.device)
                for j in range(Tk):
                    idx = Bk * j
                    Kj = K[b, idx:idx+Bk, :]
                    Vj = V[b, idx:idx+Bk, :]
                    Si = Qi @ Kj.T / math.sqrt(d) # shape=(Bq, Bk)
                    # causal_mask
                    if is_causal:
                        Q_mask_base = i * Bq
                        Q_mask = Q_mask_base + torch.arange(math.min(Bq, Nq - Q_mask_base), device=Si.device)
                        K_mask_base = j * Bk
                        K_mask = K_mask_base + torch.arange(math.min(Bk, Nk - K_mask_base), device=Si.device)
                        causal_mask_reverse = Q_mask[..., None] < K_mask[None, ...]
                        Si = Si.masked_fill(causal_mask_reverse, -1e6)
                    # 计算到当前位置Si每行的最大值
                    # 关于torch.max，输入dim参数后会返回values、indices这两项内容；如果不加dim参数，就返回单一最大值的tensor。
                    mi_new = torch.max(mi, torch.max(Si, dim=-1).values)
                    Pi = torch.exp(Si - mi_new.unsqueeze(-1)) # shape=(Bq, Bk)
                    li = torch.exp(mi - mi_new) * li + torch.sum(Pi, dim=-1)
                    Oi = torch.exp(mi - mi_new).unsqueeze(-1) * Oi + Pi @ Vj # shape=(Bq, d)
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
        '''
        grad_out.shape = (batch_size, Nq, d)
        '''
        L, Q, K, V, O = ctx.saved_tensors
        d = Q.shape[-1]
        scale = 1 / math.sqrt(d)
        S = Q @ K.transpose(-1, -2) * scale # shape=(batch_size, Nq, Nk)
        if ctx.is_causal:
            Nq, Nk = S.shape[-2:]
            Q_mask = torch.arange(Nq, device=S.device)
            K_mask = torch.aragne(Nk, device=S.device)
            causal_mask_reverse = Q_mask[..., None] < K_mask[None, ...] # shape=(Nq, Nk)
            S = S.masked_fill(causal_mask_reverse, -1e6)
        P = torch.exp(S - L.unsqueeze(-1)) # shape=(batch_size, Nq, Nk)
        dV = P.transpose(-1, -2) @ grad_out # shape=(batch_size, Nk, d)
        dP = grad_out @ V.transpose(-1, -2) # shape=(batch_size, Nq, Nk)
        # 注意这里是逐元素相乘，而不是矩阵乘法：rowsum(P ◦ dP)
        D = torch.sum(P * dP, dim=-1, keepdim=True) # shape=(batch_size, Nq, 1)
        temp = dP - D # shape=(batch_size, Nq, Nk)
        dS = P * temp # shape=(batch_size, Nq, Nk)
        dQ = dS @ K * scale # shape=(batch_size, Nq, d)
        dK = dS.transpose(-1, -2) @ Q * scale # shape=(batch_size, Nk, d)

        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    Q_stride_b, Q_stride_row, Q_stride_dim,
    K_stride_b, K_stride_row, K_stride_dim,
    V_stride_b, V_stride_row, V_stride_dim,
    O_stride_b, O_stride_row, O_stride_dim,
    L_stride_b, L_stride_row,
    Nq, Nk, scale,
    D: tl.constexpr,
    Bq: tl.constexpr, Bk: tl.constexpr,
    is_causal: tl.constexpr
):
    # 注意：下面给出的不是i、j，而是以tile_size为单位的第x th个
    Q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_base = batch_idx * Q_stride_b
    K_base = batch_idx * K_stride_b
    V_base = batch_idx * V_stride_b
    O_base = batch_idx * O_stride_b
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
        V_ptr + V_base,
        shape=(Nk, D),
        strides=(V_stride_row, V_stride_dim),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + O_base,
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
    # Triton中跨越循环的临时变量需要事先声明
    Oi = tl.zeros((Bq, D), dtype=tl.float32)
    li = tl.zeros((Bq,), dtype=tl.float32)
    mi = tl.full((Bq,), -float('inf'), dtype=tl.float32)

    # 在下面的循环中，Q的指针并不会被移动，所以直接在循环前加载Q
    Q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    for i in range(tl.cdiv(Nk, Bk)):
        Ki = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        Vi = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        # 计算Q@K的score值
        Si = tl.dot(Q, tl.trans(Ki)) * scale
        # casual_mask
        if is_causal:
            Q_mask = Q_tile_idx * Bq + tl.arange(0, Bq) # shape=(Bq,)
            K_mask = i * Bk + tl.arange(0, Bk) # shape=(Bk,)
            causal_mask = Q_mask[..., None] >= K_mask[None, ...] # shape=(Bq, Bk)
            Si = tl.where(causal_mask, Si, -1e6)
        # 更新mi的最大值；max用于矩阵内部比较，maximum用于两个矩阵进行比较
        mi_new = tl.maximum(mi, tl.max(Si, axis=-1)) # shape=(Bq,)
        Pi = tl.exp(Si - mi_new[..., None]) # shape=(Bq, Bk)
        # 更新softmax的分母
        li = tl.exp(mi - mi_new) * li + tl.sum(Pi, axis=-1) # shape=(Bq,)
        # 更新softmax的分子
        temp = tl.exp(mi - mi_new) # shape=(Bq,)
        # 作业建议使用tl.dot(acc=)参数完成累加操作
        Oi = temp * Oi
        Oi = tl.dot(Pi.to(Vi.dtype), Vi, acc=Oi)
        mi = mi_new

        # 移动K, V指针
        K_block_ptr = tl.advance(K_block_ptr, (Bk, 0))
        V_block_ptr = tl.advance(V_block_ptr, (Bk, 0))
    
    Oi = Oi / li[..., None] # shape=(Bq, D)
    Li = mi + tl.log(li) # shape=(Bq,)

    # 将Oi、Li写入block_ptr；要求中有提到"cast Oi to the appropriate dtype before writing it to global memory"
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        bz, Nq, D = Q.shape
        _, Nk, _ = K.shape
        scale = 1.0 / D ** 0.5
        
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors."
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous"

        ctx.D = D
        ctx.Bq = ctx.Bk = 16
        ctx.is_causal = is_causal
        
        # 初始化O和L的empty tensor
        O = torch.empty_like(Q)
        L = torch.empty((bz, Nq), dtype=torch.float32, device=Q.device)
        
        grid = ((Nq + ctx.Bq - 1)//ctx.Bq, bz) # 数学上的向上取整
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            Nq, Nk, scale,
            ctx.D,
            ctx.Bq, ctx.Bk,
            ctx.is_causal
        )

        ctx.save_for_backward(L, Q, K, V, O)

        return O