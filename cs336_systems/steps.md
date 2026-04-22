● 你的步骤只覆盖了 kernel
  内部的主循环逻辑，但缺少一些整体框架和细节。我补充一下：

  📋 建议补全的步骤

  Step 0 — 整体架构

  需要写两层：
  - Triton kernel（@triton.jit 装饰的函数）— 真正跑在 GPU 上的并行代码
  - 外部 Python wrapper（torch.autograd.Function）— 负责 launch kernel、算
      grid、保存 ctx

  Step 1 — Kernel 签名要想清楚

  @triton.jit
  def flash_fwd_kernel(
      Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,           # 5 个 raw pointer
      stride_qb, stride_qq, stride_qd,              # Q 每一维的 stride
      stride_kb, stride_kk, stride_kd,              # K 的 stride
      stride_vb, stride_vk, stride_vd,              # V 的 stride
      stride_ob, stride_oq, stride_od,              # O 的 stride
      stride_lb, stride_lq,                         # L 的 stride
      N_QUERIES, N_KEYS,                            # 运行时参数
      scale,                                        # 1/sqrt(d)
      D: tl.constexpr,                              # 编译期常量
      Q_TILE_SIZE: tl.constexpr,
      K_TILE_SIZE: tl.constexpr,
      is_causal: tl.constexpr,                      # 注意是 constexpr
  ):

  关键点： tl.constexpr 表示该值在 kernel 编译时是已知常量（tile
  尺寸、D、is_causal）；stride/尺寸是运行时参数。

  Step 2 — Grid 和 Program ID

  # 外部 launch
  grid = (Tq, batch_size)
  flash_fwd_kernel[grid](...)

  # Kernel 内部
  query_tile_idx = tl.program_id(0)   # 我是处理哪个 Q tile
  batch_idx      = tl.program_id(1)   # 我是第几个 batch
  每个 program 处理一个 batch 中的一个 Q tile（对应你 (a)
  里的双重循环的外层）。

  Step 3 — Block Pointer 初始化（你 Step 1 的细化）

  关键陷阱： 要手动把 batch 的偏移加到 base pointer 上：

  Q_block_ptr = tl.make_block_ptr(
      base=Q_ptr + batch_idx * stride_qb,          # ⬅ 加 batch 偏移
      shape=(N_QUERIES, D),
      strides=(stride_qq, stride_qd),
      offsets=(query_tile_idx * Q_TILE_SIZE, 0),
      block_shape=(Q_TILE_SIZE, D),
      order=(1, 0),
  )

  - Q_block_ptr 和 O_block_ptr 初始 offset 有 query_tile_idx * Q_TILE_SIZE
  - K_block_ptr 和 V_block_ptr 初始 offset 是 (0, 0)，靠循环里
    .advance((K_TILE_SIZE, 0)) 推进
  - L_block_ptr 是 1D 的，shape=(N_QUERIES,)，offset=(query_tile_idx *
    Q_TILE_SIZE,)

  Step 4 — 累加变量用 fp32

  m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
  l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
  O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

  Step 5 — 主循环（你的 Step 3 补几个点）

  Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  # 预先 load 一次 Q，整个循环中 Q_i 都不变

  for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
      K_j = tl.load(K_block_ptr, boundary_check=(0, 1),
  padding_option="zero")
      V_j = tl.load(V_block_ptr, boundary_check=(0, 1),
  padding_option="zero")

      S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale    # 用乘法，避免除法
    
      # 【补充】causal mask（如果 is_causal=True）
      # 对 S_ij 中 col > row 的位置设成 -inf
    
      m_new = tl.maximum(m_i, tl.max(S_ij, axis=-1))
      P_ij  = tl.exp(S_ij - m_new[:, None])
      alpha = tl.exp(m_i - m_new)
      l_i   = alpha * l_i + tl.sum(P_ij, axis=-1)
      O_i   = alpha[:, None] * O_i + tl.dot(P_ij.to(V_j.dtype), V_j)
      m_i   = m_new
    
      K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
      V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

  要点：
  - boundary_check=(0, 1) + padding_option="zero" 处理 N 不能被 tile
    整除的情况
  - tl.dot 的输入 dtype 要匹配（P_ij 可能是 fp32，V_j 可能是 fp16）
  - scale 作为参数传入，而不是在 kernel 里算 sqrt

  Step 6 — 循环结束后的写回（你遗漏的步骤）

  O_i = O_i / l_i[:, None]
  L_i = m_i + tl.log(l_i)

  tl.store(O_block_ptr, O_i.to(O_ptr.dtype.element_ty), boundary_check=(0,
   1))
  tl.store(L_block_ptr, L_i, boundary_check=(0,))

  Step 7 — 外部 wrapper（也是你遗漏的）

  class FlashAttention2Triton(torch.autograd.Function):
      @staticmethod
      def forward(ctx, Q, K, V, is_causal=False):
          batch, Nq, D = Q.shape
          _, Nk, _ = K.shape
          Q_TILE, K_TILE = 16, 16      # 或 32/64，先从 16 开始

          O = torch.empty_like(Q)
          L = torch.empty((batch, Nq), dtype=torch.float32,
  device=Q.device)

          grid = (triton.cdiv(Nq, Q_TILE), batch)
          flash_fwd_kernel[grid](
              Q, K, V, O, L,
              Q.stride(0), Q.stride(1), Q.stride(2),
              K.stride(0), K.stride(1), K.stride(2),
              # ... V/O/L 的 stride
              Nq, Nk,
              1.0 / math.sqrt(D),
              D=D, Q_TILE_SIZE=Q_TILE, K_TILE_SIZE=K_TILE,
              is_causal=is_causal,
          )
          ctx.save_for_backward(Q, K, V, O, L)
          ctx.is_causal = is_causal
          return O

  Step 8 — 调试顺序建议

  1. 先忽略 is_causal，跑通非 causal 的 forward
  2. 跑测试 pytest tests/ -k flash_forward_pass_triton 的 [False] 用例
  3. 再加 causal mask，跑 [True] 用例
