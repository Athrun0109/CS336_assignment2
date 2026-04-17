import math
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention

def main():
    for d_model in [16, 32, 64, 128]:
        for seq_len in [256, 1024, 4096, 8192]:
            except_flag = False
            try:
                K = torch.randn(8, seq_len, d_model, device="cuda", requires_grad=True)
                Q = torch.randn(8, seq_len, d_model, device="cuda", requires_grad=True)
                V = torch.randn(8, seq_len, d_model, device="cuda", requires_grad=True)
                mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool))

                # warmup
                for i in range(5):
                    attn_output = scaled_dot_product_attention(K, Q, V, mask=mask)
                    loss = torch.sum(attn_output)
                    loss.backward()

                forward_time = backward_time = 0
                # 测试forward 100次时间
                for i in range(100):
                    torch.cuda.synchronize()
                    time1 = timeit.default_timer()
                    attn_output = scaled_dot_product_attention(K, Q, V, mask=mask)
                    loss = torch.sum(attn_output)
                    torch.cuda.synchronize()
                    time2 = timeit.default_timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    forward_time += time2 - time1
                    backward_time += timeit.default_timer() - time2
                print(f"Forward consumed time with d_model={d_model}, seq_len={seq_len}: {forward_time}")
                print(f"Backward consumed time with d_model={d_model}, seq_len={seq_len}: {backward_time}")

                # 测backward前的显存占用
                torch.cuda.reset_peak_memory_stats()
                attn_output = scaled_dot_product_attention(K, Q, V, mask=mask)
                loss = torch.sum(attn_output)
                mem_before_backward = torch.cuda.max_memory_allocated()
                print(f"Forward consumed memory with d_model={d_model}, seq_len={seq_len}: {mem_before_backward}")

            except torch.cuda.OutOfMemoryError:
                print(f"OOM: d_model = {d_model}, seq_len = {seq_len}")
                except_flag = True

            finally:
                torch.cuda.empty_cache()
                if except_flag:
                    break

if __name__ == '__main__':
    main()
