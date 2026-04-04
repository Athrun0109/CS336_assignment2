import os
import timeit
import argparse
import math
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_cosine_lr, AdamW
from cs336_basics.nn_utils import softmax

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def annotated_scaled_dot_product_attention(Q, K, V, mask):
    '''
    Replace scaled_dot_product_attention with added nvtx function
    '''
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, -math.inf)

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("computing final matmul"):
        ret = attention_weights @ V

    return ret

# 函数替换
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def main(args):
    if args.mixed_precision == 'fp16':
        autocast = torch.autocast(device_type='cuda', dtype=torch.float16)
    elif args.mixed_precision == 'bf16':
        autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        autocast = nullcontext()

    model = BasicsTransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta
    )
    model.to(DEVICE)

    # 注意输入的tensor需要时int类型
    input_tensor = torch.randint(0, args.vocab_size, size=(args.batch_size, args.context_length)).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    step_times = []

    # train loop
    for i in range(args.warmup_steps + args.num_steps):

        # start timer
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        # with mixed precision, model forward+loss

        # 测试显存占用情况（跳过warmup阶段）
        if i == args.warmup_steps:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        with autocast:
            output_tensor = model.forward(input_tensor)
            # dummy loss function
            loss = torch.mean(output_tensor)

        if not args.forward_only:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        if i >= args.warmup_steps:
            step_times.append(end_time - start_time)

    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{args.mixed_precision}_{args.forward_only}_ctx{args.context_length}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"== Execution time with warmup steps: {args.warmup_steps} ==")
    step_times = torch.tensor(step_times)
    mean = torch.mean(step_times)
    std = torch.std(step_times)
    print(f"Average of timings over {args.num_steps} measurement steps: {mean}")
    print(f"Standard deviation of timings over 10 measurement steps: {std}")

if __name__ == "__main__":
    def clearfy_true_false(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif s.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(description='benchmarking script')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size.')
    parser.add_argument('--context_length', type=int, default=64, help='Context length.')
    parser.add_argument('--d_model', type=int, default=768, help='Embedding dimension.')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers.')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of heads.')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension.')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta value.')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--num_steps', type=int, default=10, help='Measurement steps.')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Warm up learning rate.')
    parser.add_argument('--forward_only', type=clearfy_true_false, default=False, help='Either only forward, or both forward and backward passes.')

    parser.add_argument('--mixed_precision', type=str, default='none', choices=['none', 'fp16', 'bf16'], help='Autocast precision.')

    args = parser.parse_args()

    args.d_model = 2560
    args.d_ff = 10240
    args.num_layers = 32
    args.num_heads = 32
    # 测试内存占用，只跑一个epoch
    args.num_steps = 1

    args.mixed_precision = 'bf16'

    # Test time with warmup = 5
    main(args)

    # # Test time with warmup = 0
    # args.warmup_steps = 0
    # main(args)
