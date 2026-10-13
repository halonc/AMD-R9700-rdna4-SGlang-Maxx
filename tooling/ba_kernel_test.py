# in_proj_ba 自写 kernel 的离线对拍 + 计时：x[M,5120] @ W[48,5120]^T -> bf16
import os, time
os.environ.setdefault("SGLANG_USE_AITER", "0")
import torch, triton, triton.language as tl

K, N = 5120, 48

@triton.jit
def _ba_kernel(X, W, Out, M, sx, K: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    n = tl.program_id(0)
    mb = tl.program_id(1)
    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_m[:, None] * sx + offs_k[None, :],
                    mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
        w = tl.load(W + n * K + offs_k).to(tl.float32)
        acc += tl.sum(x * w[None, :], axis=1)
    tl.store(Out + offs_m * N + n, acc.to(Out.dtype.element_ty), mask=offs_m < M)


def ba_gemm(x, w, out):
    M = x.shape[0]
    BM = 16
    _ba_kernel[(N, (M + BM - 1) // BM)](x, w, out, M, x.stride(0), K, N, BM, 1024)
    return out


torch.manual_seed(0)
W = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
ok = True
for M in (1, 9, 16, 17, 64):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, W)
    got = ba_gemm(x, W, torch.empty(M, N, device="cuda", dtype=torch.bfloat16))
    rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    exact = torch.equal(got, ref)
    print("M=%-3d rel=%.5f  bitexact=%s" % (M, rel, exact))
    ok = ok and rel < 2e-3

def timeit(fn, reps=200, burst=16):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps // burst):
        for _ in range(burst):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6

for M in (9, 16):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    t_ref = timeit(lambda: torch.nn.functional.linear(x, W))
    t_new = timeit(lambda: ba_gemm(x, W, out))
    print("M=%-3d  F.linear %6.1f us   自写 kernel %6.1f us   -> 每步 48 次省 %.2f ms"
          % (M, t_ref, t_new, (t_ref - t_new) * 48 / 1e3))
print("数值 OK" if ok else "数值不对")
