from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe import MoeRunner
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.quantization.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_moe_permute_scales,
    marlin_permute_scales,
    moe_awq_to_marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import get_scalar_types, replace_parameter
from sglang.srt.utils import is_hip, is_xpu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

awq_marlin_moe_repack = None
awq_marlin_repack = None


def _unsupported_awq_dequantize(*args, **kwargs):
    raise RuntimeError("AWQ GPU kernels are unavailable on the current platform.")


awq_dequantize = _unsupported_awq_dequantize

if is_xpu():
    try:
        from sgl_kernel import awq_dequantize
    except ImportError:
        pass
elif is_hip():
    try:
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )
    except ImportError:
        pass
else:
    try:
        from sglang.kernels.ops.quantization.awq_dequantize import awq_dequantize
        from sglang.kernels.ops.quantization.awq_marlin_repack import (
            awq_marlin_moe_repack,
            awq_marlin_repack,
        )
        from sglang.srt.utils.custom_op import register_custom_op_from_extern

        awq_dequantize = register_custom_op_from_extern(
            awq_dequantize,
            fake_impl=lambda qweight, scales, qzeros: qweight.new_empty(
                qweight.shape[:-1] + (qweight.shape[-1] * 8,), dtype=scales.dtype
            ),
        )
    except ImportError:
        try:
            from sglang.kernels.ops.quantization.awq_triton import (
                awq_dequantize_triton as awq_dequantize,
            )
        except ImportError:
            pass

_, scalar_types = get_scalar_types()

# R9700: fused int4 AWQ GEMV for M=1 decode (HIP/ROCm, BF16). Built from
# components/.../awq/awq_gemv_hip.cu via scripts/build_awq_gemv.sh. Reads int4
# weights directly instead of dequantizing the full weight to bf16 + a dense
# rocBLAS GEMM (which together were 94% of qwen36-27b decode time at ~4.5 tok/s).
try:
    from awq_gemv_hip_ext import awq_gemv_bf16_hip as _awq_gemv_bf16
    from awq_gemv_hip_ext import awq_gemv_hip as _awq_gemv_fp16
except ImportError:
    _awq_gemv_bf16 = None
    _awq_gemv_fp16 = None
    import logging as _logging

    # Fail-open detector: dense AWQ decode silently drops to dequant+rocBLAS
    # (~5x slower) when the GEMV ext is missing — make the boot log say so.
    _logging.getLogger(__name__).warning(
        "awq_gemv_hip_ext NOT importable — dense AWQ M=1 decode will use the "
        "slow dequant fallback (~5x). Build via scripts/build_awq_gemv.sh."
    )


from sglang.srt.hardware_backend.gpu.quantization import r4d_w4 as _r4d_w4


class AWQLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        scales = layer.scales.data
        # R9700: checkpoints ship FP16 scales; the fused M=1 GEMV requires scales
        # dtype == activation dtype (bf16 or fp16 entrypoints). Cast to the model
        # compute dtype (SGLang sets torch default dtype = model dtype during load);
        # without this the dispatch silently never fires (4.6 tok/s qwen36-27b).
        if _awq_gemv_bf16 is not None and scales.dtype != torch.get_default_dtype():
            scales = scales.to(torch.get_default_dtype())
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        # R9700 / libr4d: pre-permute the int4 weights into the WMMA fragment order
        # once, so the M<=64 band (decode + speculative verify) can run the
        # 4-bit-weight GEMM directly instead of dequantizing to bf16 per call.
        layer.r4d_wq = None
        if _r4d_w4.available() and getattr(layer, "qweight", None) is not None:
            K = layer.qweight.shape[0]
            N = layer.qweight.shape[-1] * 8
            if (
                N % 16 == 0
                and K % _r4d_w4.KPB == 0
                and getattr(self.quant_config, "group_size", None) == _r4d_w4.GROUP
                and _r4d_w4._best_sk(K)
            ):
                try:
                    _dev = layer.qweight.device
                    _prefix = getattr(layer, "prefix", None)
                    _K = int(layer.qweight.shape[0])
                    _N = int(layer.qweight.shape[-1] * 8)
                    _host_qw = layer.qweight.data.to("cpu")
                    _host_sc = layer.scales.data.to("cpu")
                    _host_qz = layer.qzeros.data.to("cpu")
                    _wq, _wsz = _r4d_w4.repack_cached(_host_qw, _host_sc, _prefix, "cpu")
                    # IN PLACE. The fragment pack is byte-for-byte the same size as
                    # the AWQ qweight (both 4 bits per weight), so the packed data
                    # is written back over the existing device storage and only
                    # re-viewed. process_weights_after_loading runs once the model
                    # and its KV pool are resident, so allocating a second copy
                    # would OOM a 32GB card.
                    _flat = layer.qweight.data.reshape(-1).view(torch.uint8)
                    _flat.copy_(_wq.reshape(-1).view(torch.uint8))
                    layer.r4d_wq = layer.qweight.data.view(torch.int32).reshape(
                        _N // 16, _K // 64, 32, 4
                    )
                    layer.r4d_wsz = _wsz.to(_dev).contiguous()
                    layer.r4d_K = _K
                    layer.r4d_N = _N
                    # Keep the AWQ tensors on the host: the M>64 (prefill) band
                    # uses them through the dequant+GEMM path, because the M<=64
                    # kernel would re-read every weight once per 64 rows. A
                    # second device copy does not fit on a 32GB card.
                    layer.r4d_awq_host = (_host_qw, _host_sc, _host_qz)
                    layer.scales.data = layer.scales.data.to("cpu")
                    layer.qzeros.data = layer.qzeros.data.to("cpu")
                except Exception as e:  # fail open to the existing paths
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "libr4d repack failed (%s); falling back", e
                    )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        # R9700 FIX: for M=1 decode use the fused int4 GEMV (memory-bound on int4)
        # instead of dequantizing the whole weight to bf16 + a dense GEMM. The old
        # path was ~20x under the int4 memory roofline (e.g. qwen36-27b 4.5 tok/s);
        # awq_dequantize + rocBLAS GEMM were 94% of decode time. Validated cos~1.0
        # vs that reference. M>1 (prefill/batched) and non-bf16 keep the old path.
        if (
            _r4d_w4.available()
            and getattr(layer, "r4d_wq", None) is not None
            and reshaped_x.dtype == torch.bfloat16
        ):
            if reshaped_x.shape[0] <= _r4d_w4.MAX_M:
                # decode / speculative verify: the libr4d W4A16 WMMA kernel keeps
                # the weights 4-bit, ~20x faster than awq_dequantize + matmul.
                r4d_out = _r4d_w4.gemm(
                    reshaped_x, layer.r4d_wq, layer.r4d_wsz,
                    layer.r4d_N, layer.r4d_K,
                )
                if r4d_out is not None:
                    if bias is not None:
                        r4d_out.add_(bias.to(r4d_out.dtype))
                    return r4d_out.reshape(x.shape[:-1] + (layer.r4d_N,))
            else:
                # prefill: unpack the fragment once and run the exact W4A8 int8
                # GEMM over the whole chunk (the M<=64 kernel would re-read every
                # weight once per 64 rows, ~30x the traffic).
                try:
                    _pf = _r4d_w4.w4a8_gemm(
                        reshaped_x, layer.r4d_wq, layer.r4d_wsz,
                        layer.r4d_N, layer.r4d_K,
                    )
                except Exception:
                    _pf = None
                if _pf is not None:
                    if bias is not None:
                        _pf.add_(bias)
                    return _pf.reshape(x.shape[:-1] + (layer.r4d_N,))
                # fall back to streaming the AWQ tensors from the host
                _host = getattr(layer, "r4d_awq_host", None)
                if _host is not None:
                    _qw = _host[0].to(reshaped_x.device, non_blocking=True)
                    _sc = _host[1].to(reshaped_x.device, non_blocking=True)
                    _qz = _host[2].to(reshaped_x.device, non_blocking=True)
                    if _sc.dtype != reshaped_x.dtype:
                        _sc = _sc.to(reshaped_x.dtype)
                    out = awq_dequantize(_qw, _sc, _qz)
                    out = torch.matmul(reshaped_x, out)
                    if bias is not None:
                        out.add_(bias)
                    return out.reshape(x.shape[:-1] + (layer.r4d_N,))
        # AWQ fallback: only reachable when the libr4d repack did not apply.
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        if qweight.device != reshaped_x.device:      # host-resident after the repack
            qweight = qweight.to(reshaped_x.device)
            scales = scales.to(reshaped_x.device)
            qzeros = qzeros.to(reshaped_x.device)
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        if (
            _awq_gemv_bf16 is not None
            and reshaped_x.shape[0] == 1
            and reshaped_x.dtype == torch.bfloat16
            and scales.dtype == torch.bfloat16
        ):
            out = _awq_gemv_bf16(reshaped_x, qweight, scales, qzeros, 0).reshape(1, -1)
        elif (
            _awq_gemv_fp16 is not None
            and reshaped_x.shape[0] == 1
            and reshaped_x.dtype == torch.float16
            and scales.dtype == torch.float16
        ):
            out = _awq_gemv_fp16(reshaped_x, qweight, scales, qzeros, 0).reshape(1, -1)
        else:
            if scales.dtype != reshaped_x.dtype:
                scales = scales.to(reshaped_x.dtype)
            out = awq_dequantize(qweight, scales, qzeros)
            out = torch.matmul(reshaped_x, out)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)


class AWQMarlinLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        layer.workspace = marlin_make_workspace(device)

        marlin_qweight = awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qweight", marlin_qweight)

        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "scales", marlin_scales)

        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qzeros", marlin_zp)

        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.qzeros,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias,
        )


class AWQMoEKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.runner: Optional[MoeRunner] = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_qweight.shape[0]
        device = layer.w13_qweight.device

        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

        marlin_w13_qweight = awq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            size_k=layer.w13_qweight.shape[1],
            size_n=layer.w13_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)

        marlin_w2_qweight = awq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            size_k=layer.w2_qweight.shape[1],
            size_n=layer.w2_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        marlin_w13_zp = moe_awq_to_marlin_zero_points(
            layer.w13_qzeros,
            size_k=layer.w13_qzeros.shape[1],
            size_n=layer.w13_qzeros.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qzeros", marlin_w13_zp)

        marlin_w2_zp = moe_awq_to_marlin_zero_points(
            layer.w2_qzeros,
            size_k=layer.w2_qzeros.shape[1],
            size_n=layer.w2_qzeros.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qzeros", marlin_w2_zp)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if self.runner is None:
            raise RuntimeError("moe runner is not initialized")

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_qweight,
            w2_qweight=layer.w2_qweight,
            w13_scales=layer.w13_scales,
            w2_scales=layer.w2_scales,
            w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
            w13_qzeros=layer.w13_qzeros,
            w2_qzeros=layer.w2_qzeros,
            weight_bits=self.quant_config.weight_bits,
        )
        return self.runner.run(dispatch_output, quant_info)

# --- R4D MXFP4 (path A): convert once, free the AWQ copy, ONE forward path ----------
# The AWQ checkpoint is read exactly once -> requantised to MXFP4 (E2M1 + E8M0 g32)
# -> the AWQ tensors are dropped immediately.  A converted layer therefore keeps a
# single weight representation live, and apply() has exactly one code path (no
# fallback is possible anyway: the fallback's tensors are gone).  No disk cache, no
# module-order key, no "release guard" — those existed only because the old design
# kept both copies resident at once.
#
# Bug fixed here (root cause of every "garbage output" round):
#   AWQ packs 8 output columns per int32 in the order 0,2,4,6,1,3,5,7.  Reading the
#   nibbles sequentially permutes the weight matrix — measured 108.8% away from
#   SGLang's own awq_dequantize_triton on the real checkpoint — which silently wrecks
#   the model while every downstream check still agrees (kernel vs reference, both
#   fed the same wrong weights).  Mapping reverse-solved from the production dequant.
import os as _mx_os
import torch as _mx_torch
from sglang.srt.hardware_backend.gpu.quantization import r4d_mxfp4 as _mx
from nvfp4_to_mxfp4 import to_mxfp4 as _mx_to_mxfp4

_AWQ_INTERLEAVE = (0, 4, 1, 5, 2, 6, 3, 7)

_W3 = None
if _mx_os.environ.get("SGLANG_W3", "0") == "1":
    from sglang.srt.hardware_backend.gpu.quantization import r4d_w3 as _W3
    # w3 的 chunked/prefill 要 plain [M,K] fp8 A；atiled 的 fragment-tiled A 会被静默读错。
    # norm 生产者不知道消费者是谁，所以 W3 开着时全局关掉 atiled。
    _mx.ATILED_MIN_M = 0
    print("[r4d.w3] enabled from %s (atiled disabled: A layout)" % _W3.W3_DIR, flush=True)
_MX_FREE = ("qweight", "qzeros", "scales", "r4d_wq", "r4d_wsz", "r4d_awq_host",
            "r4d_K", "r4d_N")
_MX_SELFCHECK = int(_mx_os.environ.get("SGLANG_MXFP4_SELFCHECK", "0"))
_MX_N = {"built": 0, "failed": 0}


def _mx_unpack4(p):
    """[K, N/8] int32 -> [K, N] codes, honouring the AWQ interleave."""
    K, N8 = p.shape
    b = p.view(_mx_torch.uint8).view(K, N8, 4)
    seq = _mx_torch.stack([b & 0x0F, (b >> 4) & 0x0F], dim=-1).reshape(K, N8, 8)
    return seq[..., list(_AWQ_INTERLEAVE)].reshape(K, N8 * 8)


def _mx_build(layer):
    """AWQ (qweight/qzeros/scales) -> MXFP4 packed [N,K/2] u8 + E8M0 [K/32,N] u8."""
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    w_q = _mx_unpack4(qw).to(_mx_torch.float32)
    z_q = _mx_unpack4(qz).to(_mx_torch.float32)
    K, N = w_q.shape
    g = sc.shape[0]
    dense = ((w_q.view(g, 128, N) - z_q[:, None, :])
             * sc.to(_mx_torch.float32)[:, None, :]).view(K, N).t().contiguous()
    del w_q, z_q
    pk, e8 = _mx_to_mxfp4(dense)
    return pk, e8, dense


def _mx_selfcheck(tag, dense, pk, e8):
    """MXFP4 kernel output vs the dense AWQ weight on the same x.

    This is the check the previous rounds never ran: their 0.14% figure compared the
    kernel against a reference *fed by the same MXFP4 tensors*, so it could not see a
    wrong AWQ -> MXFP4 conversion.  Expected value here is ~10% (pure requant error).
    """
    K = dense.shape[1]
    for M in (8, 2048):
        x = (torch.randn(M, K, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()
        ref = x.float() @ dense.t()
        out = _mx.mxfp4_linear(x, pk, e8, _mx.make_row_ref(e8))
        err = float((out.float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-9))
        band = "decode(r4d M<=%d)" % _mx.R4D_DECODE_MAX_M if M <= _mx.R4D_DECODE_MAX_M else "prefill(folded)"
        print("[r4d.mxfp4] selfcheck [%s] %s: kernel vs 稠密 AWQ 权重 = %.3f%% (期望 ~10~11%%)"
              % (band, tag, err * 100), flush=True)


def _mx_process(layer):
    if getattr(layer, "qweight", None) is None:
        return
    if _W3 is not None and _W3.build(layer):
        # 3-bit 直接来自 overlay：跳过 AWQ->MXFP4 重量化，AWQ 副本照旧释放
        for a in _MX_FREE:
            try:
                if getattr(layer, a, None) is not None:
                    setattr(layer, a, None)
            except Exception:
                pass
        return
    tag = "%s %s" % (tuple(layer.qweight.shape), getattr(layer, "prefix", None))
    pk, e8, dense = _mx_build(layer)
    K, N = dense.shape[1], dense.shape[0]
    if _mx.WPERM:
        # fragment order: required by libr4d's decode kernel and consumed by the HIP prefill
        # kernel through RADIANCE_MXFP4_WPERM.  A byte permutation, numerics unchanged.
        if N % 16 or K % 16:
            raise RuntimeError("cannot place N=%d K=%d in fragment order" % (N, K))
        pk = _mx.fragment_order(pk, K)
    if _MX_N["built"] < _MX_SELFCHECK:
        _mx_selfcheck(tag, dense, pk, e8)
    del dense
    layer.mxfp4_w = pk.contiguous()
    layer.mxfp4_ws = e8.contiguous()
    layer.mxfp4_ref = _mx.make_row_ref(layer.mxfp4_ws)
    ch = _mx.build_chunks(layer.mxfp4_w, layer.mxfp4_ws, layer.mxfp4_ref)
    if ch is not None:
        layer.mxfp4_wc, layer.mxfp4_wsc, layer.mxfp4_refc = ch
        print("[r4d.mxfp4] chunked N=%d -> %d x %d"
              % (layer.mxfp4_w.shape[0], len(ch[0]), ch[0][0].shape[0]), flush=True)
    _mx.init_decode_scratch(pk.device)
    _MX_N["built"] += 1
    for a in _MX_FREE:                       # drop the AWQ copy right away
        try:
            if getattr(layer, a, None) is not None:
                setattr(layer, a, None)
        except Exception:
            pass
    if _MX_N["built"] % 64 == 0:
        print("[r4d.mxfp4] converted %d layers" % _MX_N["built"], flush=True)


def _mx_install():
    _o_p = AWQLinearKernel.process_weights_after_loading
    _o_a = AWQLinearKernel.apply

    def _p(self, layer):
        try:
            _mx_process(layer)
        except Exception as e:
            import traceback
            _MX_N["failed"] += 1
            print("[r4d.mxfp4] convert failed, keeping AWQ path: %r" % (e,), flush=True)
            traceback.print_exc()
            for a in ("mxfp4_w", "mxfp4_ws", "mxfp4_ref"):
                if hasattr(layer, a):
                    setattr(layer, a, None)
        if getattr(layer, "mxfp4_w", None) is None and getattr(layer, "w3_wk", None) is None:
            return _o_p(self, layer)          # unconverted layer keeps the native AWQ init

    def _a(self, layer, x, bias=None):
        if getattr(layer, "w3_wk", None) is not None:
            return _W3.apply(x, layer, bias)
        w = getattr(layer, "mxfp4_w", None)
        if w is None:
            if isinstance(x, tuple):
                q, s = x
                x = (q.float() * s.view(-1, 1)).to(torch.bfloat16)
            return _o_a(self, layer, x, bias)
        wc = getattr(layer, "mxfp4_wc", None)
        if wc is not None and isinstance(x, tuple) and x[0].shape[0] <= _mx.R4D_DECODE_MAX_M:
            out = _mx.mxfp4_linear_pq_chunked(q0 := x[0], x[1], wc, layer.mxfp4_wsc,
                                              layer.mxfp4_refc, w.shape[0])
            return out if bias is None else out + bias
        if wc is not None and not isinstance(x, tuple):
            lead0 = x.shape[:-1]
            xr = x.reshape(-1, x.shape[-1])
            if xr.shape[0] <= _mx.R4D_DECODE_MAX_M:
                out = _mx.mxfp4_linear_chunked(xr, wc, layer.mxfp4_wsc,
                                               layer.mxfp4_refc, w.shape[0])
                out = out.reshape(*lead0, out.shape[-1])
                return out if bias is None else out + bias
        if isinstance(x, tuple):
            q, s = x
            out = _mx.mxfp4_linear_pq(q, s, w, layer.mxfp4_ws, layer.mxfp4_ref)
            return out if bias is None else out + bias
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if x.shape[-1] != w.shape[1] * 2:
            raise RuntimeError("MXFP4 shape mismatch: x K=%d vs weight K=%d"
                               % (x.shape[-1], w.shape[1] * 2))
        out = _mx.mxfp4_linear(x, w, layer.mxfp4_ws, layer.mxfp4_ref)
        out = out.reshape(*lead, out.shape[-1])
        return out if bias is None else out + bias

    AWQLinearKernel.process_weights_after_loading = _p
    AWQLinearKernel.apply = _a
    print("[r4d.mxfp4] path A wired on AWQLinearKernel (single path, AWQ freed)")


if _mx_os.environ.get("SGLANG_MXFP4_OFF", "0") != "1":
    _mx_install()
