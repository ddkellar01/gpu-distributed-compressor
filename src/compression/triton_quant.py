import torch
import triton
import triton.language as tl

@triton.jit
def _int8_quant_kernel(
    x_ptr, y_ptr, scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    
    # Quantize to INT8
    y = tl.extra.cuda.libdevice.round(x / scale)
    y = tl.clamp(y, -128.0, 127.0)
    
    tl.store(y_ptr + offsets, y.to(tl.int8), mask=mask)

def triton_quantize(tensor: torch.Tensor):
    assert tensor.is_cuda, "Tensor must be on CUDA"
    n_elements = tensor.numel()
    max_val = torch.max(torch.abs(tensor))
    scale = max_val / 127.0
    scale = torch.clamp(scale, min=1e-8)
    
    quantized = torch.empty_like(tensor, dtype=torch.int8)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _int8_quant_kernel[grid](
        tensor, quantized, scale,
        n_elements,
        BLOCK_SIZE=1024
    )
    return quantized, scale
