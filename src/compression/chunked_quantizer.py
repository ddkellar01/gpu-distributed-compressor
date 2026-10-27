import torch

class ChunkedQuantizer:
    """
    Splits ultra-large model gradients into memory-bound chunks before quantizing to prevent OOM spikes.
    """
    def __init__(self, chunk_size: int = 1048576, num_bits: int = 8):
        self.chunk_size = chunk_size
        self.num_bits = num_bits
        self.qmax = (2 ** (num_bits - 1)) - 1

    def quantize_chunked(self, tensor: torch.Tensor):
        flat = tensor.view(-1)
        chunks = torch.split(flat, self.chunk_size)
        quantized_chunks = []
        scales = []

        for chunk in chunks:
            max_val = torch.max(torch.abs(chunk))
            scale = torch.clamp(max_val / self.qmax, min=1e-8)
            q_chunk = torch.clamp(torch.round(chunk / scale), -self.qmax, self.qmax).to(torch.int8)
            quantized_chunks.append(q_chunk)
            scales.append(scale)

        return (quantized_chunks, scales), tensor.shape

    def decompress_chunked(self, compressed_data):
        (quantized_chunks, scales), original_shape = compressed_data
        decompressed_chunks = [q.to(s.dtype) * s for q, s in zip(quantized_chunks, scales)]
        return torch.cat(decompressed_chunks).view(original_shape)
