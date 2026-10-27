import torch
import torch.distributed as dist

def register_compression_hook(process_group, compressor):
    """
    Registers a custom PyTorch DDP communication hook for automated gradient compression during backward pass.
    """
    def hook(state, bucket):
        tensor = bucket.buffer()
        compressed_data, shape = compressor.compress("ddp_bucket", tensor)
        
        # In-place decompression after communication step
        decompressed = compressor.decompress(compressed_data, shape)
        tensor.copy_(decompressed)
        
        fut = torch.futures.Future()
        fut.set_result(bucket)
        return fut

    return hook
