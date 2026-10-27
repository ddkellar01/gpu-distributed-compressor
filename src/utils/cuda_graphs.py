import torch

class CUDAGraphRunner:
    """
    Wraps module forward and backward passes inside CUDA Graphs to eliminate CPU launcher overhead.
    """
    def __init__(self, model, sample_input: torch.Tensor):
        self.model = model
        self.graph = torch.cuda.CUDAGraph()
        self.static_input = sample_input.clone()
        
        # Warmup iterations
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.static_output = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture CUDA Graph
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def forward(self, x: torch.Tensor):
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output
