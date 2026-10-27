import unittest
import torch
from models.model_builder import build_model
from utils.distributed import init_distributed

class TestPipeline(unittest.TestCase):
    def test_model_forward(self):
        model = build_model()
        dummy_input = torch.randint(0, 32000, (2, 64))
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 64, 32000))

    def test_distributed_init(self):
        rank, local_rank, world_size = init_distributed()
        self.assertGreaterEqual(rank, 0)
        self.assertGreaterEqual(world_size, 1)

if __name__ == "__main__":
    unittest.main()
