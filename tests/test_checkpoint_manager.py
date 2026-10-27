import os
import shutil
import tempfile
import unittest
from utils.checkpoint_manager import CheckpointManager

class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_rotation(self):
        manager = CheckpointManager(checkpoint_dir=self.test_dir, max_to_keep=2)
        manager.save({"model": 1}, step=100, score=0.9)
        manager.save({"model": 2}, step=200, score=0.95)
        manager.save({"model": 3}, step=300, score=0.98)

        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 2)

if __name__ == "__main__":
    unittest.main()
