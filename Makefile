.PHONY: test lint build clean run-single run-multi

test:
	pytest tests/

lint:
	flake8 src/ tests/

build:
	docker build -t gpu-distributed-compressor:latest -f docker/Dockerfile .

clean:
	bash scripts/clean_checkpoints.sh

run-single:
	bash scripts/run_single_node.sh 8 configs/ds_config_zero2.json

run-multi:
	sbatch scripts/run_multi_node.sh
