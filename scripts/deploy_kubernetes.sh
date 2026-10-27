#!/usr/bin/env bash
set -e

echo "[K8S] Deploying PyTorch Distributed Training Job via Kubeflow Training Operator..."

kubectl apply -f - <<EOF
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "gpu-distributed-compressor-job"
  namespace: "default"
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: ghcr.io/ddkellar01/gpu-distributed-compressor:latest
              command: ["python", "src/train.py", "--deepspeed", "--deepspeed_config", "configs/ds_config_zero2.json"]
              resources:
                limits:
                  nvidia.com/gpu: 8
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: ghcr.io/ddkellar01/gpu-distributed-compressor:latest
              command: ["python", "src/train.py", "--deepspeed", "--deepspeed_config", "configs/ds_config_zero2.json"]
              resources:
                limits:
                  nvidia.com/gpu: 8
EOF

echo "[K8S] PyTorchJob submitted successfully."
