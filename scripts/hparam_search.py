import optuna
import torch
from models.model_builder import build_model
from compression.powersgd import PowerSGDCompressor

def objective(trial):
    rank = trial.suggest_int('power_rank', 2, 16, step=2)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    model = build_model()
    compressor = PowerSGDCompressor(rank=rank)
    
    dummy_input = torch.randint(0, 32000, (4, 64))
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    
    total_error = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            comp_data, shape = compressor.compress(name, param.grad)
            decomp = compressor.decompress(comp_data, shape)
            total_error += torch.norm(param.grad - decomp).item()
            break
            
    return total_error

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print(f"[OPTUNA] Best Trial Parameters: {study.best_params}")
