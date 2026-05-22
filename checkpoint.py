"""
© 2026 Nicholas J. Calabro. All rights reserved.

The purpose of this class is to manage the checkpoint file,
keeping direct save implementations free from train.py.

Currently not utilized in classifier.py, but it does not save either,
which means the helpers in MultiClassifier are equally readable
"""

import os

import torch


class CheckpointFile:
    def __init__(self, best_path: str, device: torch.device):
        self.best_path = best_path
        self.device = device

    def save_periodic(self, path: str, epoch: int, classifier, best_val: float, no_improve: int):
        torch.save({
            "epoch": epoch,
            "model": classifier.model.state_dict(),
            "ema_model": classifier.ema_model.module.state_dict(),
            "optimizer": classifier.optimizer.state_dict(),
            "best_val": best_val,
            "no_improve": no_improve,
        }, path)

    def save(self, model_state: dict, thresholds, temperature_scaler):
        torch.save({
            "model": model_state,
            "thresholds": thresholds,
            "temperature": temperature_scaler.temps.detach().cpu().tolist(),
        }, self.best_path)
        print(f"Temperature saved: {temperature_scaler.temps}")

    def load_best(self) -> dict:
        if not os.path.exists(self.best_path):
            raise FileNotFoundError(f"No checkpoint at {self.best_path}. Training may have failed early.")
        return torch.load(self.best_path, map_location=self.device, weights_only=False)