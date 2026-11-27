"""Render the synthetic dataset (20 train + 4 test sequences x 24 frames, ~3 min on an M2 Pro).

    .venv/bin/python generate_data.py
"""
import torch

from lu.data import generate

if __name__ == "__main__":
    torch.set_num_threads(8)
    generate()
