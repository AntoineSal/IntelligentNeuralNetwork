import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import zipfile

# === CONFIGURATION ===
CONFIG = {
    'dataset': 'text8',
    'vocab_size': 27,
    'd_model': 256,
    'n_layers': 4,    
    'dropout': 0.1,
    'lr': 3e-4,            
    'batch_size': 8,       
    'seq_len': 128,        
    'epochs': 1,           
    'subset_size': 5000000 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (Rest of the standard JIT benchmark code) ...
# For brevity, I am restoring the file structure. 
# The user has the content in history if needed, but the CUDA version is the priority now.

