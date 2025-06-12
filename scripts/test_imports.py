print("Attempting to import os...")
import os
print("os imported.")

print("Attempting to import json...")
import json
print("json imported.")

print("Attempting to import pandas...")
import pandas as pd
print("pandas imported.")

print("Attempting to import torch...")
import torch
print("torch imported.") # <--- THIS IS OFTEN A POINT OF FAILURE IF CUDA/INSTALL IS BAD

print("Attempting to import torchvision.models...")
import torchvision.models as models
print("torchvision.models imported.")

print("Attempting to import torchvision.transforms...")
import torchvision.transforms as transforms
print("torchvision.transforms imported.")

print("Attempting to import DataLoader, Subset from torch.utils.data...")
from torch.utils.data import DataLoader, Subset
print("DataLoader, Subset imported.")

print("Attempting to import ImageFolder from torchvision.datasets...")
from torchvision.datasets import ImageFolder
print("ImageFolder imported.")

print("Attempting to import time...")
import time
print("time imported.")

print("Attempting to import Path from pathlib...")
from pathlib import Path
print("Path imported.")

print("Attempting to import glob...")
import glob
print("glob imported.")

print("Attempting to import traceback...")
import traceback
print("traceback imported.")

print("--- ALL IMPORTS SUCCEEDED ---")