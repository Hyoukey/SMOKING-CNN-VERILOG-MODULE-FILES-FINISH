import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. DEFINE THE MODEL
# ==========================================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)     
        self.pool = nn.MaxPool2d(2, 2)                  
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5)     
        self.conv_dropout = nn.Dropout(p=0.25) 
        self.fc1 = nn.Linear(48, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_dropout(x)
        x = x.view(-1, 48)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 2. CONFIGURATION
# ==========================================
MODEL_PATH = 'smoking_cnn_best.pth' 

def to_hex(val, bits=8):
    val = int(val)
    if val < 0:
        val = (1 << bits) + val
    val = val & ((1 << bits) - 1)
    return f"8'h{val:02x}"

def analyze_and_export(layer_name, weights, biases, default_scale):
    print(f"\n--- Analyzing {layer_name} ---")
    
    # Convert to numpy
    w_np = weights.detach().cpu().numpy()
    b_np = biases.detach().cpu().numpy()
    
    # 1. DIAGNOSTICS: Check the raw float values
    max_val = np.max(np.abs(w_np))
    avg_val = np.mean(np.abs(w_np))
    print(f"   Max Weight (Float): {max_val:.6f}")
    print(f"   Avg Weight (Float): {avg_val:.6f}")

    # 2. SMART SCALING
    # If weights are tiny, boost the scale.
    current_scale = default_scale
    
    # Check if the current scale causes collapse
    if max_val * current_scale < 1.0:
        print(f"   [WARNING] Weights are too small for Scale {current_scale}!")
        # Calculate a scale that maps the max value to at least integer 10
        new_scale = 10.0 / max_val
        print(f"   -> Boosting Scale to {new_scale:.2f}")
        current_scale = new_scale
    else:
        print(f"   -> Using Scale {current_scale}")

    # 3. EXPORT WEIGHTS
    w_flat = w_np.flatten()
    filename_w = f"{layer_name}_weights.txt"
    zeros_count = 0
    
    with open(filename_w, "w") as f:
        for i, w in enumerate(w_flat):
            w_int = int(round(w * current_scale))
            # Count zeros for debug
            if w_int == 0: zeros_count += 1
            
            w_int = max(min(w_int, 127), -128)
            f.write(f"assign weight[{i}] = {to_hex(w_int)};\n")
    
    print(f"   Saved weights to {filename_w}")
    print(f"   (Weights rounding to Zero: {zeros_count}/{len(w_flat)})")

    # 4. EXPORT BIASES
    filename_b = f"{layer_name}_biases.txt"
    with open(filename_b, "w") as f:
        for i, b in enumerate(b_np):
            b_int = int(round(b * current_scale))
            b_int = max(min(b_int, 127), -128)
            f.write(f"assign bias[{i}] = {to_hex(b_int)};\n")
    print(f"   Saved biases to {filename_b}")

# ==========================================
# 3. EXECUTION
# ==========================================
model = CNNModel()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Layer 1: Try Scale 10 (Higher than 0.5, but safe for 255 input)
analyze_and_export("conv1", model.conv1.weight, model.conv1.bias, default_scale=10.0)

# Layer 2: Standard Scale 64
analyze_and_export("conv2", model.conv2.weight, model.conv2.bias, default_scale=64.0)

# FC Layer: Standard Scale 64
analyze_and_export("fc", model.fc1.weight, model.fc1.bias, default_scale=64.0)