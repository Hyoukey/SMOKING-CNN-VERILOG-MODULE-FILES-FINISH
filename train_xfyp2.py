import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import copy
import os

# --- IMPORT YOUR DATA LOADER ---
try:
    import preprocess_2_new as pp2 
    train_loader = pp2.train_loader
    val_loader = pp2.val_loader
except ImportError:
    print("Error: 'preprocess_2_new.py' not found.")
    exit()

# ==========================================
# 1. THE XFYP2-STYLE ARCHITECTURE (Adapted for 28x28)
# ==========================================
class SmokeNetX(nn.Module):
    def __init__(self):
        super(SmokeNetX, self).__init__()
        
        # Layer 1: 1 Input -> 3 Filters (3x3 kernels)
        # 28x28 -> 26x26 -> Pool -> 13x13
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, bias=False) 
        
        # Layer 2: 3 Inputs -> 3 Filters (3x3 kernels)
        # 13x13 -> 11x11 -> Pool -> 5x5
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, bias=False)
        
        # Layer 3: 3 Inputs -> 3 Filters (3x3 kernels)
        # 5x5 -> 3x3 -> Pool -> 1x1
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, bias=False)
        
        # Fully Connected
        # Input is tiny! Just 3 channels * 1 pixel = 3 features.
        self.fc1 = nn.Linear(3, 2, bias=False) 

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 3
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc1(x)
        return x

# ==========================================
# 2. TRAINING SETUP
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SmokeNetX().to(device)

# Initialize Weights (He Init for ReLU)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(weights_init)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
print(f"--- Training SmokeNet-X on {device} ---")
best_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())

for epoch in range(100): # 100 Epochs
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        best_wts = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}: New Best Accuracy: {best_acc:.2%}")

print(f"Training Complete. Best Accuracy: {best_acc:.2%}")
model.load_state_dict(best_wts)

# ==========================================
# 4. HEX EXPORTER (Auto-Scaling for Hardware)
# ==========================================
print("\n--- EXPORTING WEIGHTS FOR VERILOG ---")

def to_hex(val):
    val = int(val)
    val = max(min(val, 127), -128) # Clamp 8-bit
    if val < 0: val = 256 + val
    return f"8'h{val:02x}"

def export_layer(name, weights):
    # Auto-Scale Logic: Map max weight to 127
    w_np = weights.data.cpu().numpy()
    max_val = np.max(np.abs(w_np))
    if max_val == 0: max_val = 1.0
    scale = 127.0 / max_val
    
    print(f"Exporting {name} (Scale: {scale:.2f})...")
    
    # Flatten and Save
    w_flat = w_np.flatten()
    with open(f"{name}_weights.txt", "w") as f:
        for i, val in enumerate(w_flat):
            hex_str = to_hex(val * scale)
            f.write(f"assign weight[{i}] = {hex_str};\n")

# Export all layers
export_layer("conv1", model.conv1.weight)
export_layer("conv2", model.conv2.weight)
export_layer("conv3", model.conv3.weight)
export_layer("fc", model.fc1.weight)

print("Done! Weights saved as .txt files.")