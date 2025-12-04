# smokenetx_train_and_export.py
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
    raise

# ==========================================
class SmokeNetX(nn.Module):
    def __init__(self):
        super(SmokeNetX, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, bias=False) 
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, bias=False)
        self.fc1 = nn.Linear(3, 2, bias=False) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SmokeNetX().to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
model.apply(weights_init)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAINING (short loop; change epochs as needed)
best_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())
for epoch in range(500):  # reduce for quick run, increase for final training
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # validation
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
# EXPORTER (verilog-friendly .txt)
# ==========================================
def to_hex8(val):
    # val is integer (after scaling)
    val = int(np.round(val))
    val = max(min(val, 127), -128)
    if val < 0:
        val = 256 + val
    return f"8'h{val:02x}"

def export_layer(name, weights_tensor):
    w_np = weights_tensor.data.cpu().numpy()
    max_val = np.max(np.abs(w_np))
    if max_val == 0: max_val = 1.0
    scale = 127.0 / max_val
    print(f"Exporting {name} (scale {scale:.3f})")

    w_flat = w_np.flatten()
    fname = f"{name}_weights.txt"
    with open(fname, "w") as f:
        for i, v in enumerate(w_flat):
            h = to_hex8(v * scale)
            f.write(f"assign weight[{i}] = {h};\n")
    print(f"Saved {fname}")

os.makedirs("weights_verilog", exist_ok=True)
os.chdir("weights_verilog")
export_layer("conv1", model.conv1.weight)  # shape (3,1,3,3)
export_layer("conv2", model.conv2.weight)  # shape (3,3,3,3)
export_layer("conv3", model.conv3.weight)  # shape (3,3,3,3)
export_layer("fc", model.fc1.weight)       # shape (2,3)
print("Done exporting.")
