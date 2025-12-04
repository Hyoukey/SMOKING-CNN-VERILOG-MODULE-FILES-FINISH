import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 32

# --- UPDATED TRANSFORM PIPELINE ---
# We now add Data Augmentation to the training data
# to fight overfitting.
train_transform = transforms.Compose([
    transforms.Grayscale(),
    
    # --- NEW: Data Augmentation ---
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip image
    transforms.RandomRotation(10),        # Rotate between -10 and +10 degrees
    # ------------------------------
    
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Validation/Test data should NOT be augmented
val_test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
# (Using the paths from your successful run)
train_dataset = datasets.ImageFolder('Dataset/training_data/', transform=train_transform) # <-- Use new transform
val_dataset = datasets.ImageFolder('Dataset/validation_data/', transform=val_test_transform) # <-- Use val transform
test_dataset = datasets.ImageFolder('Dataset/testing_data/test_pictures_preprocessed', transform=val_test_transform) # <-- Use val transform

# ... (all your transform and data loader code stays the same) ...
# ...
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- UPDATED CNN architecture (Simpler) ---
# We are reducing the number of channels to fight overfitting
# preprocess_2_new.py (Modified for Senior's Architecture)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # --- WAS: (1, 6) ---
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)     # Input: 1x28x28 -> Output: 4x24x24
        self.pool = nn.MaxPool2d(2, 2)                  # Output: 4x12x12
        
        # --- WAS: (6, 16) ---
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5)     # Output: 8x8x8 -> pool -> 8x4x4
        
        self.conv_dropout = nn.Dropout(p=0.25) 
        #self.fc_dropout = nn.Dropout(p=0.5)

        # --- WAS: 16 * 4 * 4 = 256 inputs ---
        # --- NOW: 8 * 4 * 4 = 128 inputs ---
        self.fc1 = nn.Linear(48, 2) # <-- Reduced
        #self.fc2 = nn.Linear(64, 32)        # <-- Reduced
        #self.fc3 = nn.Linear(32, 2)         # <-- Reduced

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.conv_dropout(x)
        
        # --- MUST MATCH fc1 input ---
        x = x.view(-1, 48) # <-- Updated
        
        #x = F.relu(self.fc1(x))
        #x = self.fc_dropout(x)
        #x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# ... (rest of the file is the same) ...
# Instantiate model, loss, optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.8) # (Optimizer will be redefined in train_model.py)
print("Model created successfully!")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# ...
#print("Model created successfully!")
# ...