import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

# 1. Configuration Constants
BASE_WEIGHTS_PATH = os.path.join('models', 'densenet201_final.pth')
OUTPUT_WEIGHTS_PATH = os.path.join('models', 'densenet201_updated.pth')
DATA_DIR = 'new_training_data'

BATCH_SIZE = 8       # Small batch size keeps memory low for CPU/low-end GPU laptops
EPOCHS = 8         # Fine-tuning requires very few epochs
LEARNING_RATE = 1e-6 # Micro learning rate protects baseline features from wiping out

def setup_data_loaders():
    # Validation / Test transforms (No distortion)
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Safe clinical augmentations for training
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(DATA_DIR, 'train')
    val_path = os.path.join(DATA_DIR, 'val')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing data directories inside '{DATA_DIR}/'. Please populate folders first.")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

def initialize_incremental_model(device):
    print("Rebuilding DenseNet-201 network profile...")
    model = models.densenet201()
    
    # Modify head to match binary diagnostic criteria
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    
    if os.path.exists(BASE_WEIGHTS_PATH):
        print(f"Baseline weights discovered at {BASE_WEIGHTS_PATH}. Injecting states...")
        model.load_state_dict(torch.load(BASE_WEIGHTS_PATH, map_location=device))
    else:
        print(f"⚠️ Warning: {BASE_WEIGHTS_PATH} not found. Model will train from scratch.")
        
    # Unfreeze all layers so entire model can subtly adapt to new inputs
    for param in model.parameters():
        param.requires_grad = True
        
    return model.to(device)

def train_model():
    # Hardware acceleration selector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nExecution Engine Engine Target: [{device.type.upper()}]")
    if device.type == 'cpu':
        print("⚠️ Running on CPU mode. Training steps might process slowly.")

    try:
        train_loader, val_loader, num_train, num_val = setup_data_loaders()
        print(f"Datasets found! Training Samples: {num_train} | Validation Samples: {num_val}")
    except Exception as e:
        print(f"Initialization Aborted: {str(e)}")
        return

    model = initialize_incremental_model(device)
    
    # Balanced Loss configuration — auto-computed from actual class counts
    train_dataset = train_loader.dataset
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")
    print(f"Applied class weights: {dict(zip(train_dataset.classes, [round(w, 4) for w in class_weights]))}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\nStarting Fine-Tuning Execution Run...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # --- TRAINING SECTION ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_train_loss = running_loss / num_train
        epoch_train_acc = (running_corrects.double() / num_train).item()

        # --- VALIDATION SECTION ---
        model.eval()
        running_val_loss, running_val_corrects = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_val_corrects += torch.sum(preds == labels.data)

                epoch_val_loss = running_val_loss / num_val
        epoch_val_acc = (running_val_corrects.double() / num_val).item()

        # Step the scheduler using validation loss — reduces LR if it plateaus
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f"  → Learning rate reduced: {prev_lr:.2e} → {new_lr:.2e}")

        duration = time.time() - start_time
        
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | "
              f"Time: {duration:.1f}s")

        # Keep track of best validation accuracy weights
        if epoch_val_acc >= best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nTraining Complete. Peak Local Validation Accuracy: {best_acc * 100:.2f}%")
    
    # Save output weights safely to disk
    os.makedirs('models', exist_ok=True)
    torch.save(best_model_wts, OUTPUT_WEIGHTS_PATH)
    print(f"Optimized model weights saved cleanly to: {OUTPUT_WEIGHTS_PATH}")

if __name__ == '__main__':
    train_model()