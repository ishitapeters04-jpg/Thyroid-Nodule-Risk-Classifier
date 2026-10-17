import os
import shutil
import random
from tqdm import tqdm  # Optional: adds progress bars if installed, fails gracefully if not

# 1. Path Rules Configurations
SRC_STAGING_DIR = 'staging_area'
DEST_OUTPUT_DIR = 'new_training_data'
SPLIT_RATIO = 0.8  # 80% Training, 20% Validation
RANDOM_SEED = 42   # Guarantees identical shuffles across teammates' systems

def bootstrap_directories(base_output, classes):
    """Creates standard training and validation subdirectories smoothly."""
    for stage in ['train', 'val']:
        for cls in classes:
            target_path = os.path.join(base_output, stage, cls)
            os.makedirs(target_path, exist_ok=True)

def split_and_stage_data():
    classes = ['benign', 'malignant']
    
    # Verify staging area initialization
    if not os.path.exists(SRC_STAGING_DIR):
        print(f"Creating missing staging area root. Please tell your team to use: '{SRC_STAGING_DIR}/'")
        for cls in classes:
            os.makedirs(os.path.join(SRC_STAGING_DIR, cls), exist_ok=True)
        print("Aborting split run: Staging area folders are currently empty.")
        return

    # Set up production output directory paths
    bootstrap_directories(DEST_OUTPUT_DIR, classes)

    print("\n=== Initializing Stratified 80:20 Data Splitter ===")
    
    for cls in classes:
        source_class_folder = os.path.join(SRC_STAGING_DIR, cls)
        
        # Gather valid ultrasound diagnostic image paths
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        all_images = [
            f for f in os.listdir(source_class_folder) 
            if f.lower().endswith(valid_extensions)
        ]
        
        total_count = len(all_images)
        if total_count == 0:
            print(f"ℹ️ Class folder '{cls}' inside staging space contains 0 images. Skipping configuration...")
            continue
            
        # Shuffle using a fixed seed to protect structural uniformity across different machines
        random.seed(RANDOM_SEED)
        random.shuffle(all_images)
        
        # Compute exact cutoff limits
        split_index = int(total_count * SPLIT_RATIO)
        train_pool = all_images[:split_index]
        val_pool = all_images[split_index:]
        
        print(f"\nProcessing Group [{cls.upper()}] -> Found: {total_count} total files.")
        print(f"  -> Allocating {len(train_pool)} items to Train subset")
        print(f"  -> Allocating {len(val_pool)} items to Validation subset")
        
        # Transfer training allocations
        for img_name in train_pool:
            src_file = os.path.join(source_class_folder, img_name)
            dest_file = os.path.join(DEST_OUTPUT_DIR, 'train', cls, img_name)
            shutil.copy(src_file, dest_file)
            
        # Transfer validation allocations
        for img_name in val_pool:
            src_file = os.path.join(source_class_folder, img_name)
            dest_file = os.path.join(DEST_OUTPUT_DIR, 'val', cls, img_name)
            shutil.copy(src_file, dest_file)

    print(f"\nTarget assets processed and copied into '{DEST_OUTPUT_DIR}/' workspace successfully.")

if __name__ == '__main__':
    split_and_stage_data()