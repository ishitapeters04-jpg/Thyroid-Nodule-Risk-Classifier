import os
import shutil
import random

# Configuration
STAGING_DIR = os.path.join('histopathology_data', 'staging')
OUTPUT_DIR = os.path.join('histopathology_data', 'new_training_data')
VAL_SPLIT = 0.2  # 20% validation
SEED = 42

def split_data():
    random.seed(SEED)

    if not os.path.exists(STAGING_DIR):
        print(f"Missing staging area root. Please tell your team to use: '{STAGING_DIR}/'")
        return

    classes = ['benign', 'malignant']

    # Check staging folders aren't empty
    for cls in classes:
        cls_path = os.path.join(STAGING_DIR, cls)
        if not os.path.exists(cls_path) or len(os.listdir(cls_path)) == 0:
            print(f"Aborting split run: Staging area folder '{cls}' is currently empty.")
            return

    print("=== Initializing Histopathology 80:20 Data Splitter ===\n")

    # Clean previous split output so re-runs don't mix old + new
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for cls in classes:
        src_dir = os.path.join(STAGING_DIR, cls)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        total = len(files)
        val_count = max(1, int(total * VAL_SPLIT))
        train_count = total - val_count

        val_files = files[:val_count]
        train_files = files[val_count:]

        print(f"Processing Group [{cls.upper()}] -> Found: {total} total files.")
        print(f"  -> Allocating {train_count} items to Train subset")
        print(f"  -> Allocating {val_count} items to Validation subset\n")

        # Create destination folders
        train_dest = os.path.join(OUTPUT_DIR, 'train', cls)
        val_dest = os.path.join(OUTPUT_DIR, 'val', cls)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)

        # Copy files
        for f in train_files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(train_dest, f))
        for f in val_files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(val_dest, f))

    print(f"Target assets processed and copied into '{OUTPUT_DIR}/' workspace successfully.")

if __name__ == '__main__':
    split_data()
    