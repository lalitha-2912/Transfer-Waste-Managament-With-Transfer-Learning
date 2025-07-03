import os
import shutil
import random

# Configuration
DATASET_DIR = 'Dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
CLASSES = ['Biodegradable Images', 'Recyclable Images', 'Trash Images']
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Create train/test directories and class subfolders
for split_dir in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(split_dir, exist_ok=True)
    for class_name in CLASSES:
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

for class_name in CLASSES:
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_path):
        print(f"Class folder not found: {class_path}")
        continue
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Move images to train
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, class_name, img)
        shutil.move(src, dst)

    # Move images to test
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TEST_DIR, class_name, img)
        shutil.move(src, dst)

    print(f"Moved {len(train_images)} images to train/{class_name}, {len(test_images)} to test/{class_name}")

print("Data split complete!") 