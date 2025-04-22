import os
import shutil
from pathlib import Path

def fix_ade20k_structure():
    # Define absolute paths
    base_dir = Path(r'C:\Users\Shijie Wang\Desktop\Research\VFM\OpenClip\VFM_lite_evaluation-main\datasets\ADE20K')
    images_dir = base_dir / 'images' / 'validation'
    annotations_dir = base_dir / 'annotations' / 'validation'
    
    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")
    
    # Create annotations directory if it doesn't exist
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Move .png files from images to annotations
    print("\nMoving annotation files to correct directory...")
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            src = images_dir / file
            dst = annotations_dir / file
            print(f"Moving {file} to annotations directory")
            shutil.move(str(src), str(dst))
    
    # Verify the move
    print("\nVerifying directory structure:")
    print("Images (.jpg files):")
    for file in sorted(os.listdir(images_dir)):
        if file.endswith('.jpg'):
            print(f"  {file}")
    
    print("\nAnnotations (.png files):")
    for file in sorted(os.listdir(annotations_dir)):
        if file.endswith('.png'):
            print(f"  {file}")

if __name__ == '__main__':
    fix_ade20k_structure()