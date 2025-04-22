import os
import urllib.request
import zipfile
from tqdm import tqdm
import shutil
from pathlib import Path
import time

def download_with_progress(url, output_path):
    """Download with progress bar"""
    print(f"Downloading from {url}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, output_path, 
                                 reporthook=t.update_to)

def safe_remove(path):
    """Safely remove a file with retries"""
    max_attempts = 5
    for i in range(max_attempts):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception as e:
            if i < max_attempts - 1:
                print(f"Attempt {i+1} failed to remove file, waiting before retry...")
                time.sleep(1)  # Wait 1 second before retry
            else:
                print(f"Could not remove file {path}, but continuing anyway...")
    return False

def download_ade20k():
    # Set up paths
    base_dir = Path(r'C:\Users\Shijie Wang\Desktop\Research\VFM\OpenClip\VFM_lite_evaluation-main\datasets\ADE20K')
    images_dir = base_dir / 'images' / 'validation'
    annotations_dir = base_dir / 'annotations' / 'validation'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean existing directories
    for file in images_dir.glob('*'):
        safe_remove(file)
    for file in annotations_dir.glob('*'):
        safe_remove(file)
    
    zip_path = base_dir / "ADEChallengeData2016.zip"
    
    try:
        # Only download if the file doesn't exist
        if not zip_path.exists():
            print("Downloading ADE20K dataset...")
            url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
            download_with_progress(url, str(zip_path))
        else:
            print("Using existing zip file...")
        
        print("\nExtracting files...")
        with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
            # Extract all files first
            zip_ref.extractall(str(base_dir))
        
        # Source directories from extracted files
        src_img_dir = base_dir / 'ADEChallengeData2016' / 'images' / 'validation'
        src_ann_dir = base_dir / 'ADEChallengeData2016' / 'annotations' / 'validation'
        
        # Get first 10 image files
        image_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith('.jpg')])[:10]
        
        print("\nCopying image-annotation pairs...")
        for img_file in image_files:
            ann_file = img_file.replace('.jpg', '.png')
            
            # Copy image
            shutil.copy2(src_img_dir / img_file, images_dir / img_file)
            # Copy annotation
            shutil.copy2(src_ann_dir / ann_file, annotations_dir / ann_file)
            print(f"Copied pair: {img_file} - {ann_file}")
        
        # Remove the extracted directory
        print("\nCleaning up extracted files...")
        try:
            shutil.rmtree(base_dir / 'ADEChallengeData2016')
        except Exception as e:
            print(f"Note: Could not remove extracted directory, but continuing anyway...")
        
        # Try to remove zip file
        print("Cleaning up zip file...")
        safe_remove(zip_path)
        
        print("\nVerifying final dataset...")
        images = sorted(list(images_dir.glob('*.jpg')))
        annotations = sorted(list(annotations_dir.glob('*.png')))
        
        print(f"Images found: {len(images)}")
        print(f"Annotations found: {len(annotations)}")
        
        if len(images) == 10 and len(annotations) == 10:
            print("\nDataset organization completed successfully!")
        else:
            print("\nWarning: Number of files is not as expected, but files were copied.")
        
        # Create/update config file
        config_dir = base_dir.parent.parent / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        config_content = f"""# ------ root_path/dataset_name ------
ade20k_path: '{str(base_dir)}'
cache_dir: 'cache_dir'
dataset: 'ade20k'

# ------ Basic Config ------
model: 'open_clip'
backbone: 'ViT-B/32'

load_pre_feat: False
"""
        
        with open(config_dir / 'ade20k.yaml', 'w') as f:
            f.write(config_content)
        
        print("\nConfiguration file updated at: configs/ade20k.yaml")
        
    except Exception as e:
        print(f"Error during process: {str(e)}")
        print("Note: If files were copied successfully, you can ignore this error.")

if __name__ == '__main__':
    download_ade20k()