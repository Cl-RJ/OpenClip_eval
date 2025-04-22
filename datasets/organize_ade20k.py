import os
import shutil
from tqdm import tqdm

def organize_ade20k_data():
    """
    Reorganize the downloaded ADE20K data into the correct structure
    """
    # Source directories (downloaded data)
    src_img_dir = os.path.join('ADEChallengeData2016', 'images', 'validation')
    src_ann_dir = os.path.join('ADEChallengeData2016', 'annotations', 'validation')
    
    # Target directories (where our code expects the data)
    target_base = 'ADE20K'
    target_img_dir = os.path.join(target_base, 'images', 'validation')
    target_ann_dir = os.path.join(target_base, 'annotations', 'validation')
    
    # Create target directories
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_ann_dir, exist_ok=True)
    
    try:
        # Get first 10 image files
        image_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith('.jpg')])[:10]
        
        print("Moving files to the correct directory structure...")
        
        # Move images and their corresponding annotations
        for img_file in tqdm(image_files, desc="Moving files"):
            # Image paths
            src_img_path = os.path.join(src_img_dir, img_file)
            target_img_path = os.path.join(target_img_dir, img_file)
            
            # Annotation paths (replace .jpg with .png)
            ann_file = img_file.replace('.jpg', '.png')
            src_ann_path = os.path.join(src_ann_dir, ann_file)
            target_ann_path = os.path.join(target_ann_dir, ann_file)
            
            # Copy files
            shutil.copy2(src_img_path, target_img_path)
            shutil.copy2(src_ann_path, target_ann_path)
        
        print("\nOrganization complete!")
        print(f"Dataset is now properly organized in: {os.path.abspath(target_base)}")
        print("Number of images moved: 10")
        
        # Verify the organization
        num_images = len(os.listdir(target_img_dir))
        num_annotations = len(os.listdir(target_ann_dir))
        print(f"\nVerification:")
        print(f"Images in ADE20K/images/validation: {num_images}")
        print(f"Annotations in ADE20K/annotations/validation: {num_annotations}")
        
        if num_images == 10 and num_annotations == 10:
            print("\nSuccess! The dataset is correctly organized.")
        else:
            print("\nWarning: The number of files is not as expected.")
            
    except Exception as e:
        print(f"Error during organization: {str(e)}")

if __name__ == '__main__':
    organize_ade20k_data()