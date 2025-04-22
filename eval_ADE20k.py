import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import time

from datasets.ade20k import ADE20K
from model.openai_clip import openai_clip

def calculate_metrics(pred_mask, true_mask, num_classes=150):
    """Calculate multiple metrics including mIOU, precision, and recall"""
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()
        
        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum() - intersection
        
        # Calculate IoU
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()
        iou_per_class.append(iou)
        
        # Calculate Precision
        if pred_cls.sum() == 0:
            precision = float('nan')
        else:
            precision = (intersection / pred_cls.sum()).item()
        precision_per_class.append(precision)
        
        # Calculate Recall
        if true_cls.sum() == 0:
            recall = float('nan')
        else:
            recall = (intersection / true_cls.sum()).item()
        recall_per_class.append(recall)
    
    # Calculate mean metrics (excluding NaN values)
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    valid_precisions = [p for p in precision_per_class if not np.isnan(p)]
    valid_recalls = [r for r in recall_per_class if not np.isnan(r)]
    
    miou = np.mean(valid_ious) if valid_ious else 0
    mean_precision = np.mean(valid_precisions) if valid_precisions else 0
    mean_recall = np.mean(valid_recalls) if valid_recalls else 0
    
    return {
        'miou': miou,
        'class_ious': iou_per_class,
        'mean_precision': mean_precision,
        'class_precisions': precision_per_class,
        'mean_recall': mean_recall,
        'class_recalls': recall_per_class
    }

def main():
    # Start timing
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings in yaml format')
    args = parser.parse_args()
    
    assert os.path.exists(args.config)
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    # Set random seeds
    random.seed(1)
    torch.manual_seed(1)
    
    # Load model and preprocessing
    model, preprocess = openai_clip(cfg['backbone'])
    model.eval()
    
    # Initialize dataset
    dataset = ADE20K(cfg['ade20k_path'], preprocess, num_samples=10)
    test_loader = torch.utils.data.DataLoader(
        dataset.test,
        batch_size=1,
        num_workers=4,
        shuffle=False
    )
    
    # Setup results logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'ade20k_evaluation_results_{timestamp}.txt'
    total_metrics = {
        'miou': 0,
        'mean_precision': 0,
        'mean_recall': 0
    }
    all_results = []
    
    # Initialize the conv layer
    if torch.cuda.is_available():
        conv_layer = torch.nn.Conv2d(512, 150, 1).cuda().half()
    else:
        conv_layer = torch.nn.Conv2d(512, 150, 1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    with torch.no_grad():
        for images, masks, img_names in tqdm(test_loader, desc="Evaluating"):
            if torch.cuda.is_available():
                images = images.cuda().half()
                masks = masks.cuda()
            
            # Get CLIP features
            features = model.encode_image(images)
            
            # Reshape features
            B = features.shape[0]
            features = features.view(B, 512, 1, 1)
            
            # Upsample to match mask size
            features = F.interpolate(features, 
                                   size=masks.shape[1:],
                                   mode='bilinear', 
                                   align_corners=False)
            
            # Get predictions
            pred_logits = conv_layer(features)
            pred_logits = pred_logits.float()
            pred_masks = pred_logits.argmax(dim=1)
            
            # Calculate metrics
            metrics = calculate_metrics(pred_masks, masks)
            
            # Update total metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Store results
            all_results.append({
                'image_name': img_names[0],
                **metrics
            })
    
    # Calculate average metrics
    num_images = len(test_loader)
    avg_metrics = {k: v / num_images for k, v in total_metrics.items()}
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Save detailed results
    with open(results_file, 'w') as f:
        # Write header information
        f.write(f"ADE20K Evaluation Results for CLIP {cfg['backbone']}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of images evaluated: {len(dataset.test)}\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n\n")
        
        # Write average metrics
        f.write("=== Overall Metrics ===\n")
        f.write(f"Average mIOU: {avg_metrics['miou']:.4f}\n")
        f.write(f"Average Precision: {avg_metrics['mean_precision']:.4f}\n")
        f.write(f"Average Recall: {avg_metrics['mean_recall']:.4f}\n\n")
        
        # Write per-image results
        f.write("=== Per-Image Results ===\n")
        for result in all_results:
            f.write(f"\nImage: {result['image_name']}\n")
            f.write(f"mIOU: {result['miou']:.4f}\n")
            f.write(f"Precision: {result['mean_precision']:.4f}\n")
            f.write(f"Recall: {result['mean_recall']:.4f}\n")
            
            f.write("\nPer-class metrics:\n")
            for cls_idx in range(len(result['class_ious'])):
                iou = result['class_ious'][cls_idx]
                precision = result['class_precisions'][cls_idx]
                recall = result['class_recalls'][cls_idx]
                if not (np.isnan(iou) and np.isnan(precision) and np.isnan(recall)):
                    f.write(f"Class {cls_idx}:\n")
                    f.write(f"  IoU: {iou:.4f}\n")
                    f.write(f"  Precision: {precision:.4f}\n")
                    f.write(f"  Recall: {recall:.4f}\n")
    
    print(f"\n**** Evaluation Results ****")
    print(f"Average mIOU: {avg_metrics['miou']:.4f}")
    print(f"Average Precision: {avg_metrics['mean_precision']:.4f}")
    print(f"Average Recall: {avg_metrics['mean_recall']:.4f}")
    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()