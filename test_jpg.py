import os
import cv2
import torch
import numpy as np
import argparse
from glob import glob

from pytorch_lightning import seed_everything
from Burst_Super_Resolution.Network import BIPNet
from Burst_Super_Resolution.data_processing import synthetic_burst_generation as syn_burst_utils

seed_everything(13)

def process_custom_images(input_dir, output_dir, weights_path):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = BIPNet.load_from_checkpoint(weights_path, map_location=torch.device('cpu'))
    model.eval()

    image_paths = sorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.jpeg')))
    print(f"Found {len(image_paths)} images.")

    for idx, img_path in enumerate(image_paths):
        print(f"Processing {img_path}...")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if image is too large for CPU inference (> 512x512)
        h, w = img.shape[:2]
        if h > 512 or w > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # Generate synthetic burst
        burst_transformation_params = {'max_translation': 24.0, 'max_rotation': 1.0, 'max_shear': 0.0, 'max_scale': 0.0, 'border_crop': 24}
        image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}
        
        # Generate burst_size=14 synthetic variations
        burst, frame_gt, burst_rgb, flow_vectors, meta_info = syn_burst_utils.rgb2rawburst(
            img_tensor, burst_size=14, downsample_factor=4,
            burst_transformation_params=burst_transformation_params,
            image_processing_params=image_processing_params,
            interpolation_type='bilinear'
        )

        burst = burst.unsqueeze(0) # (1, 14, 4, H, W)
        
        with torch.no_grad():
            net_pred = model(burst)
        
        # Output is (1, 3, 4H, 4W)
        net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2**14).cpu().numpy().astype(np.uint16)
        
        # Convert 14-bit to 8-bit for saving as standard image
        out_img = (net_pred_np / (2**14) * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        basename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, f"restored_{basename}")
        cv2.imwrite(out_path, out_img)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default='./Results/Custom/')
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    
    process_custom_images(args.input_dir, args.output_dir, args.weights)
