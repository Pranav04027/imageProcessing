import torch
import cv2
import os
import numpy as np
from Network import BIPNet
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB
from datasets.synthetic_burst_train_set import SyntheticBurst

def main():
    # 1. Load Model
    print("Loading model...")
    model = BIPNet()
    # Loading state dict
    state_dict = torch.load('Models/BIPNet.pth', map_location='cpu')
    
    # Check if 'state_dict' or 'model_state_dict' key exists
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[6:] if k.startswith('model.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded.")

    # 2. Load Dataset
    print("Loading samples from Zurich-RAW-to-DSLR-Dataset...")
    image_dir = "../Zurich-RAW-to-DSLR-Dataset"
    base_dataset = ZurichRAW2RGB(root=image_dir, split="test")
    
    if len(base_dataset) == 0:
        print("Warning: No images found in Zurich-RAW-to-DSLR-Dataset/test, trying train...")
        base_dataset = ZurichRAW2RGB(root=image_dir, split="train")

    if len(base_dataset) == 0:
        print("Error: No images found in Zurich-RAW-to-DSLR-Dataset")
        return

    burst_dataset = SyntheticBurst(base_dataset, burst_size=14, crop_sz=384)
    
    # 3. Setup Results Folder
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f"Results will be saved in: {os.path.abspath(results_dir)}")

    # 4. Process Multiple Images
    num_to_process = min(5, len(burst_dataset))
    print(f"Processing {num_to_process} images...")

    for i in range(num_to_process):
        print(f"--- Processing Image {i+1}/{num_to_process} ---")
        burst, frame_gt, flow_vectors, meta_info = burst_dataset[i]
        burst_in = burst.unsqueeze(0) # Add batch dim

        # Inference
        print("Running inference...")
        with torch.no_grad():
            output = model(burst_in)

        # Save Results
        print("Saving images...")
        # output: (1, 3, 4H, 4W)
        output_np = (output.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        # Original (Ground Truth HR image)
        gt_np = (frame_gt.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
        gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(results_dir, f'original_{i+1}.png'), gt_bgr)
        cv2.imwrite(os.path.join(results_dir, f'output_{i+1}.png'), output_bgr)
        
        print(f"Sample {i+1} saved: original_{i+1}.png, output_{i+1}.png")

    print(f"All {num_to_process} samples processed successfully.")

if __name__ == "__main__":
    main()
