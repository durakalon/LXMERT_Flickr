import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

# Ensure output directory exists
OUTPUT_DIR = "flickr30k_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval() # Set to eval mode
        self.model.to(device)

        # Expose sub-modules for easier access
        self.backbone = self.model.backbone
        self.rpn = self.model.rpn
        self.roi_heads = self.model.roi_heads
        self.transform = self.model.transform

    def forward(self, images):
        # 1. Transform (Resizing, Normalization)
        # images is a list of tensors (0-1)
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, None)
        
        # 2. Backbone
        features = self.backbone(images.tensors)
        
        # 3. RPN
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
            
        proposals, _ = self.rpn(images, features, targets)
        
        # 4. RoI Heads (Extract features for ALL proposals first)
        # box_roi_pool expects: features, proposals, image_shapes
        box_features = self.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        
        # Pass through the box head (TwoMLPHead) -> [N, 1024]
        box_features = self.roi_heads.box_head(box_features)
        
        # Pass through predictor to get scores/boxes -> [N, num_classes]
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        
        # 5. Post-process to get final detections (NMS)
        # We need to mimic postprocess_detections but keep track of indices or features
        # To simplify, we will do a custom selection strategy here.
        # We want the features corresponding to the surviving boxes.
        
        result_feats = []
        result_boxes = []
        
        # We process batch_size=1 usually in this script, but let's handle the list
        boxes_per_image = [box.shape[0] for box in proposals]
        
        # Split back into per-image
        box_features_list = box_features.split(boxes_per_image, 0)
        class_logits_list = class_logits.split(boxes_per_image, 0)
        box_regression_list = box_regression.split(boxes_per_image, 0)
        
        for i in range(len(proposals)):
            # Per image logic
            box_feats = box_features_list[i]
            logits = class_logits_list[i]
            box_reg = box_regression_list[i]
            prop = proposals[i]
            image_shape = images.image_sizes[i]
            
            # Apply box regression to proposals
            boxes = self.roi_heads.box_coder.decode(box_reg, [prop])
            # Boxes are now [N, num_classes * 4]
            
            # Get scores
            scores = torch.softmax(logits, dim=-1)
            
            # Remove background (class 0)
            # scores[:, 1:] are object scores
            # Standard approach: iterate classes, apply NMS, collect.
            # Simplified approach for "Top K Features":
            # Just take the max score across classes (excluding background)
            
            obj_scores, obj_labels = torch.max(scores[:, 1:], dim=1) # [N]
            # Map back to real labels (1-indexed)
            obj_labels = obj_labels + 1
            
            # Pick the corresponding box for that best class
            # boxes shape: [N, C, 4] or [N, C*4]. torchvision is [N, C*4] usually flattened
            # box_regression: [N, C*4]
            # decoded boxes: `decode` returns a Tensor [N, C, 4] or similar depending on implementation
            # Actually torchvision `decode` returns boxes where the shape matches input box_regression roughly.
            # Let's check: box_coder.decode returns pred_boxes (N, C, 4) if input is (N, C*4) and proposals (N, 4).
            # Wait, torchvision's box_coder.decode(rel_codes, boxes)
            # if rel_codes is (N, C*4), result is (N, C*4).
            
            boxes = boxes.reshape(boxes.shape[0], -1, 4) # [N, C, 4]
            
            # We select the box corresponding to the max score class
            # gather
            # labels is [N], we need [N, 1, 4]
            idx = obj_labels.unsqueeze(1).unsqueeze(2).expand(-1, 1, 4)
            final_boxes = torch.gather(boxes, 1, idx).squeeze(1) # [N, 4]
            
            # Clip boxes
            final_boxes = torchvision.ops.clip_boxes_to_image(final_boxes, image_shape)
            
            # NMS (Class-agnostic for simplicity to get generic regions)
            keep = torchvision.ops.nms(final_boxes, obj_scores, 0.5)
            
            # Select top K from kept
            keep = keep[:36] # At most 36
            
            final_boxes = final_boxes[keep]
            final_feats = box_feats[keep]
            
            # Padding to 36
            num_boxes = final_boxes.shape[0]
            if num_boxes < 36:
                pad_len = 36 - num_boxes
                # Pad features with 0
                final_feats = torch.cat([final_feats, torch.zeros(pad_len, 1024).to(device)], dim=0)
                # Pad boxes with 0
                final_boxes = torch.cat([final_boxes, torch.zeros(pad_len, 4).to(device)], dim=0)
            elif num_boxes > 36:
                final_feats = final_feats[:36]
                final_boxes = final_boxes[:36]
                
            # Normalize boxes [x1, y1, x2, y2] / [W, H, W, H]
            # Note: image_shape is (H, W) in tuple
            h, w = original_image_sizes[i] 
            # Wait, the `final_boxes` are in the RESIZED coordinate space because we decoded against `proposals` which were generated on resized images.
            # We need to map them back to original size first OR normalize using the RESIZED W,H.
            # Normalization is invariant to scale if we use the W,H that the boxes correspond to.
            # The boxes from `box_coder` are in the `images` (resized) coordinate system.
            # So uses `images.image_sizes[i]` which is (H', W').
            
            h_resized, w_resized = image_shape
            scaler = torch.tensor([w_resized, h_resized, w_resized, h_resized], device=device)
            final_boxes = final_boxes / scaler
            
            result_feats.append(final_feats.cpu())
            result_boxes.append(final_boxes.cpu())
            
        return result_feats, result_boxes

def main():
    extractor = FeatureExtractor()
    
    # Glob images
    # Check nesting
    image_dir = "flickr30k_images/flickr30k_images"
    if not os.path.exists(image_dir):
        # Fallback to parent if not nested
        image_dir = "flickr30k_images"
    
    img_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"Found {len(img_paths)} images in {image_dir}")
    
    # Process
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    # Use simple loop
    for img_path in tqdm(img_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).to(device)
            
            with torch.no_grad():
                feats, boxes = extractor([img_tensor])
                
            # Save
            img_name = os.path.basename(img_path).split('.')[0]
            save_data = {
                "visual_feats": feats[0], # [36, 1024]
                "visual_pos": boxes[0]    # [36, 4]
            }
            torch.save(save_data, os.path.join(OUTPUT_DIR, f"{img_name}.pt"))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
