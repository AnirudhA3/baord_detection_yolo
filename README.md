# board_detection_yolo
import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Parse arguments
parser = argparse.ArgumentParser(description="YOLO Image Detection Script")
parser.add_argument('--model', required=True, help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")')
parser.add_argument('--image', required=True, help='Path to input image file (example: "test.jpg")')
parser.add_argument('--thresh', type=float, default=0.5, help='Minimum confidence threshold for detection')
parser.add_argument('--save', action='store_true', help='Save the output image with bounding boxes')
args = parser.parse_args()

model_path = args.model
img_path = args.image
min_thresh = args.thresh
save_result = args.save

# Validate model and image paths
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    sys.exit(0)

if not os.path.exists(img_path):
    print(f"❌ Image file not found: {img_path}")
    sys.exit(0)

# Load YOLO model
print("🔄 Loading YOLO model...")
model = YOLO(model_path)

# Load image
print(f"📷 Running detection on image: {img_path}")
frame = cv2.imread(img_path)
if frame is None:
    print("❌ Failed to read the image. Please check the path.")
    sys.exit(0)

# Run inference
results = model(frame, verbose=False)
detections = results[0].boxes
labels = model.names

# Draw bounding boxes
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
               (88,159,106), (96,202,231), (159,124,168), (169,162,241),
               (98,118,150)]

object_count = 0

for i in range(len(detections)):
    conf = detections[i].conf.item()
    if conf < min_thresh:
        continue

    xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
    xmin, ymin, xmax, ymax = xyxy
    class_idx = int(detections[i].cls.item())
    class_name = labels[class_idx]
    color = bbox_colors[class_idx % len(bbox_colors)]

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    label = f'{class_name}: {int(conf * 100)}%'
    cv2.putText(frame, label, (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    object_count += 1

# Display results
print(f"✅ Detected {object_count} object(s)")
cv2.imshow('YOLO Detection Results', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result if requested
if save_result:
    output_path = os.path.splitext(img_path)[0] + '_detected.jpg'
    cv2.imwrite(output_path, frame)
    print(f"💾 Saved output image to: {output_path}")
