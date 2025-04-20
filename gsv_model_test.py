#!/usr/bin/env python3
"""
Google Street View Homeless Detection

This script loads a trained Faster R-CNN model from the models directory to perform
homeless detection on Google Street View images for a given area.
"""

import os
import time
import requests
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import csv
import sys

try:
    import google_streetview.api
except ImportError:
    print("Error: google_streetview.api module not found.")
    print("Please install it with: pip install google-streetview")
    sys.exit(1)

# Google Street View API key
API_KEY = ""  # Replace with your actual API key

# Define bounding box coordinates (area to cover)
top_left = (34.044133, -118.243896)
bottom_right = (34.038049, -118.242965)

# Grid dimensions
num_rows = 5   # vertical (latitude)
num_cols = 10  # horizontal (longitude)

# Output directories
download_dir = "Google Street View Downloader/Test3"
original_dir = "Google Street View Downloader/Original_GSV_images"
predicted_dir = "Google Street View Downloader/Predicted_GSV_images"
csv_path = os.path.join(predicted_dir, "predictions.csv")

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(original_dir, exist_ok=True)
os.makedirs(predicted_dir, exist_ok=True)

# Class mapping
label_map = {
    1: "Homeless_People",
    2: "Homeless_Encampments",
    3: "Homeless_Cart",
    4: "Homeless_Bike"
}

def load_model():
    """
    Load the trained Faster R-CNN model from models directory.
    Handles various error conditions and model selection.
    """
    num_classes = 5  # Background + 4 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    
    # Look for models in the models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found.")
        print("Please create this directory and add a model file.")
        sys.exit(1)
    
    # Find all .pth files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"Error: No model files (.pth) found in '{models_dir}' directory.")
        print("Please add a model file.")
        sys.exit(1)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

def draw_predictions(image, boxes, labels, scores, score_thresh=0.5):
    """Draw bounding boxes and labels on the image for predictions."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        if scores[i] < score_thresh:
            continue
        xmin, ymin, xmax, ymax = box
        cls_id = int(labels[i])
        cls_name = label_map.get(cls_id, str(cls_id))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=3)
        draw.text((xmin, ymin - 10), f"{cls_name} {scores[i]:.2f}", fill=(255, 0, 0), font=font)
    return image

def main():
    if not API_KEY:
        print("Error: Please set your Google Street View API Key")
        sys.exit(1)
    
    # Load the model
    print("Loading model...")
    model, device = load_model()
    print(f"Model loaded successfully on {device}")
    
    # Generate evenly spaced grid points
    latitudes = [top_left[0] + i * (bottom_right[0] - top_left[0]) / (num_rows - 1) for i in range(num_rows)]
    longitudes = [top_left[1] + j * (bottom_right[1] - top_left[1]) / (num_cols - 1) for j in range(num_cols)]
    grid_points = [(lat, lon) for lat in latitudes for lon in longitudes]
    
    print(f"Generated {len(grid_points)} grid points")
    
    # CSV headers
    csv_fields = ["filename", "lat", "lon", "heading", "date", "class", "confidence"]
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
    
    # Run detection and logging
    total_points = len(grid_points) * 2  # 2 headings per point
    processed = 0
    
    for idx, (lat, lon) in enumerate(grid_points):
        for heading in [0, 180]:
            processed += 1
            print(f"Processing point {processed}/{total_points}: {lat}, {lon}, heading {heading}°")
            
            params = [{
                'size': '640x640',
                'location': f'{lat},{lon}',
                'heading': str(heading),
                'pitch': '0',
                'key': API_KEY
            }]
            
            try:
                results = google_streetview.api.results(params)
                
                metadata = results.metadata[0]
                pano_id = metadata.get('pano_id', 'unknown')
                date = metadata.get('date', 'unknown')
                filename = f"streetview_{pano_id}_{date}_{lat}_{lon}_heading{heading}.jpg"
                original_path = os.path.join(original_dir, filename)
                predicted_path = os.path.join(predicted_dir, filename)
                
                if results.links:
                    response = requests.get(results.links[0])
                    with open(original_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                    
                    image = Image.open(original_path).convert("RGB")
                    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        prediction = model(img_tensor)[0]
                    
                    pred_boxes = prediction["boxes"].cpu().numpy()
                    pred_labels = prediction["labels"].cpu().numpy()
                    pred_scores = prediction["scores"].cpu().numpy()
                    
                    if len(pred_boxes) > 0:
                        print(f"✅ Homeless entities detected in {filename}!")
                        image.save(original_path)
                        pred_image = draw_predictions(image.copy(), pred_boxes, pred_labels, pred_scores)
                        pred_image.save(predicted_path)
                        
                        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=csv_fields + ['boxes', 'labels'])
                            for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                                if score < 0.5:
                                    continue
                                cls_name = label_map.get(int(label), str(label))
                                # Convert box coordinates to integers for cleaner display
                                box_coords = [int(coord) for coord in box]
                                writer.writerow({
                                    "filename": filename,
                                    "lat": lat,
                                    "lon": lon,
                                    "heading": heading,
                                    "date": date,
                                    "class": cls_name,
                                    "confidence": round(float(score), 3),
                                    "boxes": box_coords,  # Store as integer coordinates
                                    "labels": int(label)  # Store the numeric label
                                })
                    else:
                        print(f"❌ No homeless-related objects detected in {filename}.")
                
                else:
                    print(f"No image available for {lat}, {lon}, heading {heading}")
                
                # Rate limit to avoid API throttling
                time.sleep(1)
            
            except Exception as e:
                print(f"Error processing {lat}, {lon}, heading {heading}: {e}")
    
    print(f"Processing completed. Results saved to {predicted_dir}")

if __name__ == "__main__":
    main() 