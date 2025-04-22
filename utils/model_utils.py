"""
Utility functions for the model page - extracted from app.py to prevent
conflicts with st.set_page_config
"""
import os
import time
import torch
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import folium
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import traceback
import google_streetview.api
import torchvision.transforms as transforms
from pathlib import Path
import uuid
import inspect
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ORIGINAL_DIR = os.path.join(RESULTS_DIR, "original")
PREDICTED_DIR = os.path.join(RESULTS_DIR, "predicted")
CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# Label mapping
LABEL_MAP = {
    1: "People",
    2: "Encampments",
    3: "Cart",
    4: "Bike"
}

# Category colors
CATEGORY_COLORS = {
    1: (255, 0, 0),    # Red for People
    2: (0, 255, 0),    # Green for Encampments
    3: (0, 0, 255),    # Blue for Carts
    4: (255, 165, 0)   # Orange for Bikes
}

# Short labels
SHORT_LABELS = {
    1: "Person",
    2: "Camp",
    3: "Cart",
    4: "Bike" 
}

def ensure_file_exists(file_path, max_retries=3, retry_delay=0.5):
    """Check if a file exists with retries"""
    for i in range(max_retries):
        if os.path.exists(file_path):
            return True
        if i < max_retries - 1:  # Don't sleep on the last try
            time.sleep(retry_delay)
    return False

def draw_predictions_with_colors(image, boxes, labels, scores, threshold=0.5):
    """Draw bounding boxes on the image with different colors per category"""
    draw = ImageDraw.Draw(image)
    
    # Try to load a smaller font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        try:
            # Try to load system font
            font = ImageFont.load_default()
        except:
            # Create a very simple font as fallback
            font = None
    
    # Draw each prediction
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            # Get category ID and corresponding color
            category_id = int(label)
            color = CATEGORY_COLORS.get(category_id, (255, 0, 0))  # Default to red if not found
            
            # Draw box with category-specific color
            box = [int(i) for i in box]
            draw.rectangle(box, outline=color, width=2)  # Thinner line
            
            # Use a shorter label
            short_name = SHORT_LABELS.get(category_id, str(category_id))
            text = f"{short_name}:{score:.2f}"
            
            # Get text size
            if font:
                text_size = draw.textbbox((0, 0), text, font=font)[2:4]
            else:
                # Estimate size if no font available
                text_size = (len(text) * 6, 12)
            
            # Create label background
            bg_box = (box[0], box[1] - text_size[1] - 2, box[0] + text_size[0] + 4, box[1])
            
            # Determine text color based on background
            if category_id in [2, 4]:  # Green and Orange
                text_color = (0, 0, 0)  # Black text for better contrast
            else:
                text_color = (255, 255, 255)  # White text
            
            # Draw label above the box instead of inside
            draw.rectangle(bg_box, fill=color)
            if font:
                draw.text((box[0] + 2, box[1] - text_size[1] - 2), text, fill=text_color, font=font)
            else:
                draw.text((box[0] + 2, box[1] - text_size[1] - 2), text, fill=text_color)
    
    return image

def load_model():
    """Load the detection model"""
    try:
        logger.info("================== MODEL LOADING START ==================")
        logger.info("Attempting to load model")
        
        # Import here to avoid circular imports
        from detection_system.model_adapter import get_model_adapter, create_custom_fasterrcnn_with_bn
        
        # Initialize model adapter
        logger.info("Creating ModelAdapter instance")
        model_adapter = get_model_adapter()
        logger.info(f"ModelAdapter created: {type(model_adapter)}")
        
        # First, look specifically for the preferred model
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Directory exists: {os.path.exists(models_dir)}")
        
        # List all files in models directory
        if os.path.exists(models_dir):
            all_files = os.listdir(models_dir)
            logger.info(f"All files in models directory: {all_files}")
            pth_files = [f for f in all_files if f.endswith('.pth')]
            logger.info(f"PTH files found: {pth_files}")
        else:
            logger.error("Models directory does not exist")
            return None, None
        
        # Use the first available model
        if pth_files:
            model_path = os.path.join(models_dir, pth_files[0])
            logger.info(f"Using model: {model_path}")
        else:
            logger.error("No .pth model files found")
            return None, None
        
        # Load the model with direct approach
        with st.spinner(f"Loading model, please wait..."):
            logger.info("Creating model directly using create_custom_fasterrcnn_with_bn")
            
            # Create model instance
            model = create_custom_fasterrcnn_with_bn(num_classes=5)
            logger.info(f"Model created: {type(model)}")
            
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Load weights
            logger.info(f"Loading weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            
            # Load the state dict
            logger.info("Loading state dict into model")
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("State dict loaded with strict=True")
            except Exception as strict_error:
                logger.warning(f"Strict loading failed: {str(strict_error)}")
                logger.info("Attempting to load with strict=False")
                model.load_state_dict(state_dict, strict=False)
                logger.info("State dict loaded with strict=False")
            
            # Set model to evaluation mode
            model.eval()
            model.to(device)
            logger.info("Model evaluation mode set")
            
            # Create a simple adapter-like wrapper
            class SimpleAdapter:
                def __init__(self, model, device):
                    self.model = model
                    self.device = device
                    self.transform = transforms.Compose([
                        transforms.Resize([512, 512]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                
                def predict(self, image, confidence_threshold=0.3):
                    # Preprocess image
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                    
                    # Transform and add batch dimension
                    img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Run inference
                    with torch.no_grad():
                        predictions = self.model(img_tensor)
                        prediction = predictions[0]
                        
                        # Extract predictions
                        pred_boxes = prediction["boxes"].cpu().numpy()
                        pred_labels = prediction["labels"].cpu().numpy()
                        pred_scores = prediction["scores"].cpu().numpy()
                        
                        # Filter by confidence
                        mask = pred_scores >= confidence_threshold
                        pred_boxes = pred_boxes[mask]
                        pred_labels = pred_labels[mask]
                        pred_scores = pred_scores[mask]
                        
                        return pred_boxes, pred_labels, pred_scores
                
                def get_class_map(self):
                    return {
                        1: "People",
                        2: "Encampments",
                        3: "Cart",
                        4: "Bike"
                    }
            
            # Create adapter wrapper
            adapter = SimpleAdapter(model, device)
            logger.info("Created SimpleAdapter wrapper")
            
            logger.info("Model loaded successfully")
            logger.info("================== MODEL LOADING END ==================")
            return adapter, device
            
    except Exception as e:
        logger.error(f"Error in load_model function: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("================== MODEL LOADING FAILED ==================")
        return None, None

def process_image(image_path, model_adapter, device, lat, lon, heading, pano_id, date):
    """Process a single image with the model"""
    try:
        logger.info("================= IMAGE PROCESSING START =================")
        logger.info(f"Processing image at {lat:.6f}, {lon:.6f}, heading {heading}°")
        
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return []
            
        # Load and prepare image
        try:
            logger.info("Loading image with PIL")
            image = Image.open(image_path).convert("RGB")
            
            # Resize for model
            target_size = (512, 512)
            logger.info(f"Resizing to {target_size}")
            original_size = image.size
            image_resized = transforms.Resize(target_size)(image)
        except Exception as img_error:
            logger.error(f"Error loading or resizing image: {str(img_error)}")
            return []
        
        # Run model prediction
        try:
            logger.info("Running model prediction")
            base_confidence = 0.1  # Lower threshold to catch any potential detections
            
            # Run prediction
            logger.info("Calling model_adapter.predict()")
            pred_boxes, pred_labels, pred_scores = model_adapter.predict(image_resized, confidence_threshold=base_confidence)
            
        except Exception as pred_error:
            logger.error(f"Error during model prediction: {str(pred_error)}")
            return []
        
        # Get class map
        try:
            logger.info("Getting class map")
            if hasattr(model_adapter, 'get_class_map'):
                class_map = model_adapter.get_class_map()
            else:
                class_map = {
                    1: "People",
                    2: "Encampments",
                    3: "Cart",
                    4: "Bike"
                }
        except Exception:
            class_map = {i: f"Class_{i}" for i in range(1, 5)}
        
        # Filter predictions by category and threshold
        selected_categories = getattr(st.session_state, 'selected_categories', {1: True, 2: True, 3: True, 4: True})
        category_thresholds = getattr(st.session_state, 'category_thresholds', {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5})
        
        filtered_indices = []
        for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
            category_id = int(label)
            # Check if category is selected and meets its threshold
            if (category_id in selected_categories and 
                selected_categories[category_id] and
                score >= category_thresholds[category_id]):
                filtered_indices.append(i)
        
        # Apply filters
        if filtered_indices:
            pred_boxes = pred_boxes[filtered_indices]
            pred_labels = pred_labels[filtered_indices]
            pred_scores = pred_scores[filtered_indices]
        else:
            pred_boxes = np.array([])
            pred_labels = np.array([])
            pred_scores = np.array([])
        
        # Process detections
        detections = []
        if len(pred_boxes) > 0:
            # Create paths for output images
            filename = f"streetview_{pano_id}_{date}_{lat}_{lon}_heading{heading}.jpg"
            original_path = os.path.join(ORIGINAL_DIR, filename)
            predicted_path = os.path.join(PREDICTED_DIR, filename)
            
            # Save original image
            try:
                original_image = Image.open(image_path).convert("RGB")
                original_image.save(original_path)
            except Exception as save_error:
                logger.error(f"Error saving original image: {str(save_error)}")
            
            # Draw and save prediction image
            try:
                # Convert boxes back to original image size if necessary
                if original_size != target_size:
                    scale_x = original_size[0] / target_size[0]
                    scale_y = original_size[1] / target_size[1]
                    
                    scaled_boxes = []
                    for box in pred_boxes:
                        x1, y1, x2, y2 = box
                        scaled_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        scaled_boxes.append(scaled_box)
                    
                    pred_boxes = np.array(scaled_boxes)
                
                # Draw predictions
                pred_image = draw_predictions_with_colors(
                    original_image.copy(), pred_boxes, pred_labels, pred_scores, 0.0
                )
                
                # Save prediction image
                pred_image.save(predicted_path)
                
            except Exception as draw_error:
                logger.error(f"Error drawing or saving predicted image: {str(draw_error)}")
                predicted_path = original_path  # Fallback to original image
            
            # Create detection records
            if os.path.exists(predicted_path):
                for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                    cls_id = int(label)
                    cls_name = class_map.get(cls_id, str(cls_id))
                    detections.append({
                        "filename": filename,
                        "lat": lat,
                        "lon": lon,
                        "heading": heading,
                        "date": date,
                        "class": cls_name,
                        "confidence": round(float(score), 3),
                        "image_path": predicted_path
                    })
        
        logger.info("================= IMAGE PROCESSING END =================")
        return detections
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        logger.info("================= IMAGE PROCESSING FAILED =================")
        return []

def run_detection():
    """Run detection on the current area settings"""
    # This function is stubbed - implement fully if needed
    return None

def track_resource(resource_path, resource_type="file"):
    """Log and track a resource path"""
    if resource_path and os.path.exists(resource_path):
        stats = os.stat(resource_path)
        logger.info(
            f"RESOURCE TRACK [{resource_type}]: {resource_path} "
            f"(exists={os.path.exists(resource_path)}, "
            f"size={stats.st_size}, modified={stats.st_mtime})"
        )
    else:
        logger.warning(f"RESOURCE TRACK [{resource_type}]: {resource_path} (missing)")
    return resource_path

def display_results_map(detections):
    """Display detection results on a map"""
    # This function is stubbed - implement fully if needed
    return None, None, None

def display_summary_stats(detections):
    """Display summary statistics of detections"""
    # This function is stubbed - implement fully if needed
    return None, None, {}

@st.cache_data(ttl=3600, show_spinner=False)
def generate_cached_map_html(detections, center_lat, center_lon):
    """Generate and cache the map HTML to prevent it from disappearing"""
    # This function is stubbed - implement fully if needed
    return None

def display_cached_map(cached_map):
    """Display the cached map"""
    # This function is stubbed - implement fully if needed
    pass 