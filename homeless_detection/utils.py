import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import folium
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from .model_adapter import get_model_adapter, draw_predictions
from folium.plugins import MarkerCluster
import numpy as np
from collections import defaultdict

# Class mapping matching notebook
LABEL_MAP = {
    1: "Homeless_People",
    2: "Homeless_Encampments",
    3: "Homeless_Cart",
    4: "Homeless_Bike"
}

def load_model(model_path=None, device=None):
    """Load model using adapter"""
    adapter = get_model_adapter()
    model, device, _ = adapter.load_model(model_path, device)
    return adapter, device

def draw_single_prediction(image, box, label, score, score_thresh=0.5):
    """
    Draw a single bounding box and label on an image
    
    Args:
        image (PIL.Image): Image to draw on
        box (np.ndarray): Single bounding box
        label (int): Class label
        score (float): Confidence score
        score_thresh (float): Confidence threshold
        
    Returns:
        PIL.Image: Image with single bounding box drawn
    """
    if score < score_thresh:
        return image
    
    # Create a copy of the image
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Get color for this class
    adapter = get_model_adapter()
    color_map = adapter.get_color_map()
    class_map = adapter.get_class_map()
    
    cls_id = int(label)
    color = color_map.get(cls_id, (255, 255, 255))
    
    # Draw box with thicker outline
    x1, y1, x2, y2 = map(int, box)
    for thickness in range(3):
        draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                      outline=color)
    
    # Draw label with improved visibility
    class_name = class_map.get(cls_id, f"Class {cls_id}")
    text = f"{class_name}: {score:.2f}"
    text_size = draw.textbbox((0, 0), text, font=font)[2:4]
    
    # Draw label background
    draw.rectangle([x1, y1 - text_size[1] - 4, x1 + text_size[0], y1], 
                  fill=color)
    
    # Draw text with contrasting color
    text_color = (0, 0, 0) if cls_id in [2, 4] else (255, 255, 255)
    draw.text((x1, y1 - text_size[1] - 2), text, fill=text_color, font=font)
    
    return image

def draw_predictions(image, boxes, labels, scores, score_thresh=0.5, single_box_idx=None):
    """
    Draw bounding boxes and labels on an image
    
    Args:
        image (PIL.Image): Image to draw on
        boxes (np.ndarray): Bounding boxes
        labels (np.ndarray): Class labels
        scores (np.ndarray): Confidence scores
        score_thresh (float): Confidence threshold
        single_box_idx (int, optional): If provided, only draw this box index
        
    Returns:
        PIL.Image: Image with bounding boxes drawn
    """
    if single_box_idx is not None:
        if 0 <= single_box_idx < len(boxes):
            return draw_single_prediction(
                image,
                boxes[single_box_idx],
                labels[single_box_idx],
                scores[single_box_idx],
                score_thresh
            )
        return image
    
    # Draw all boxes
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    adapter = get_model_adapter()
    color_map = adapter.get_color_map()
    class_map = adapter.get_class_map()
    
    for i, box in enumerate(boxes):
        if scores[i] < score_thresh:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(labels[i])
        color = color_map.get(cls_id, (255, 255, 255))
        
        # Draw box with thicker outline
        for thickness in range(3):
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                         outline=color)
        
        # Draw label with improved visibility
        class_name = class_map.get(cls_id, f"Class {cls_id}")
        text = f"{class_name}: {scores[i]:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        
        # Draw label background
        draw.rectangle([x1, y1 - text_size[1] - 4, x1 + text_size[0], y1], 
                      fill=color)
        
        # Draw text with contrasting color
        text_color = (0, 0, 0) if cls_id in [2, 4] else (255, 255, 255)
        draw.text((x1, y1 - text_size[1] - 2), text, fill=text_color, font=font)
    
    return image

def process_image(image_path, model_adapter, device, confidence_threshold=0.5):
    """Process a single image with the detection model"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Get predictions using adapter
        pred_boxes, pred_labels, pred_scores = model_adapter.predict(
            image, 
            confidence_threshold=confidence_threshold
        )
        
        return pred_boxes, pred_labels, pred_scores, image
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return np.array([]), np.array([]), np.array([]), None

def process_detections_for_display(detections):
    """
    Process and deduplicate detections by image path.
    Returns a list of unique image entries with all their detections.
    """
    # Group detections by image path
    image_detections = {}
    
    for detection in detections:
        if not all(k in detection for k in ['image_path', 'lat', 'lon', 'class', 'confidence']):
            continue
            
        img_path = detection['image_path']
        if img_path not in image_detections:
            image_detections[img_path] = {
                'image_path': img_path,
                'lat': detection['lat'],
                'lon': detection['lon'],
                'detections': [],
                'total_confidence': 0,
                'detection_count': 0
            }
        
        image_detections[img_path]['detections'].append({
            'class': detection['class'],
            'confidence': detection['confidence']
        })
        image_detections[img_path]['total_confidence'] += detection['confidence']
        image_detections[img_path]['detection_count'] += 1
    
    return list(image_detections.values())

def create_detection_map(detections, center_lat, center_lon):
    """Create a map with detection markers - one marker per image"""
    # Create map centered on the area
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Color mapping matching notebook
    color_map = {
        "Homeless_People": "red",
        "Homeless_Encampments": "green",
        "Homeless_Cart": "blue",
        "Homeless_Bike": "yellow"
    }
    
    # Process detections to ensure one marker per image
    unique_image_detections = process_detections_for_display(detections)
    
    for image_data in unique_image_detections:
        if not os.path.exists(image_data['image_path']):
            continue
        
        # Create popup content with maximizable image
        popup_html = f"""
        <div style="width:200px">
        <h4>Detections at this location:</h4>
        <p>Total detections: {image_data['detection_count']}</p>
        """
        
        # Group detections by class for summary
        class_summary = {}
        for det in image_data['detections']:
            if det['class'] not in class_summary:
                class_summary[det['class']] = {
                    'count': 0,
                    'total_confidence': 0
                }
            class_summary[det['class']]['count'] += 1
            class_summary[det['class']]['total_confidence'] += det['confidence']
        
        # Add detection summary
        for cls, summary in class_summary.items():
            avg_confidence = summary['total_confidence'] / summary['count']
            color = color_map.get(cls, 'gray')
            popup_html += f'<p style="color:{color}"><strong>{cls}</strong>: {summary["count"]} found (avg conf: {avg_confidence:.2f})</p>'
        
        popup_html += f'<p>Location: {image_data["lat"]:.6f}, {image_data["lon"]:.6f}</p>'
        
        # Add maximizable image
        try:
            with open(image_data['image_path'], 'rb') as img_file:
                encoded = base64.b64encode(img_file.read()).decode()
                # Add image with click-to-maximize functionality
                popup_html += f'''
                <img src="data:image/jpeg;base64,{encoded}" 
                     width="100%" 
                     onclick="window.open(this.src)" 
                     style="cursor: pointer" 
                     title="Click to maximize">
                <div style="text-align: center; font-style: italic; margin-top: 5px;">
                    Click image to maximize
                </div>
                '''
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
        
        popup_html += "</div>"
        
        # Determine marker color based on most frequent class
        most_frequent_class = max(class_summary.items(), key=lambda x: x[1]['count'])[0]
        marker_color = color_map.get(most_frequent_class, 'gray')
        
        # Add marker
        folium.Marker(
            location=[image_data['lat'], image_data['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{image_data['detection_count']} detections",
            icon=folium.Icon(color=marker_color, icon='info-sign')
        ).add_to(marker_cluster)
    
    return m

def get_unique_detection_images(detections):
    """
    Get a list of unique images with their highest confidence detections.
    This is used for the grid display to avoid duplicates.
    """
    unique_images = {}
    
    for detection in detections:
        if not all(k in detection for k in ['image_path', 'class', 'confidence']):
            continue
            
        img_path = detection['image_path']
        if img_path not in unique_images or detection['confidence'] > unique_images[img_path]['confidence']:
            unique_images[img_path] = {
                'image_path': img_path,
                'class': detection['class'],
                'confidence': detection['confidence'],
                'lat': detection.get('lat', 0),
                'lon': detection.get('lon', 0)
            }
    
    return list(unique_images.values())

def prepare_grid_images(detections, max_images=None):
    """
    Prepare images for grid display, ensuring no duplicates.
    Returns a list of unique images with their best detection.
    """
    unique_images = get_unique_detection_images(detections)
    
    # Sort by confidence (highest first)
    unique_images.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Limit number of images if specified
    if max_images is not None:
        unique_images = unique_images[:max_images]
    
    return unique_images

def create_summary_charts(detections):
    """Create summary charts for detections"""
    if not detections:
        return None, None, {}
    
    # Process detections to ensure unique counting per image
    unique_image_detections = process_detections_for_display(detections)
    
    # Initialize counters
    class_counts = {}
    confidences_by_class = defaultdict(list)
    
    # Process each unique image
    for image_data in unique_image_detections:
        # Track classes seen in this image
        classes_in_image = set()
        class_confidences = defaultdict(list)
        
        # Process all detections for this image
        for detection in image_data['detections']:
            cls = detection['class']
            conf = detection['confidence']
            classes_in_image.add(cls)
            class_confidences[cls].append(conf)
        
        # Update global counts and confidences
        for cls in classes_in_image:
            class_counts[cls] = class_counts.get(cls, 0) + 1
            # Use average confidence for this class in this image
            avg_conf = sum(class_confidences[cls]) / len(class_confidences[cls])
            confidences_by_class[cls].append(avg_conf)
    
    if not class_counts:
        return None, None, {}
    
    # Set the style to a default matplotlib style
    plt.style.use('default')
    
    # Create pie chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['red', 'green', 'blue', 'yellow']
    wedges, texts, autotexts = ax1.pie(
        list(class_counts.values()),
        labels=list(class_counts.keys()),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    ax1.set_title('Detection Class Distribution (By Unique Images)', pad=20)
    
    # Create confidence histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot confidence distribution for each class
    for i, (cls, confs) in enumerate(confidences_by_class.items()):
        if confs:  # Only plot if we have confidences for this class
            ax2.hist(confs, bins=20, range=(0, 1), alpha=0.5, 
                    label=cls, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Average Confidence Score per Image')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Score Distribution by Class')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Close the figures to free memory
    plt.close(fig1)
    plt.close(fig2)
    
    return fig1, fig2, class_counts

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 string
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert
        
    Returns:
        str: Base64 encoded string
    """
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return img_data 