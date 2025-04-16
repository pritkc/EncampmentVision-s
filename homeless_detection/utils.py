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

def draw_predictions(image, boxes, labels, scores, score_thresh=0.5):
    """
    Draw bounding boxes and labels on an image (now using the adapter)
    
    Args:
        image (PIL.Image): Image to draw on
        boxes (np.ndarray): Bounding boxes
        labels (np.ndarray): Class labels
        scores (np.ndarray): Confidence scores
        score_thresh (float): Confidence threshold
        
    Returns:
        PIL.Image: Image with bounding boxes drawn
    """
    # Use the adapter's draw_predictions function
    return draw_predictions(image, boxes, labels, scores, score_thresh)

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

def create_detection_map(detections, center_lat, center_lon):
    """Create a map with detection markers"""
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
    
    # Add markers for each detection
    for d in detections:
        if all(k in d for k in ['lat', 'lon', 'class', 'confidence']):
            # Get color for class
            color = color_map.get(d['class'], 'gray')
            
            # Create popup content
            popup_html = f"""
            <div style="width:200px">
                <h4>{d['class']}</h4>
                <p>Confidence: {d['confidence']:.3f}</p>
                <p>Location: {d['lat']:.6f}, {d['lon']:.6f}</p>
            """
            
            # Add image if available
            if 'image_path' in d and os.path.exists(d['image_path']):
                try:
                    with open(d['image_path'], 'rb') as img_file:
                        encoded = base64.b64encode(img_file.read()).decode()
                        popup_html += f'<img src="data:image/jpeg;base64,{encoded}" width="100%">'
                except Exception as e:
                    print(f"Error encoding image: {str(e)}")
            
            popup_html += "</div>"
            
            # Add marker to cluster
            folium.Marker(
                location=[d['lat'], d['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{d['class']} ({d['confidence']:.2f})",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(marker_cluster)
    
    return m

def create_summary_charts(detections):
    """Create summary charts for detections"""
    if not detections:
        return None, None, {}
    
    # Extract data
    classes = []
    confidences = []
    for d in detections:
        if 'class' in d and 'confidence' in d:
            classes.append(d['class'])
            confidences.append(d['confidence'])
    
    if not classes:
        return None, None, {}
    
    # Count classes
    class_counts = {}
    for cls in classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Create pie chart
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'green', 'blue', 'yellow']
    wedges, texts, autotexts = ax.pie(
        list(class_counts.values()),
        labels=list(class_counts.keys()),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    ax.set_title('Detection Class Distribution', pad=20)
    
    # Create confidence histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(confidences, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Score Distribution')
    ax2.grid(True, alpha=0.3)
    
    return fig, fig2, class_counts

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 string
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert
        
    Returns:
        str: Base64 encoded string
    """
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return img_data 