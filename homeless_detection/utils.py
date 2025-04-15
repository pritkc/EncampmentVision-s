import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import folium
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Class mapping
LABEL_MAP = {
    1: "Homeless_People",
    2: "Homeless_Encampments",
    3: "Homeless_Cart",
    4: "Homeless_Bike"
}

def load_model(model_path, device=None):
    """
    Load the trained model from path
    
    Args:
        model_path (str): Path to the model file
        device (torch.device): Device to load model on
        
    Returns:
        model: Loaded PyTorch model
        device: Device the model is loaded on
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 5  # Background + 4 classes
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def draw_predictions(image, boxes, labels, scores, score_thresh=0.5):
    """
    Draw bounding boxes and labels on an image
    
    Args:
        image (PIL.Image): Image to draw on
        boxes (np.ndarray): Bounding boxes
        labels (np.ndarray): Class labels
        scores (np.ndarray): Confidence scores
        score_thresh (float): Confidence threshold
        
    Returns:
        PIL.Image: Image with bounding boxes drawn
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        if scores[i] < score_thresh:
            continue
        xmin, ymin, xmax, ymax = box
        cls_id = int(labels[i])
        cls_name = LABEL_MAP.get(cls_id, str(cls_id))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=3)
        draw.text((xmin, ymin - 10), f"{cls_name} {scores[i]:.2f}", fill=(255, 0, 0), font=font)
    
    return image

def process_image(image_path, model, device, confidence_threshold=0.5):
    """
    Process a single image with the detection model
    
    Args:
        image_path (str): Path to the image file
        model: PyTorch model
        device: Device the model is loaded on
        confidence_threshold (float): Confidence threshold
        
    Returns:
        tuple: (pred_boxes, pred_labels, pred_scores)
    """
    image = Image.open(image_path).convert("RGB")
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    pred_boxes = prediction["boxes"].cpu().numpy()
    pred_labels = prediction["labels"].cpu().numpy()
    pred_scores = prediction["scores"].cpu().numpy()
    
    # Filter by confidence threshold
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]
    
    return pred_boxes, pred_labels, pred_scores, image

def create_detection_map(detections, center_lat=None, center_lon=None):
    """
    Create a folium map with detection markers
    
    Args:
        detections (list): List of detection dictionaries
        center_lat (float): Center latitude for the map
        center_lon (float): Center longitude for the map
        
    Returns:
        folium.Map: Map with detection markers
    """
    if not detections:
        return None
    
    # Group detections by location and heading
    location_groups = {}
    for det in detections:
        key = f"{det['lat']}_{det['lon']}_{det['heading']}"
        if key not in location_groups:
            location_groups[key] = {
                "lat": det["lat"],
                "lon": det["lon"],
                "heading": det["heading"],
                "filename": det["filename"],
                "image_path": det["image_path"],
                "detections": []
            }
        location_groups[key]["detections"].append(det)
    
    # Determine map center
    if center_lat is None or center_lon is None:
        first_loc = next(iter(location_groups.values()))
        center_lat = first_loc["lat"]
        center_lon = first_loc["lon"]
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
    
    # Add markers for each location with detections
    for key, info in location_groups.items():
        # Count detections by class
        class_counts = {}
        for det in info["detections"]:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Create popup content
        popup_html = f"""
        <div style="width:640px; text-align:center;">
            <h3>Detections at {info['lat']:.6f}, {info['lon']:.6f}</h3>
            <p>Heading: {info['heading']}°</p>
            <p><b>Detections:</b> {len(info['detections'])} object(s)</p>
            <ul style="text-align:left;">
        """
        
        for cls, count in class_counts.items():
            popup_html += f"<li>{cls}: {count}</li>"
        
        popup_html += "</ul>"
        
        # Add image to popup if it exists
        if "image_path" in info and os.path.exists(info["image_path"]):
            with open(info["image_path"], "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
                popup_html += f'<img src="data:image/jpeg;base64,{encoded_image}" width="600">'
        
        popup_html += "</div>"
        
        # Create popup
        popup = folium.Popup(folium.Html(popup_html, script=True), max_width=650)
        
        # Determine marker color based on detection classes
        color = "red"
        if "Homeless_Encampments" in class_counts:
            color = "darkred"
        elif "Homeless_People" in class_counts:
            color = "red"
        elif "Homeless_Cart" in class_counts or "Homeless_Bike" in class_counts:
            color = "orange"
        
        # Add marker to map
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=popup,
            icon=folium.Icon(color=color, icon="info-sign"),
            tooltip=f"{sum(class_counts.values())} detections"
        ).add_to(m)
    
    return m

def create_summary_charts(detections):
    """
    Create summary charts for detections
    
    Args:
        detections (list): List of detection dictionaries
        
    Returns:
        tuple: (pie_chart, histogram, class_counts)
    """
    if not detections:
        return None, None, {}
    
    # Create DataFrame
    df = pd.DataFrame(detections)
    
    # Count by class
    class_counts = df['class'].value_counts().to_dict()
    
    # Create pie chart
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(
        class_counts.values(),
        labels=class_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    )
    ax1.axis('equal')
    ax1.set_title("Detection Distribution by Class")
    
    # Create confidence histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df['confidence'], bins=10, range=(0, 1), color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Number of Detections')
    ax2.set_title('Distribution of Confidence Scores')
    
    return fig1, fig2, class_counts

def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64-encoded image
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert
        
    Returns:
        str: Base64-encoded image
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str 