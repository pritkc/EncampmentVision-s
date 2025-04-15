import streamlit as st
import os
import time
import requests
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import google_streetview.api
import folium
from streamlit_folium import folium_static
import base64
import io
import re
from datetime import datetime
import matplotlib.pyplot as plt
# Import utility functions from the homeless_detection package
from homeless_detection.utils import load_model as utils_load_model
from homeless_detection.utils import draw_predictions as utils_draw_predictions
from homeless_detection.utils import create_detection_map, create_summary_charts

# Set page config
st.set_page_config(
    page_title="Homeless Detection System",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and configurations
TEMP_DIR = "temp_images"
RESULTS_DIR = "results"
ORIGINAL_DIR = os.path.join(RESULTS_DIR, "original")
PREDICTED_DIR = os.path.join(RESULTS_DIR, "predicted")
CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Class mapping
LABEL_MAP = {
    1: "Homeless_People",
    2: "Homeless_Encampments",
    3: "Homeless_Cart",
    4: "Homeless_Bike"
}

# Sidebar for input parameters
st.sidebar.title("Homeless Detection System")
st.sidebar.info("This application detects homeless-related objects in Google Street View images")

# API Key input (with secure handling)
api_key = st.sidebar.text_input("Google Street View API Key", type="password")

# Area selection
st.sidebar.subheader("Area Selection")
top_left_lat = st.sidebar.number_input("Top Left Latitude", value=34.044133, format="%.6f")
top_left_lon = st.sidebar.number_input("Top Left Longitude", value=-118.243896, format="%.6f")
bottom_right_lat = st.sidebar.number_input("Bottom Right Latitude", value=34.038049, format="%.6f")
bottom_right_lon = st.sidebar.number_input("Bottom Right Longitude", value=-118.242965, format="%.6f")

# Grid dimensions
st.sidebar.subheader("Grid Dimensions")
num_rows = st.sidebar.slider("Number of Rows", min_value=2, max_value=10, value=5)
num_cols = st.sidebar.slider("Number of Columns", min_value=2, max_value=10, value=5)

# Confidence threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Model selection
# model_path = "model_final_2.pth"
# if not os.path.exists(model_path):
#     st.error(f"Model file {model_path} not found. Please make sure the model file is in the correct location.")

# Look for models in the models directory
models_dir = "models"
if not os.path.exists(models_dir):
    st.error(f"Models directory '{models_dir}' not found. Please create this directory.")
    model_path = None
    can_run_detection = False
else:
    # Find all .pth files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        st.error(f"No model files (.pth) found in '{models_dir}' directory. Please add a model file.")
        model_path = None
        can_run_detection = False
    else:
        # Always show model selector in sidebar
        st.sidebar.subheader("Model Selection")
        selected_model = st.sidebar.selectbox(
            "Select model to use:", 
            model_files
        )
        model_path = os.path.join(models_dir, selected_model)
        can_run_detection = True
        

# Main content area
st.title("Homeless Detection from Google Street View")
st.markdown("""
This application uses a trained Faster R-CNN model to detect homeless-related objects in Google Street View images.
The model can detect:
- Homeless People
- Homeless Encampments 
- Homeless Carts
- Homeless Bikes
""")

# Display a map with the selected area
def display_area_map():
    center_lat = (top_left_lat + bottom_right_lat) / 2
    center_lon = (top_left_lon + bottom_right_lon) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
    
    # Draw rectangle for the area
    folium.Rectangle(
        bounds=[(top_left_lat, top_left_lon), (bottom_right_lat, bottom_right_lon)],
        color='blue',
        fill=True,
        fill_opacity=0.2,
        tooltip="Selected Area"
    ).add_to(m)
    
    # Draw grid lines
    lat_step = (bottom_right_lat - top_left_lat) / (num_rows - 1) if num_rows > 1 else 0
    lon_step = (bottom_right_lon - top_left_lon) / (num_cols - 1) if num_cols > 1 else 0
    
    for i in range(num_rows):
        lat = top_left_lat + i * lat_step
        folium.PolyLine(
            locations=[(lat, top_left_lon), (lat, bottom_right_lon)],
            color='gray',
            weight=1,
            opacity=0.7
        ).add_to(m)
    
    for j in range(num_cols):
        lon = top_left_lon + j * lon_step
        folium.PolyLine(
            locations=[(top_left_lat, lon), (bottom_right_lat, lon)],
            color='gray',
            weight=1,
            opacity=0.7
        ).add_to(m)
    
    return m

# Display initial map
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Selected Area")
    area_map = display_area_map()
    folium_static(area_map)

with col2:
    st.subheader("Grid Information")
    st.info(f"""
    - Grid Size: {num_rows} x {num_cols} = {num_rows * num_cols} points
    - Latitude Range: {top_left_lat:.6f} to {bottom_right_lat:.6f}
    - Longitude Range: {top_left_lon:.6f} to {bottom_right_lon:.6f}
    - Points will be sampled at headings: 0° and 180°
    - Total API calls: {num_rows * num_cols * 2}
    """)

# Function to load model - use the utility function with Streamlit caching
@st.cache_resource
def load_model():
    try:
        if model_path is None:
            st.error("No valid model path available.")
            return None, None
        return utils_load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Function to process a single image
def process_image(image_path, model, device, lat, lon, heading, pano_id, date):
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
    
    detections = []
    if len(pred_boxes) > 0:
        # Save original image
        filename = f"streetview_{pano_id}_{date}_{lat}_{lon}_heading{heading}.jpg"
        original_path = os.path.join(ORIGINAL_DIR, filename)
        predicted_path = os.path.join(PREDICTED_DIR, filename)
        
        image.save(original_path)
        
        # Draw predictions using utility function
        pred_image = utils_draw_predictions(
            image.copy(), pred_boxes, pred_labels, pred_scores, confidence_threshold
        )
        pred_image.save(predicted_path)
        
        # Collect detection info
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            cls_name = LABEL_MAP.get(int(label), str(label))
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
    
    return detections

# Run detection process
def run_detection():
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your Google Street View API Key")
        return None
    
    # Generate grid points
    latitudes = [top_left_lat + i * (bottom_right_lat - top_left_lat) / (num_rows - 1) for i in range(num_rows)]
    longitudes = [top_left_lon + j * (bottom_right_lon - top_left_lon) / (num_cols - 1) for j in range(num_cols)]
    grid_points = [(lat, lon) for lat in latitudes for lon in longitudes]
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
        
    # Exit if model loading failed
    if model is None or device is None:
        st.error("Model loading failed. Cannot continue with detection.")
        return None
    
    # Initialize progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_placeholder = st.empty()
    
    # Initialize results container
    all_detections = []
    processed_images = 0
    total_images = len(grid_points) * 2  # 2 headings per point
    
    # Clear previous results
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    
    # Create CSV headers
    csv_fields = ["filename", "lat", "lon", "heading", "date", "class", "confidence"]
    pd.DataFrame(columns=csv_fields).to_csv(CSV_PATH, index=False)
    
    # Process each grid point
    for idx, (lat, lon) in enumerate(grid_points):
        for heading in [0, 180]:
            status_text.text(f"Processing location {idx+1}/{len(grid_points)}, heading {heading}°...")
            
            # Prepare Street View API parameters
            params = [{
                'size': '640x640',
                'location': f'{lat},{lon}',
                'heading': str(heading),
                'pitch': '0',
                'key': api_key
            }]
            
            try:
                # Get street view image
                results = google_streetview.api.results(params)
                
                if results.metadata[0]['status'] == 'OK':
                    metadata = results.metadata[0]
                    pano_id = metadata.get('pano_id', 'unknown')
                    date = metadata.get('date', datetime.now().strftime('%Y-%m'))
                    
                    # Download image
                    response = requests.get(results.links[0])
                    temp_image_path = os.path.join(TEMP_DIR, f"temp_{pano_id}.jpg")
                    with open(temp_image_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Process image
                    detections = process_image(temp_image_path, model, device, lat, lon, heading, pano_id, date)
                    all_detections.extend(detections)
                    
                    # Update CSV if detections found
                    if detections:
                        detection_df = pd.DataFrame(detections)
                        detection_df[csv_fields].to_csv(CSV_PATH, mode='a', header=False, index=False)
                        
                        status_text.text(f"✅ Detected {len(detections)} object(s) at {lat:.6f}, {lon:.6f}, heading {heading}°")
                    else:
                        status_text.text(f"❌ No detections at {lat:.6f}, {lon:.6f}, heading {heading}°")
                    
                    # Clean up temp file
                    os.remove(temp_image_path)
                    
                else:
                    status_text.text(f"No image available for {lat:.6f}, {lon:.6f}, heading {heading}°")
                
                # Update progress
                processed_images += 1
                progress_bar.progress(processed_images / total_images)
                
                # Add a small delay to avoid API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                status_text.text(f"Error processing {lat:.6f}, {lon:.6f}, heading {heading}°: {str(e)}")
                processed_images += 1
                progress_bar.progress(processed_images / total_images)
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("Processing completed.")
    
    return all_detections

# Function to display results on a map - leveraging utility function
def display_results_map(detections):
    if not detections:
        st.warning("No detections found in the selected area.")
        return
    
    # Create map with center on the selected area
    center_lat = (top_left_lat + bottom_right_lat) / 2
    center_lon = (top_left_lon + bottom_right_lon) / 2
    
    return create_detection_map(detections, center_lat, center_lon)

# Function to display summary statistics - leveraging utility function
def display_summary_stats(detections):
    if not detections:
        return None, None, {}
    
    return create_summary_charts(detections)

# Run button
if st.button("Run Detection", type="primary", disabled=not can_run_detection):
    with st.spinner("Running detection process..."):
        detections = run_detection()
    
    if detections:
        st.success(f"Detection completed. Found {len(detections)} objects in {len(set(d['filename'] for d in detections))} images.")
        
        # Display results on tabs
        tab1, tab2, tab3 = st.tabs(["Map", "Statistics", "Raw Data"])
        
        with tab1:
            st.subheader("Detection Map")
            results_map = display_results_map(detections)
            folium_static(results_map, width=1200, height=800)
        
        with tab2:
            st.subheader("Summary Statistics")
            if detections:
                fig, fig2, class_counts = display_summary_stats(detections)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Detection Counts")
                    for cls, count in class_counts.items():
                        st.metric(cls, count)
                
                st.subheader("Confidence Score Distribution")
                st.pyplot(fig2)
        
        with tab3:
            st.subheader("Raw Detection Data")
            st.dataframe(pd.DataFrame(detections))
            
            if os.path.exists(CSV_PATH):
                with open(CSV_PATH, "rb") as f:
                    csv_data = f.read()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="homeless_detections.csv",
                    mime="text/csv"
                )
    else:
        st.warning("No detections found in the selected area.")
elif not can_run_detection:
    st.error("Cannot run detection: No valid model available. Please add model file(s) to the 'models' directory.")

# Footer
st.markdown("---")
st.markdown("Homeless Detection System - Powered by Streamlit and PyTorch") 