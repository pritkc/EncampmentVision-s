import streamlit as st
import os
import time
import requests
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import google_streetview.api
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import base64
import io
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
import shutil
import logging
import sys
import types
import warnings
import asyncio
import inspect
import uuid
import tempfile
from pathlib import Path
from io import BytesIO
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug mode - set to True to see more detailed error messages
DEBUG = False

# Fix PyTorch path errors - more aggressive approach
import sys
import types

# Create a safe module proxy that handles the __path__ attribute correctly
class SafeModuleProxy:
    def __init__(self, real_module):
        self.real_module = real_module
        
    def __getattr__(self, name):
        try:
            if name == '__path__':
                # Silently return empty path without logging
                return types.SimpleNamespace(_path=[])
            return getattr(self.real_module, name)
        except Exception:
            # Suppress all errors from path access
            if name == '__path__':
                return types.SimpleNamespace(_path=[])
            raise

# Suppress specific PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Apply the fix to torch._classes
try:
    if hasattr(torch, '_classes'):
        torch._classes = SafeModuleProxy(torch._classes)
        sys.modules['torch._classes'] = torch._classes
except Exception:
    # Silently handle any errors during path fix
    pass

# Import utility functions from the homeless_detection package
try:
    from homeless_detection.utils import load_model as utils_load_model
    from homeless_detection.utils import draw_predictions as utils_draw_predictions
    from homeless_detection.utils import create_detection_map, create_summary_charts
    if DEBUG:
        st.write("Successfully imported homeless_detection utilities")
except Exception as e:
    st.error(f"Error importing homeless_detection utilities: {str(e)}")
    if DEBUG:
        st.write(f"Detailed error: {str(e)}")

# Set page config
st.set_page_config(
    page_title="Homeless Detection System",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state debugging
if st.session_state:
    logger.info("Current session state keys: %s", list(st.session_state.keys()))

# Constants and configurations
TEMP_DIR = "temp_images"
RESULTS_DIR = "results"
ORIGINAL_DIR = os.path.join(RESULTS_DIR, "original")
PREDICTED_DIR = os.path.join(RESULTS_DIR, "predicted")
CSV_PATH = os.path.join(RESULTS_DIR, "predictions.csv")

# Initialize persistent directories
for directory in [TEMP_DIR, RESULTS_DIR, ORIGINAL_DIR, PREDICTED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Only clean temp directory if it's a new session
if 'initialized' not in st.session_state:
    logger.info("New session detected, cleaning temporary directories")
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Error cleaning temp file {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning temp directory: {e}")
    
    st.session_state.initialized = True
    logger.info("Session initialized")

# Add session state tracking
logger.info("=== Session State Snapshot ===")
logger.info("Initialization status: %s", st.session_state.get('initialized', False))
logger.info("Has detection results: %s", 'detection_results' in st.session_state)
if 'detection_results' in st.session_state and st.session_state.detection_results is not None:
    logger.info("Number of detections: %d", len(st.session_state.detection_results))
elif 'detection_results' in st.session_state:
    logger.info("Detection results exist but are None")
logger.info("Current state keys: %s", list(st.session_state.keys()))
logger.info("=== End Snapshot ===")

# Add async state tracking
def log_async_state():
    try:
        loop = asyncio.get_event_loop()
        logger.info("=== Async State ===")
        logger.info("Has event loop: %s", loop is not None)
        logger.info("Loop running: %s", loop.is_running() if loop else "No loop")
        logger.info("Loop closed: %s", loop.is_closed() if loop else "No loop")
        logger.info("=== End Async State ===")
    except Exception as e:
        logger.error("Error checking async state: %s", str(e))

# Function to ensure file persistence
def ensure_file_exists(file_path, max_retries=3, retry_delay=0.5):
    for i in range(max_retries):
        if os.path.exists(file_path):
            return True
        if i < max_retries - 1:  # Don't sleep on the last try
            time.sleep(retry_delay)
    return False

# Class mapping
LABEL_MAP = {
    1: "Homeless_People",
    2: "Homeless_Encampments",
    3: "Homeless_Cart",
    4: "Homeless_Bike"
}

# Category display names (more user-friendly)
CATEGORY_DISPLAY = {
    1: "Homeless People",
    2: "Homeless Encampments",
    3: "Homeless Carts",
    4: "Homeless Bikes"
}

# Sidebar for input parameters
st.sidebar.title("Homeless Detection System")
st.sidebar.info("This application detects homeless-related objects in Google Street View images")

# API Key input (with secure handling)
api_key = st.sidebar.text_input("Google Street View API Key", type="password")

# Area selection
st.sidebar.subheader("Area Selection")

# Create session state for coordinates if not exists
if 'top_left_lat' not in st.session_state:
    st.session_state.top_left_lat = 34.044133
if 'top_left_lon' not in st.session_state:
    st.session_state.top_left_lon = -118.243896
if 'bottom_right_lat' not in st.session_state:
    st.session_state.bottom_right_lat = 34.038049
if 'bottom_right_lon' not in st.session_state:
    st.session_state.bottom_right_lon = -118.242965

# Callback to update coordinate values
def handle_bbox_selection(bbox_coords):
    if bbox_coords:
        try:
            coords = json.loads(bbox_coords)
            # Round coordinates to 6 decimal places for consistency
            st.session_state.top_left_lat = round(float(coords.get('topLeftLat', st.session_state.top_left_lat)), 6)
            st.session_state.top_left_lon = round(float(coords.get('topLeftLon', st.session_state.top_left_lon)), 6)
            st.session_state.bottom_right_lat = round(float(coords.get('bottomRightLat', st.session_state.bottom_right_lat)), 6)
            st.session_state.bottom_right_lon = round(float(coords.get('bottomRightLon', st.session_state.bottom_right_lon)), 6)
            st.rerun()
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error parsing coordinates: {str(e)}")

# Callback for number input changes
def update_coordinate(coordinate_name):
    # Just for triggering session state update
    pass

# Coordinate input fields
top_left_lat = st.sidebar.number_input("Top Left Latitude", value=st.session_state.top_left_lat, format="%.6f", key="top_left_lat_input", on_change=update_coordinate, args=("top_left_lat",))
top_left_lon = st.sidebar.number_input("Top Left Longitude", value=st.session_state.top_left_lon, format="%.6f", key="top_left_lon_input", on_change=update_coordinate, args=("top_left_lon",))
bottom_right_lat = st.sidebar.number_input("Bottom Right Latitude", value=st.session_state.bottom_right_lat, format="%.6f", key="bottom_right_lat_input", on_change=update_coordinate, args=("bottom_right_lat",))
bottom_right_lon = st.sidebar.number_input("Bottom Right Longitude", value=st.session_state.bottom_right_lon, format="%.6f", key="bottom_right_lon_input", on_change=update_coordinate, args=("bottom_right_lon",))

# Update session state variables when inputs change
st.session_state.top_left_lat = top_left_lat
st.session_state.top_left_lon = top_left_lon 
st.session_state.bottom_right_lat = bottom_right_lat
st.session_state.bottom_right_lon = bottom_right_lon

# Add an instruction about drawing on the map
st.sidebar.info("You can also draw a bounding box directly on the map using the draw tools. The coordinates will be automatically updated.")

# Category selection
st.sidebar.subheader("Detection Categories")
st.sidebar.markdown("Select which categories to detect:")

# Initialize session state for category selection if not exists
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = {1: True, 2: True, 3: True, 4: True}

# Initialize session state for thresholds if not exists
if 'category_thresholds' not in st.session_state:
    st.session_state.category_thresholds = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}

# Create checkboxes for each category
for category_id, display_name in CATEGORY_DISPLAY.items():
    st.session_state.selected_categories[category_id] = st.sidebar.checkbox(
        display_name, 
        value=st.session_state.selected_categories[category_id]
    )
    
    # Only show threshold slider if category is selected
    if st.session_state.selected_categories[category_id]:
        st.session_state.category_thresholds[category_id] = st.sidebar.slider(
            f"{display_name} Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.category_thresholds[category_id], 
            step=0.05
        )

# Legacy confidence threshold - hidden for backwards compatibility
confidence_threshold = 0.5  # Default value for backwards compatibility

# Grid dimensions
st.sidebar.subheader("Grid Dimensions")
num_rows = st.sidebar.slider("Number of Rows", min_value=2, max_value=10, value=5)
num_cols = st.sidebar.slider("Number of Columns", min_value=2, max_value=10, value=5)

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
""")

# Display selected categories
st.subheader("Selected Detection Categories")
selected_categories = [CATEGORY_DISPLAY[cat_id] for cat_id, selected in st.session_state.selected_categories.items() if selected]

if selected_categories:
    cols = st.columns(len(selected_categories))
    for i, cat_name in enumerate(selected_categories):
        with cols[i]:
            cat_id = [k for k, v in CATEGORY_DISPLAY.items() if v == cat_name][0]
            st.metric(
                cat_name, 
                f"{st.session_state.category_thresholds[cat_id]:.2f}",
                "threshold"
            )
else:
    st.warning("Please select at least one category to detect")

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
    
    # Add drawing capabilities to the map
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': True,
        },
        edit_options={
            'edit': True,
            'remove': True
        }
    )
    draw.add_to(m)
    
    return m

# Display initial map
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Selected Area")
    st.markdown("""
    **Draw a bounding box on the map:**
    1. Click the rectangle tool (☐) in the toolbar
    2. Click and drag on the map to draw a box
    3. Edit the box by clicking the edit tool and dragging the corners
    4. The coordinates will automatically update in the sidebar
    """)
    area_map = display_area_map()
    
    # Replace folium_static with st_folium
    map_data = st_folium(area_map, width=800, returned_objects=["all_drawings"])
    
    # Handle map data for bounding box updates
    if map_data is not None and "all_drawings" in map_data:
        drawings = map_data["all_drawings"]
        if drawings:  # If there are any drawings
            last_drawing = drawings[-1]  # Get the most recent drawing
            if "geometry" in last_drawing and "coordinates" in last_drawing["geometry"]:
                coords = last_drawing["geometry"]["coordinates"][0]  # Get coordinates
                # Extract bounds from the rectangle coordinates
                lats = [coord[1] for coord in coords]
                lons = [coord[0] for coord in coords]
                
                bbox_data = {
                    "topLeftLat": max(lats),
                    "topLeftLon": min(lons),
                    "bottomRightLat": min(lats),
                    "bottomRightLon": max(lons)
                }
                handle_bbox_selection(json.dumps(bbox_data))

with col2:
    st.subheader("Grid Information")
    st.info(f"""
    - Grid Size: {num_rows} x {num_cols} = {num_rows * num_cols} points
    - Latitude Range: {top_left_lat:.6f} to {bottom_right_lat:.6f}
    - Longitude Range: {top_left_lon:.6f} to {bottom_right_lon:.6f}
    - Points will be sampled at headings: 0° and 180°
    - Total API calls: {num_rows * num_cols * 2}
    """)
    
    # Add a tips section
    st.markdown("### Tips")
    st.markdown("""
    - Draw a rectangle on the map to set the area
    - The coordinates will update automatically in the sidebar
    - You can fine-tune coordinates by editing the numbers in the sidebar
    - Keep the shape rectangular for accurate grid calculations
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
    try:
        if not ensure_file_exists(image_path):
            logger.error(f"Image file not found after retries: {image_path}")
            return []
        
        image = Image.open(image_path).convert("RGB")
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_labels = prediction["labels"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()
        
        # Filter by selected categories and their thresholds
        filtered_indices = []
        for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
            category_id = int(label)
            # Check if category is selected and meets its threshold
            if (category_id in st.session_state.selected_categories and 
                st.session_state.selected_categories[category_id] and
                score >= st.session_state.category_thresholds[category_id]):
                filtered_indices.append(i)
        
        # Apply filters
        if filtered_indices:
            pred_boxes = pred_boxes[filtered_indices]
            pred_labels = pred_labels[filtered_indices]
            pred_scores = pred_scores[filtered_indices]
        else:
            # No detections that meet criteria
            pred_boxes = np.array([])
            pred_labels = np.array([])
            pred_scores = np.array([])
        
        detections = []
        if len(pred_boxes) > 0:
            # Create unique filename based on parameters
            filename = f"streetview_{pano_id}_{date}_{lat}_{lon}_heading{heading}.jpg"
            original_path = os.path.join(ORIGINAL_DIR, filename)
            predicted_path = os.path.join(PREDICTED_DIR, filename)
            
            # Save original image with retries
            for _ in range(3):
                try:
                    image.save(original_path)
                    if ensure_file_exists(original_path):
                        break
                except Exception as e:
                    logger.warning(f"Retry saving original image due to: {str(e)}")
                    time.sleep(0.5)
            
            # Save predicted image with retries
            try:
                pred_image = utils_draw_predictions(
                    image.copy(), pred_boxes, pred_labels, pred_scores, 0.0
                )
                for _ in range(3):
                    try:
                        pred_image.save(predicted_path)
                        if ensure_file_exists(predicted_path):
                            break
                    except Exception as e:
                        logger.warning(f"Retry saving predicted image due to: {str(e)}")
                        time.sleep(0.5)
            except Exception as e:
                logger.error(f"Could not save prediction image: {str(e)}")
                predicted_path = original_path  # Fallback to original image
            
            # Verify files exist before adding to detections
            if ensure_file_exists(predicted_path):
                detections.append({
                    "filename": filename,
                    "lat": lat,
                    "lon": lon,
                    "heading": heading,
                    "date": date,
                    "class": LABEL_MAP.get(int(pred_labels[0]), str(pred_labels[0])),
                    "confidence": round(float(pred_scores[0]), 3),
                    "image_path": predicted_path
                })
        
        return detections
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return []

# Run detection process
def run_detection():
    try:
        log_async_state()
        logger.info("=== Detection Start ===")
        logger.info("Component states: %s", [k for k in st.session_state.keys() if len(k) == 64])  # Hash-like keys
        
        # Check if API key is provided
        if not api_key:
            st.error("Please enter your Google Street View API Key")
            return None
        
        # Check if at least one category is selected
        if not any(st.session_state.selected_categories.values()):
            st.error("Please select at least one detection category")
            return None
        
        # Add API key validation check
        test_params = [{
            'size': '640x640',
            'location': '34.0522, -118.2437',  # Test with known LA coordinates
            'heading': '0',
            'pitch': '0',
            'key': api_key
        }]
        test_result = google_streetview.api.results(test_params)
        if test_result.metadata[0].get('status') != 'OK':
            st.error("❌ Invalid API Key")
            return None
        st.success("✅ API Key Valid")
        
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
                
                params = [{
                    'size': '640x640',
                    'location': f'{lat},{lon}',
                    'heading': str(heading),
                    'pitch': '0',
                    'key': api_key
                }]
                
                try:
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
                        try:
                            detections = process_image(temp_image_path, model, device, lat, lon, heading, pano_id, date)
                            all_detections.extend(detections)
                            
                            # Update CSV if detections found
                            if detections:
                                detection_df = pd.DataFrame(detections)
                                detection_df[csv_fields].to_csv(CSV_PATH, mode='a', header=False, index=False)
                                
                                status_text.text(f"✅ Detected {len(detections)} object(s) at {lat:.6f}, {lon:.6f}, heading {heading}°")
                            else:
                                status_text.text(f"❌ No detections at {lat:.6f}, {lon:.6f}, heading {heading}°")
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            if DEBUG:
                                st.write(f"Detailed error: {str(e)}")
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_image_path):
                                try:
                                    os.remove(temp_image_path)
                                except Exception as e:
                                    if DEBUG:
                                        st.write(f"Error removing temp file: {str(e)}")
                    else:
                        status_text.text(f"No image available for {lat:.6f}, {lon:.6f}, heading {heading}°")
                    
                    # Update progress
                    processed_images += 1
                    progress_bar.progress(processed_images / total_images)
                    
                    # Add a small delay to avoid API rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"Error processing location {lat}, {lon}: {str(e)}")
                    processed_images += 1
                    progress_bar.progress(processed_images / total_images)
        
        # Clean up temp directory
        try:
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
        except Exception as e:
            if DEBUG:
                st.write(f"Error cleaning up temp directory: {str(e)}")
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Processing completed.")
        
        # Log detection results
        logger.info("Detection completed with %d total detections", len(all_detections))
        logger.info("Session state before returning: %s", list(st.session_state.keys()))
        
        # After detection completes
        logger.info("=== Detection Complete ===")
        logger.info("Results added to session state: %s", 'detection_results' in st.session_state)
        logger.info("Component states after detection: %s", [k for k in st.session_state.keys() if len(k) == 64])
        return all_detections
    
    except Exception as e:
        st.error(f"An error occurred during detection: {str(e)}")
        if DEBUG:
            st.write(f"Detailed error: {str(e)}")
        return None

# Add resource tracking helper function
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

# Function to display results on a map - leveraging utility function
def display_results_map(detections):
    if not detections:
        st.warning("No detections found in the selected area.")
        return None, None, None
    
    # Verify all image paths exist with retries
    valid_detections = []
    for d in detections:
        if 'image_path' in d and ensure_file_exists(d['image_path']):
            valid_detections.append(d)
        else:
            logger.warning(f"Missing or invalid image: {d.get('image_path', 'unknown')}")
    
    if not valid_detections:
        st.error("No valid images found for map display.")
        return None, None, None
    
    # Create map with center on the selected area
    center_lat = (top_left_lat + bottom_right_lat) / 2
    center_lon = (top_left_lon + bottom_right_lon) / 2
    
    try:
        logger.info("=== MAP RESOURCE TRACKING ===")
        # Track folium temp directory
        folium_temp = None
        folium_frame = inspect.currentframe()
        while folium_frame:
            if 'self' in folium_frame.f_locals and hasattr(folium_frame.f_locals['self'], '_parent'):
                folium_obj = folium_frame.f_locals['self']
                logger.info(f"Found folium object: {type(folium_obj).__name__}")
                break
            folium_frame = folium_frame.f_back
        
        # Generate a session-specific id for resources
        map_resource_id = str(uuid.uuid4())[:8]
        logger.info(f"Map resource tracking ID: {map_resource_id}")
        
        # Create map with resource tracking
        results_map = create_detection_map(valid_detections, center_lat, center_lon)
        
        # Log HTML output directory
        temp_dir = tempfile.gettempdir()
        logger.info(f"System temp directory: {temp_dir}")
        for file in Path(temp_dir).glob("*.html"):
            if file.stat().st_mtime > (time.time() - 300):  # Modified in last 5 minutes
                track_resource(str(file), "html")
        
        # Log detection image paths
        for i, d in enumerate(valid_detections[:3]):  # Log first 3 detections
            if 'image_path' in d:
                track_resource(d['image_path'], f"detection_{i}")
        
        logger.info("Created map with %d valid detections", len(valid_detections))
        logger.info("=== END MAP RESOURCE TRACKING ===")
        return results_map, valid_detections, map_resource_id
    except Exception as e:
        logger.error(f"Error creating map: {str(e)}")
        return None, None, None

# Function to display summary statistics - leveraging utility function
def display_summary_stats(detections):
    if not detections:
        st.warning("No detections to display in statistics.")
        return None, None, {}
    
    # Verify all detections have valid images
    valid_detections = []
    for d in detections:
        if 'image_path' in d and os.path.exists(d['image_path']):
            valid_detections.append(d)
    
    if not valid_detections:
        st.warning("No valid images found for statistics.")
        return None, None, {}
    
    try:
        fig, fig2, class_counts = create_summary_charts(valid_detections)
        return fig, fig2, class_counts
    except Exception as e:
        st.error(f"Error creating summary charts: {str(e)}")
        # Return empty placeholders
        return None, None, {}

# Create a persistent results placeholder in session state
if 'results_displayed' not in st.session_state:
    st.session_state.results_displayed = False

# Cache map generation
@st.cache_data(ttl=3600, show_spinner=False)
def generate_cached_map_html(detections, center_lat, center_lon):
    """Generate and cache the map HTML to prevent it from disappearing"""
    try:
        if not detections:
            return None
        
        # Create the map using existing function
        map_obj = create_detection_map(detections, center_lat, center_lon)
        
        # Save the map to HTML string
        map_data = BytesIO()
        map_obj.save(map_data, close_file=False)
        map_data.seek(0)
        html_string = map_data.read().decode()
        
        # Also save individual detection images to base64 for display
        detection_images = []
        for i, d in enumerate(detections):
            if 'image_path' in d and os.path.exists(d['image_path']):
                try:
                    with open(d['image_path'], "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        detection_images.append({
                            "data": img_data,
                            "lat": d['lat'],
                            "lon": d['lon'],
                            "class": d['class'],
                            "confidence": d['confidence']
                        })
                except Exception as e:
                    logger.error(f"Error encoding image {d['image_path']}: {e}")
        
        return {
            "html": html_string,
            "detection_count": len(detections),
            "detection_images": detection_images
        }
    except Exception as e:
        logger.error(f"Error generating cached map: {e}")
        return None

# Function to display cached map
def display_cached_map(cached_map):
    """Display the cached map and detection images"""
    if not cached_map:
        st.warning("No map data available")
        return
    
    # Display the map using HTML
    st.components.v1.html(cached_map["html"], height=600, scrolling=False)
    
    # Also display detection images as a fallback
    if cached_map["detection_images"]:
        st.subheader(f"Detection Images ({len(cached_map['detection_images'])} found)")
        
        # Create a 3-column grid for images
        cols = st.columns(3)
        
        # Display each detection image
        for i, img in enumerate(cached_map["detection_images"]):
            col_idx = i % 3
            with cols[col_idx]:
                st.image(
                    f"data:image/jpeg;base64,{img['data']}", 
                    caption=f"{img['class']} ({img['confidence']:.2f}) at {img['lat']:.6f}, {img['lon']:.6f}",
                    use_container_width=True
                )

# Modify the run button section to use our cached approach
if st.button("Run Detection", type="primary", disabled=not can_run_detection or not any(st.session_state.selected_categories.values())):
    with st.spinner("Running detection process..."):
        # Get detection results
        detection_results = run_detection()
        
        # Store in session state
        st.session_state.detection_results = detection_results
        
        if detection_results:
            # Generate and cache map immediately
            center_lat = (top_left_lat + bottom_right_lat) / 2
            center_lon = (top_left_lon + bottom_right_lon) / 2
            
            cached_map = generate_cached_map_html(detection_results, center_lat, center_lon)
            
            # Store cached map in session state
            st.session_state.cached_map = cached_map
            
            # Mark that results should be displayed
            st.session_state.results_displayed = True
            
            # Force a rerun with the cached data ready
            st.rerun()

# After the button, check if we should display results
if st.session_state.get('results_displayed', False) and 'cached_map' in st.session_state:
    st.success(f"Detection completed. Found {len(st.session_state.detection_results)} objects.")
    
    # Create tabs for results
    tab1, tab2, tab3 = st.tabs(["Map", "Statistics", "Raw Data"])
    
    with tab1:
        st.subheader("Detection Map")
        # Use the cached map instead of regenerating
        display_cached_map(st.session_state.cached_map)
    
    with tab2:
        st.subheader("Summary Statistics")
        if st.session_state.detection_results:
            fig, fig2, class_counts = display_summary_stats(st.session_state.detection_results)
            
            if fig and fig2:
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Detection Counts")
                    if class_counts:
                        for cls, count in class_counts.items():
                            st.metric(cls, count)
                    else:
                        st.warning("No class counts available")
            
            st.subheader("Confidence Score Distribution")
            st.pyplot(fig2)
        else:
            st.warning("No detections found for statistics display")
    
    with tab3:
        st.subheader("Raw Detection Data")
        st.dataframe(pd.DataFrame(st.session_state.detection_results))
        
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "rb") as f:
                csv_data = f.read()
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="homeless_detections.csv",
                mime="text/csv"
            )
elif not can_run_detection:
    st.error("Cannot run detection: No valid model available. Please add model file(s) to the 'models' directory.")

# Footer
st.markdown("---")
st.markdown("Homeless Detection System - Powered by Streamlit and PyTorch")

# After results display
logger.info("=== Final State Check ===")
logger.info("Session still initialized: %s", st.session_state.get('initialized', False))
logger.info("Results still present: %s", 'detection_results' in st.session_state)
logger.info("Final state keys: %s", list(st.session_state.keys()))
logger.info("=== End Final Check ===")

# At the end of the file
log_async_state()
logger.info("=== Final Component State ===")
logger.info("Final component states: %s", [k for k in st.session_state.keys() if len(k) == 64])

# Add resource verification in final checks
logger.info("=== Final Resource Verification ===")
if 'detection_map' in st.session_state and st.session_state['detection_map'].get('resource_id'):
    resource_id = st.session_state['detection_map']['resource_id']
    logger.info(f"Final map resource check, ID: {resource_id}")
    
    # Check temp directory for html files
    temp_dir = tempfile.gettempdir()
    html_count = 0
    for file in Path(temp_dir).glob("*.html"):
        if file.stat().st_mtime > (time.time() - 300):
            track_resource(str(file), "final_html")
            html_count += 1
    
    logger.info(f"Found {html_count} recent HTML files in temp directory")
    
    # Check if detection results still exist
    if 'detection_results' in st.session_state and st.session_state.detection_results is not None:
        valid_count = 0
        for d in st.session_state.detection_results:
            if 'image_path' in d and os.path.exists(d['image_path']):
                valid_count += 1
        
        logger.info(f"Final detection file check: {valid_count}/{len(st.session_state.detection_results)} still valid")
    elif 'detection_results' in st.session_state:
        logger.info("Detection results exist in session state but are None")
    else:
        logger.info("No detection results in session state")
logger.info("=== End Final Resource Verification ===")

# At the end of the file
log_async_state()
logger.info("=== Final Component State ===")
logger.info("Final component states: %s", [k for k in st.session_state.keys() if len(k) == 64]) 