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
from folium.plugins import Draw, Geocoder, Fullscreen, MousePosition
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
import traceback
from pathlib import Path
from io import BytesIO
from functools import partial
import torchvision.transforms as transforms
from detection_system.model_adapter import get_model_adapter, draw_predictions
from detection_system.utils import process_detections_for_display, prepare_grid_images

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

# Import utility functions from the detection_system package
try:
    from detection_system.model_adapter import get_model_adapter
    from detection_system.utils import draw_predictions as utils_draw_predictions
    from detection_system.utils import create_detection_map, create_summary_charts
    if DEBUG:
        st.write("Successfully imported detection_system utilities")
except Exception as e:
    st.error(f"Error importing detection_system utilities: {str(e)}")
    if DEBUG:
        st.write(f"Detailed error: {str(e)}")

# Set page config
st.set_page_config(
    page_title="VisionAid",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to fix map control display issues
st.markdown("""
<style>
/* Ensure map controls display correctly with proper z-index */
.leaflet-control {
    z-index: 1000 !important;
}

.leaflet-control-zoom {
    z-index: 1000 !important;
}

.leaflet-control-geocoder {
    z-index: 1001 !important; /* Place search above other controls */
}

.leaflet-draw {
    z-index: 1000 !important;
}

/* Fix for mouse wheel zoom */
.folium-map {
    width: 100%;
    height: 100%;
}

/* Make maps responsive but with minimum height */
.stfolium {
    min-height: 500px;
}

/* Ensure controls are visible and not hidden */
.leaflet-control-container .leaflet-top, 
.leaflet-control-container .leaflet-bottom {
    pointer-events: auto;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# Session state debugging
if st.session_state:
    logger.info("Current session state keys: %s", list(st.session_state.keys()))

# Constants and configurations
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
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
    1: "People",
    2: "Encampments",
    3: "Cart",
    4: "Bike"
}

# Category display names (more user-friendly)
CATEGORY_DISPLAY = {
    1: "People",
    2: "Encampments",
    3: "Carts",
    4: "Bikes"
}

# Add color mapping for categories
CATEGORY_COLORS = {
    1: (255, 0, 0),    # Red for People
    2: (0, 255, 0),    # Green for Encampments
    3: (0, 0, 255),    # Blue for Carts
    4: (255, 165, 0)   # Orange for Bikes
}

# Add short names for more compact labels
SHORT_LABELS = {
    1: "Person",
    2: "Camp",
    3: "Cart",
    4: "Bike" 
}

# Custom draw predictions function with color support
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

# Sidebar for input parameters
st.sidebar.title("VisionAid")
st.sidebar.info("This application detects encampment-related objects in Google Street View images")

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
    logger.info(f"Found model files: {model_files}")
    
    if not model_files:
        st.error(f"No model files (.pth) found in '{models_dir}' directory. Please add a model file.")
        model_path = None
        can_run_detection = False
    else:
        # Always show model selector in sidebar
        st.sidebar.subheader("Model Selection")
        
        # Create model adapter to get available models
        try:
            logger.info("About to initialize ModelAdapter")
            model_adapter = get_model_adapter()
            logger.info(f"ModelAdapter type: {type(model_adapter)}")
            logger.info(f"ModelAdapter attributes: {dir(model_adapter)}")
            
            # Use direct model file selection instead of configuration lookup
            model_options = {file: file for file in model_files}
            
            # Let user select model
            selected_model = st.sidebar.selectbox(
                "Select model to use:", 
                list(model_options.keys())
            )
            model_path = os.path.join(models_dir, selected_model)
            can_run_detection = True
            
            # Show model info based on file name
            if "FAST_R_CNN" in selected_model:
                st.sidebar.info("FAST_R_CNN Custom Model for Encampment Detection")
            elif "4classes" in selected_model:
                st.sidebar.info("Model trained for detecting 4 classes: People, Encampments, Carts, and Bikes")
            elif "2classes" in selected_model:
                st.sidebar.info("Model trained for detecting 2 classes: People and Encampments")
        except Exception as e:
            logger.error(f"Error initializing model adapter: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to simple model selection
            selected_model = st.sidebar.selectbox(
                "Select model to use:", 
                model_files
            )
            model_path = os.path.join(models_dir, selected_model)
            can_run_detection = True

# Main content area
st.title("VisionAid - Encampment Detection")
st.markdown("""
This application allows you to run the detection model on Google Street View images to detect encampment-related objects.
Provide your API key, select an area, and adjust settings before running the detection process.
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
    
    # Create base map with proper zoom controls but simplified options
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=15,
        zoom_control=True,
        scrollWheelZoom=True,
        dragging=True,
        preferCanvas=True  # Use canvas renderer for better performance
    )
    
    # Add fullscreen control separately with clear positioning
    folium.plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit',
        force_separate_button=True
    ).add_to(m)
    
    # Create a feature group for the grid and bounding box
    fg = folium.FeatureGroup(name="Selection Area")
    
    # Draw rectangle for the area with higher z-index
    rect = folium.Rectangle(
        bounds=[(top_left_lat, top_left_lon), (bottom_right_lat, bottom_right_lon)],
        color='blue',
        weight=3,
        fill=True,
        fill_opacity=0.2,
        tooltip="Selected Area",
        popup="Current Selection"
    )
    rect.add_to(fg)
    
    # Draw grid lines with lower opacity to reduce visual clutter
    lat_step = (bottom_right_lat - top_left_lat) / (num_rows - 1) if num_rows > 1 else 0
    lon_step = (bottom_right_lon - top_left_lon) / (num_cols - 1) if num_cols > 1 else 0
    
    for i in range(num_rows):
        lat = top_left_lat + i * lat_step
        folium.PolyLine(
            locations=[(lat, top_left_lon), (lat, bottom_right_lon)],
            color='gray',
            weight=1,
            opacity=0.4
        ).add_to(fg)
    
    for j in range(num_cols):
        lon = top_left_lon + j * lon_step
        folium.PolyLine(
            locations=[(top_left_lat, lon), (bottom_right_lat, lon)],
            color='gray',
            weight=1,
            opacity=0.4
        ).add_to(fg)
    
    # Add the feature group to the map
    fg.add_to(m)
    
    # Simplified drawing options - only rectangle with clear settings
    draw_options = {
        'polyline': False,
        'polygon': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'rectangle': {
            'shapeOptions': {
                'color': '#ff7800',
                'weight': 3,
                'opacity': 0.7,
                'fillOpacity': 0.2
            }
        }
    }
    
    # Add drawing capabilities to the map with fixed position
    draw = folium.plugins.Draw(
        position='topleft',
        draw_options=draw_options,
        edit_options={
            'featureGroup': None,  # Changed from fg to None
            'edit': False,  # Disable edit to prevent click conflicts
            'remove': False  # Disable remove to prevent click conflicts
        }
    )
    draw.add_to(m)
    
    # Add Geocoder search control with proper API integration
    if api_key:
        folium.plugins.Geocoder(
            api_key=api_key,
            position='topright'
        ).add_to(m)
    
    # Add mouse position display for better coordinate visibility
    folium.plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        num_digits=6,
        prefix="Coordinates: "
    ).add_to(m)
    
    # Add custom JavaScript to fix interactions and enhance stability
    fix_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for map to be fully loaded
        setTimeout(function() {
            var map = document.querySelector('.folium-map');
            if (!map) return;
            
            // Ensure map has focus handling
            map.addEventListener('mouseenter', function() {
                this.focus();
            });
            
            // Fix for draw tool click handling
            var drawControl = document.querySelector('.leaflet-draw');
            if (drawControl) {
                // Prevent draw tool from being obscured
                drawControl.style.zIndex = "1001";
                
                // Improve click handling for draw buttons
                var drawButtons = drawControl.querySelectorAll('.leaflet-draw-draw-rectangle');
                drawButtons.forEach(function(btn) {
                    btn.style.pointerEvents = 'all';
                    btn.style.cursor = 'pointer';
                    
                    // Prevent double-click issues
                    btn.addEventListener('click', function(e) {
                        // Prevent event bubbling
                        e.stopPropagation();
                    });
                });
            }
            
            // Fix for drawing mode conflicts
            var leafletMap = window.L && window.L.DomUtil && 
                document.querySelector('.leaflet-map-pane') ? 
                window.L.DomEvent.getMousePosition : null;
                
            if (leafletMap && window.L.Draw && window.L.Draw.Rectangle) {
                // Improve rectangle drawing behavior
                var originalRectangle = window.L.Draw.Rectangle.prototype._onMouseMove;
                window.L.Draw.Rectangle.prototype._onMouseMove = function(e) {
                    // Prevent multiple rectangle captures
                    if (!this._startLatLng) return;
                    
                    // Call original handler
                    originalRectangle.call(this, e);
                };
                
                // Fix mouseup handling for rectangle drawing
                var originalMouseUp = window.L.Draw.Rectangle.prototype._onMouseUp;
                window.L.Draw.Rectangle.prototype._onMouseUp = function(e) {
                    // Ensure proper completion of drawing
                    if (this._shape && this._startLatLng) {
                        originalMouseUp.call(this, e);
                        
                        // Reset state to prevent ghost drawings
                        this._startLatLng = null;
                        this._shape = null;
                    }
                };
            }
        }, 800);
    });
    </script>
    """
    m.get_root().html.add_child(folium.Element(fix_js))
    
    return m

# Handle map interaction for area selection
def handle_map_interaction(col1, col2):
    with col1:
        # Clear instructions about map functionality
        st.markdown("""
        ### Map Selection Tool
        
        Use this map to select the area where you want to run detection:
        
        1. **Search** for a location using the search icon (🔍) in the top-right
        2. **Zoom** with the scroll wheel or +/- controls (top-left)
        3. **Draw box**: Click the rectangle tool (☐) and draw on the map
        4. **View coordinates** at the bottom left as you move the mouse
        5. **Switch to fullscreen** for easier selection (top-right icon)
        """)
        
        # Create the map
        area_map = display_area_map()
        
        # Configure st_folium with stable settings
        map_data = st_folium(
            area_map, 
            height=500,
            width=700,
            returned_objects=["all_drawings", "last_active_drawing"],
            use_container_width=True
        )
        
        # Clear handling of map response data
        if map_data is not None and "all_drawings" in map_data and map_data["all_drawings"]:
            try:
                # Get the most recent drawing
                last_drawing = map_data["all_drawings"][-1]
                
                # Only process if we have valid geometry
                if "geometry" in last_drawing and "coordinates" in last_drawing["geometry"]:
                    # Extract coordinates safely
                    coords = last_drawing["geometry"]["coordinates"]
                    
                    # Only process rectangular coordinates (4 points in first array)
                    if coords and len(coords) > 0 and len(coords[0]) >= 4:
                        # Get points
                        points = coords[0]
                        
                        # Extract lat/lon values correctly
                        lats = [point[1] for point in points if len(point) >= 2]
                        lons = [point[0] for point in points if len(point) >= 2]
                        
                        if lats and lons:
                            # Calculate bounding box - avoid extreme values
                            max_lat, min_lat = max(lats), min(lats)
                            min_lon, max_lon = min(lons), max(lons)
                            
                            # Update bounding box with valid values
                            bbox_data = {
                                "topLeftLat": max_lat,
                                "topLeftLon": min_lon,
                                "bottomRightLat": min_lat,
                                "bottomRightLon": max_lon
                            }
                            
                            # Update state with valid values only
                            if all(isinstance(v, (int, float)) for v in bbox_data.values()):
                                handle_bbox_selection(json.dumps(bbox_data))
                                
                                # Log successful update for debugging
                                logger.info(f"Updated bounding box: {bbox_data}")
            except Exception as e:
                # Safely handle any errors during processing
                logger.error(f"Error processing map drawing: {e}")
                
    # Display grid information in the second column
    with col2:
        st.subheader("Grid Information")
        st.info(f"""
        **Selected Area:**
        - Grid Size: {num_rows} x {num_cols} = {num_rows * num_cols} points
        - Latitude Range: {top_left_lat:.6f} to {bottom_right_lat:.6f}
        - Longitude Range: {top_left_lon:.6f} to {bottom_right_lon:.6f}
        
        **Sampling:**
        - Points will be sampled at headings: 0° and 180°
        - Total API calls: {num_rows * num_cols * 2}
        """)
        
        # Add a tips section with clearer instructions
        st.markdown("### Tips")
        st.markdown("""
        - For best results, draw a rectangle by clicking and dragging in one continuous motion
        - You can adjust the coordinates manually using the number inputs in the sidebar
        - Keep the grid size reasonable (5x5 recommended) to avoid excessive API usage
        - Make sure your Google API key is correctly entered in the sidebar
        """)

# Main layout
col1, col2 = st.columns([2, 1])

# Handle map interaction through a dedicated function
handle_map_interaction(col1, col2)

# Function to load model with error handling
def load_model():
    try:
        logger.info("================== MODEL LOADING START ==================")
        logger.info("Attempting to load model")
        
        # Initialize model adapter
        logger.info("Creating ModelAdapter instance")
        model_adapter = get_model_adapter()
        logger.info(f"ModelAdapter created: {type(model_adapter)}")
        
        # First, look specifically for the preferred model
        models_dir = os.path.join(os.path.dirname(__file__), "models")
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
            
            # Track file size and permissions
            if os.path.exists(model_path):
                file_stats = os.stat(model_path)
                logger.info(f"Model file size: {file_stats.st_size / (1024*1024):.2f} MB")
                logger.info(f"Model file permissions: {oct(file_stats.st_mode)}")
            else:
                logger.error(f"Selected model file does not exist: {model_path}")
                return None, None
        else:
            logger.error("No .pth model files found")
            return None, None
        
        # Load the model with detailed error logging
        with st.spinner(f"Loading model, please wait..."):
            logger.info("Calling model_adapter.load_model()")
            try:
                # First try direct loading without using the adapter
                from detection_system.model_adapter import create_custom_fasterrcnn_with_bn
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
                logger.info(f"State dict keys: {list(state_dict.keys())[:5]}...")
                
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
                
                # Return the adapter and device
                logger.info("Model loaded successfully")
                logger.info("================== MODEL LOADING END ==================")
                return adapter, device
                
            except Exception as direct_error:
                logger.error(f"Direct model loading failed: {str(direct_error)}")
                logger.error(traceback.format_exc())
                
                # Fall back to adapter method
                try:
                    logger.info("Falling back to adapter.load_model method")
                    model, device, config = model_adapter.load_model(model_path)
                    logger.info(f"Model loaded using adapter: {type(model)}")
                    logger.info(f"Device: {device}")
                    
                    # Set model to evaluation mode
                    model.eval()
                    logger.info("Model set to evaluation mode")
                    
                    # Return the adapter and device
                    logger.info("Model loaded successfully via adapter")
                    logger.info("================== MODEL LOADING END ==================")
                    return model_adapter, device
                except Exception as adapter_error:
                    logger.error(f"Adapter loading also failed: {str(adapter_error)}")
                    logger.error(traceback.format_exc())
                    logger.info("================== MODEL LOADING FAILED ==================")
                    return None, None
    
    except Exception as e:
        logger.error(f"Error in load_model function: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("================== MODEL LOADING FAILED ==================")
        return None, None

# Function to process a single image
def process_image(image_path, model_adapter, device, lat, lon, heading, pano_id, date):
    try:
        logger.info("================= IMAGE PROCESSING START =================")
        logger.info(f"Processing image at {lat:.6f}, {lon:.6f}, heading {heading}°")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Model adapter type: {type(model_adapter)}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return []
        
        # Check file properties
        file_stats = os.stat(image_path)
        file_size_kb = file_stats.st_size / 1024
        logger.info(f"Image file size: {file_size_kb:.2f} KB")
        
        if file_size_kb < 1:  # Less than 1KB is probably empty
            logger.error("Image file appears to be empty or corrupted")
            return []
        
        # Load and prepare image
        try:
            logger.info("Loading image with PIL")
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Original image size: {image.size}")
            logger.info(f"Image mode: {image.mode}")
            
            # Save a debug copy of the input image
            debug_dir = os.path.join(os.path.dirname(__file__), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"input_{pano_id}_{heading}.jpg")
            image.save(debug_path)
            logger.info(f"Saved debug input image to {debug_path}")
            
            # Resize for model
            target_size = (512, 512)
            logger.info(f"Resizing to {target_size}")
            original_size = image.size
            image_resized = transforms.Resize(target_size)(image)
            logger.info(f"Resized image size: {image_resized.size}")
        except Exception as img_error:
            logger.error(f"Error loading or resizing image: {str(img_error)}")
            logger.error(traceback.format_exc())
            return []
        
        # Run model prediction
        try:
            logger.info("Running model prediction")
            base_confidence = 0.1  # Lower threshold to catch any potential detections
            logger.info(f"Using base confidence threshold: {base_confidence}")
            
            # Ensure model_adapter has predict method
            if not hasattr(model_adapter, 'predict'):
                logger.error(f"Model adapter has no 'predict' method. Available methods: {dir(model_adapter)}")
                return []
            
            # Run prediction
            logger.info("Calling model_adapter.predict()")
            pred_boxes, pred_labels, pred_scores = model_adapter.predict(image_resized, confidence_threshold=base_confidence)
            
            # Log raw prediction results
            logger.info(f"Raw prediction boxes shape: {pred_boxes.shape if isinstance(pred_boxes, np.ndarray) else 'empty'}")
            logger.info(f"Raw prediction labels: {pred_labels}")
            logger.info(f"Raw prediction scores: {pred_scores}")
        except Exception as pred_error:
            logger.error(f"Error during model prediction: {str(pred_error)}")
            logger.error(traceback.format_exc())
            return []
        
        # Get class map
        try:
            logger.info("Getting class map")
            if hasattr(model_adapter, 'get_class_map'):
                class_map = model_adapter.get_class_map()
                logger.info(f"Class map: {class_map}")
            else:
                logger.warning("Model adapter has no get_class_map method, using default")
                class_map = {
                    1: "People",
                    2: "Encampments",
                    3: "Cart",
                    4: "Bike"
                }
        except Exception as class_error:
            logger.error(f"Error getting class map: {str(class_error)}")
            class_map = {i: f"Class_{i}" for i in range(1, 5)}
        
        # Filter predictions by category and threshold
        logger.info("Filtering predictions by category and threshold")
        logger.info(f"Selected categories: {st.session_state.selected_categories}")
        logger.info(f"Category thresholds: {st.session_state.category_thresholds}")
        
        filtered_indices = []
        for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
            category_id = int(label)
            logger.info(f"Checking prediction {i}: category={category_id}, score={score:.3f}")
            # Check if category is selected and meets its threshold
            if (category_id in st.session_state.selected_categories and 
                st.session_state.selected_categories[category_id] and
                score >= st.session_state.category_thresholds[category_id]):
                logger.info(f"  Accepted: category {category_id} with score {score:.3f}")
                filtered_indices.append(i)
            else:
                logger.info(f"  Rejected: category {category_id} with score {score:.3f}")
        
        # Apply filters
        if filtered_indices:
            logger.info(f"Keeping {len(filtered_indices)} detections after filtering")
            pred_boxes = pred_boxes[filtered_indices]
            pred_labels = pred_labels[filtered_indices]
            pred_scores = pred_scores[filtered_indices]
        else:
            logger.info("No detections passed filtering criteria")
            pred_boxes = np.array([])
            pred_labels = np.array([])
            pred_scores = np.array([])
        
        # Process detections
        detections = []
        if len(pred_boxes) > 0:
            logger.info(f"Processing {len(pred_boxes)} valid detections")
            
            # Create paths for output images
            filename = f"streetview_{pano_id}_{date}_{lat}_{lon}_heading{heading}.jpg"
            original_path = os.path.join(ORIGINAL_DIR, filename)
            predicted_path = os.path.join(PREDICTED_DIR, filename)
            logger.info(f"Original path: {original_path}")
            logger.info(f"Predicted path: {predicted_path}")
            
            # Save original image
            try:
                logger.info("Saving original image")
                original_image = Image.open(image_path).convert("RGB")
                original_image.save(original_path)
                logger.info(f"Original image saved to: {original_path}")
                logger.info(f"File exists: {os.path.exists(original_path)}")
            except Exception as save_error:
                logger.error(f"Error saving original image: {str(save_error)}")
            
            # Draw and save prediction image
            try:
                logger.info("Drawing prediction boxes")
                # Convert boxes back to original image size if necessary
                if original_size != target_size:
                    logger.info("Scaling boxes to original image size")
                    scale_x = original_size[0] / target_size[0]
                    scale_y = original_size[1] / target_size[1]
                    logger.info(f"Scale factors: x={scale_x}, y={scale_y}")
                    
                    scaled_boxes = []
                    for box in pred_boxes:
                        x1, y1, x2, y2 = box
                        scaled_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        scaled_boxes.append(scaled_box)
                    
                    pred_boxes = np.array(scaled_boxes)
                
                # Draw predictions
                logger.info("Drawing predictions on image")
                pred_image = draw_predictions_with_colors(
                    original_image.copy(), pred_boxes, pred_labels, pred_scores, 0.0
                )
                
                # Save prediction image
                logger.info(f"Saving prediction image to: {predicted_path}")
                pred_image.save(predicted_path)
                logger.info(f"Predicted image saved: {os.path.exists(predicted_path)}")
                
                # Also save to debug directory
                debug_pred_path = os.path.join(debug_dir, f"pred_{pano_id}_{heading}.jpg")
                pred_image.save(debug_pred_path)
                logger.info(f"Debug prediction image saved to: {debug_pred_path}")
            except Exception as draw_error:
                logger.error(f"Error drawing or saving predicted image: {str(draw_error)}")
                logger.error(traceback.format_exc())
                predicted_path = original_path  # Fallback to original image
            
            # Create detection records
            if os.path.exists(predicted_path):
                logger.info("Creating detection records")
                for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                    cls_id = int(label)
                    cls_name = class_map.get(cls_id, str(cls_id))
                    logger.info(f"Detection {i}: class={cls_name}, score={score:.3f}")
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
            else:
                logger.error(f"Failed to create predicted image: {predicted_path}")
        
        logger.info(f"Returning {len(detections)} detections")
        logger.info("================= IMAGE PROCESSING END =================")
        return detections
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("================= IMAGE PROCESSING FAILED =================")
        return []

# Run detection process
def run_detection():
    try:
        logger.info("================== DETECTION START ==================")
        log_async_state()
        logger.info("Component states: %s", [k for k in st.session_state.keys() if len(k) == 64])  # Hash-like keys
        
        # Check if API key is provided
        if not api_key:
            logger.error("Missing API key")
            st.error("Please enter your Google Street View API Key")
            return None
        
        # Check if at least one category is selected
        logger.info(f"Selected categories: {st.session_state.selected_categories}")
        if not any(st.session_state.selected_categories.values()):
            logger.error("No categories selected")
            st.error("Please select at least one detection category")
            return None
        
        # Add API key validation check
        try:
            logger.info("Validating API key with test request")
            test_params = [{
                'size': '640x640',
                'location': '34.0522, -118.2437',  # Test with known LA coordinates
                'heading': '0',
                'pitch': '0',
                'key': api_key
            }]
            test_result = google_streetview.api.results(test_params)
            
            logger.info(f"API test status: {test_result.metadata[0].get('status')}")
            if test_result.metadata[0].get('status') != 'OK':
                logger.error(f"Invalid API key: {test_result.metadata[0]}")
                st.error("❌ Invalid API Key")
                return None
                
            logger.info("API key validation successful")
            st.success("✅ API Key Valid")
        except Exception as api_error:
            logger.error(f"API key validation error: {str(api_error)}")
            st.error(f"Error validating API key: {str(api_error)}")
            return None
        
        # Generate grid points
        logger.info(f"Generating grid: {num_rows}x{num_cols}")
        logger.info(f"Area bounds: ({top_left_lat}, {top_left_lon}) to ({bottom_right_lat}, {bottom_right_lon})")
        
        latitudes = [top_left_lat + i * (bottom_right_lat - top_left_lat) / (num_rows - 1) for i in range(num_rows)]
        longitudes = [top_left_lon + j * (bottom_right_lon - top_left_lon) / (num_cols - 1) for j in range(num_cols)]
        grid_points = [(lat, lon) for lat in latitudes for lon in longitudes]
        
        logger.info(f"Generated {len(grid_points)} grid points")
        
        # Load model
        logger.info("Loading model")
        with st.spinner("Loading model..."):
            model_adapter, device = load_model()
            
        # Exit if model loading failed
        if model_adapter is None or device is None:
            logger.error("Model loading failed")
            st.error("Model loading failed. Cannot continue with detection.")
            return None
            
        logger.info(f"Model loaded successfully: {type(model_adapter)}")
        
        # Ensure output directories exist
        logger.info("Checking output directories")
        for directory in [TEMP_DIR, RESULTS_DIR, ORIGINAL_DIR, PREDICTED_DIR]:
            logger.info(f"Ensuring directory exists: {directory}")
            os.makedirs(directory, exist_ok=True)
        
        # Initialize progress indicators
        logger.info("Initializing progress indicators")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()
        
        # Initialize results container
        all_detections = []
        processed_images = 0
        total_images = len(grid_points) * 2  # 2 headings per point
        logger.info(f"Total images to process: {total_images}")
        
        # Clear previous results
        if os.path.exists(CSV_PATH):
            logger.info(f"Removing previous CSV file: {CSV_PATH}")
            os.remove(CSV_PATH)
        
        # Create CSV headers
        logger.info("Creating new CSV file with headers")
        csv_fields = ["filename", "lat", "lon", "heading", "date", "class", "confidence"]
        pd.DataFrame(columns=csv_fields).to_csv(CSV_PATH, index=False)
        
        # Process each grid point
        logger.info("Beginning image processing loop")
        for idx, (lat, lon) in enumerate(grid_points):
            logger.info(f"Processing point {idx+1}/{len(grid_points)}: ({lat:.6f}, {lon:.6f})")
            
            for heading in [0, 180]:
                logger.info(f"Processing heading: {heading}°")
                status_text.text(f"Processing location {idx+1}/{len(grid_points)}, heading {heading}°...")
                
                params = [{
                    'size': '640x640',
                    'location': f'{lat},{lon}',
                    'heading': str(heading),
                    'pitch': '0',
                    'key': api_key,
                    'source': 'outdoor'  # Add this parameter to only get official Street View car images
                }]
                
                try:
                    logger.info("Fetching Street View image")
                    results = google_streetview.api.results(params)
                    
                    if results.metadata[0]['status'] == 'OK':
                        logger.info("Street View image available")
                        metadata = results.metadata[0]
                        pano_id = metadata.get('pano_id', 'unknown')
                        date = metadata.get('date', datetime.now().strftime('%Y-%m'))
                        logger.info(f"Image metadata - pano_id: {pano_id}, date: {date}")
                        
                        # Download image
                        logger.info("Downloading image")
                        response = requests.get(results.links[0])
                        temp_image_path = os.path.join(TEMP_DIR, f"temp_{pano_id}.jpg")
                        
                        with open(temp_image_path, 'wb') as f:
                            f.write(response.content)
                            
                        # Verify image download
                        if os.path.exists(temp_image_path):
                            file_size = os.path.getsize(temp_image_path)
                            logger.info(f"Image downloaded: {temp_image_path} ({file_size} bytes)")
                            
                            if file_size == 0:
                                logger.error("Downloaded image is empty")
                                status_text.text(f"❌ Empty image at {lat:.6f}, {lon:.6f}")
                                continue
                        else:
                            logger.error("Image download failed")
                            status_text.text(f"❌ Download failed for {lat:.6f}, {lon:.6f}")
                            continue
                        
                        # Process image
                        logger.info(f"Processing downloaded image: {temp_image_path}")
                        try:
                            detections = process_image(temp_image_path, model_adapter, device, lat, lon, heading, pano_id, date)
                            
                            # Log detection results
                            logger.info(f"Received {len(detections)} detections from process_image")
                            if detections:
                                for i, det in enumerate(detections):
                                    logger.info(f"Detection {i+1}: {det['class']} with confidence {det['confidence']}")
                            
                            all_detections.extend(detections)
                            
                            # Update CSV if detections found
                            if detections:
                                logger.info("Writing detections to CSV")
                                detection_df = pd.DataFrame(detections)
                                detection_df[csv_fields].to_csv(CSV_PATH, mode='a', header=False, index=False)
                                
                                status_text.text(f"✅ Detected {len(detections)} object(s) at {lat:.6f}, {lon:.6f}, heading {heading}°")
                                
                                # Display preview of the latest detection
                                if len(detections) > 0 and 'image_path' in detections[0] and os.path.exists(detections[0]['image_path']):
                                    logger.info(f"Displaying preview image: {detections[0]['image_path']}")
                                    results_placeholder.image(
                                        detections[0]['image_path'], 
                                        caption=f"Latest detection: {detections[0]['class']} ({detections[0]['confidence']:.2f})",
                                        width=-1
                                    )
                            else:
                                logger.info("No detections found at this location")
                                status_text.text(f"❌ No detections at {lat:.6f}, {lon:.6f}, heading {heading}°")
                        except Exception as proc_error:
                            logger.error(f"Error processing image: {str(proc_error)}")
                            logger.error(traceback.format_exc())
                            st.error(f"Error processing image: {str(proc_error)}")
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_image_path):
                                try:
                                    logger.info(f"Removing temporary file: {temp_image_path}")
                                    os.remove(temp_image_path)
                                except Exception as rm_error:
                                    logger.warning(f"Error removing temp file: {str(rm_error)}")
                    else:
                        logger.info(f"No Street View image available: {results.metadata[0]['status']}")
                        status_text.text(f"No image available for {lat:.6f}, {lon:.6f}, heading {heading}°")
                    
                    # Update progress
                    processed_images += 1
                    progress_percent = processed_images / total_images
                    logger.info(f"Progress: {processed_images}/{total_images} ({progress_percent:.1%})")
                    progress_bar.progress(progress_percent)
                    
                    # Add a small delay to avoid API rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error in grid point processing: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"Error processing location {lat}, {lon}: {str(e)}")
                    processed_images += 1
                    progress_bar.progress(processed_images / total_images)
        
        # Clean up temp directory but make sure to preserve the directory itself
        try:
            logger.info("Final cleanup of temp directory")
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Temp directory cleaned")
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {str(e)}")
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Processing completed.")
        
        # Log detection results
        logger.info(f"Detection completed with {len(all_detections)} total detections")
        logger.info(f"Session state keys: {list(st.session_state.keys())}")
        
        # Store detections in a debug file
        if all_detections:
            try:
                debug_file = os.path.join(os.path.dirname(__file__), "debug", "detections.json")
                with open(debug_file, 'w') as f:
                    json.dump(all_detections, f, indent=2, default=str)
                logger.info(f"Saved debug detections to {debug_file}")
            except Exception as json_error:
                logger.error(f"Error saving debug detections: {str(json_error)}")
        
        logger.info("================== DETECTION COMPLETE ==================")
        return all_detections
    
    except Exception as e:
        logger.error(f"Fatal error in run_detection: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred during detection: {str(e)}")
        logger.info("================== DETECTION FAILED ==================")
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
        
        return {
            "html": html_string,
            "detection_count": len(detections)
        }
    except Exception as e:
        logger.error(f"Error generating cached map: {e}")
        return None

# Function to display cached map
def display_cached_map(cached_map):
    """Display the cached map"""
    if not cached_map:
        st.warning("No map data available")
        return
    
    # Display the map using HTML
    st.components.v1.html(cached_map["html"], height=600, scrolling=False)

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
    tab1, tab2, tab3, tab4 = st.tabs(["Map", "Statistics", "Detection Images", "Raw Data"])
    
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
        st.subheader("Detection Images")
        if st.session_state.detection_results:
            # Group detections by image to avoid duplicates
            image_detections = {}
            for det in st.session_state.detection_results:
                if det['image_path'] not in image_detections:
                    image_detections[det['image_path']] = {
                        'detections': [],
                        'lat': det['lat'],
                        'lon': det['lon'],
                        'heading': det['heading'],
                        'boxes': [],  # Add boxes list
                        'labels': [],  # Add labels list
                        'scores': []   # Add scores list
                    }
                image_detections[det['image_path']]['detections'].append({
                    'class': det['class'],
                    'confidence': det['confidence']
                })
                # Store bounding box information if available
                if 'boxes' in det:
                    image_detections[det['image_path']]['boxes'].append(det['boxes'])
                    image_detections[det['image_path']]['labels'].append(det['class'])
                    image_detections[det['image_path']]['scores'].append(det['confidence'])
            
            # Create a 3-column grid for images
            cols = st.columns(3)
            col_idx = 0
            
            for img_path, data in image_detections.items():
                if os.path.exists(img_path):
                    with cols[col_idx]:
                        try:
                            # Load the original image
                            img = Image.open(img_path).convert("RGB")
                            
                            # Create detection summary
                            det_summary = []
                            for det in data['detections']:
                                det_summary.append(f"{det['class']} ({det['confidence']:.2f})")
                            
                            # Location info
                            location_info = f"Location: {data['lat']:.6f}, {data['lon']:.6f}\nHeading: {data['heading']}°"
                            
                            # Display main image with all detections
                            main_img = img.copy()
                            if data['boxes'] and len(data['boxes']) > 0:
                                main_img = draw_predictions_with_colors(
                                    main_img,
                                    np.array(data['boxes']),
                                    [LABEL_MAP.get(label, label) for label in data['labels']],
                                    np.array(data['scores'])
                                )
                            
                            # Display the main image with caption
                            st.image(main_img, 
                                    caption=f"All Detections: {', '.join(det_summary)}\n{location_info}",
                                    width=-1
                                )
                            
                            # Add expandable section for individual detections
                            with st.expander("View Individual Detections"):
                                # Show each detection separately with its own bounding box
                                for i, (det, box) in enumerate(zip(data['detections'], data['boxes'])):
                                    st.markdown(f"**{det['class']} ({det['confidence']:.2f})**")
                                    
                                    # Create image with single detection
                                    single_det_img = img.copy()
                                    single_det_img = draw_predictions_with_colors(
                                        single_det_img,
                                        np.array([box]),
                                        np.array([det['class']]),
                                        np.array([det['confidence']])
                                    )
                                    
                                    st.image(single_det_img, 
                                            caption=f"{det['class']} - Confidence: {det['confidence']:.2f}\n{location_info}",
                                            width=-1
                                        )
                                    
                                    # Add a separator between detections
                                    if i < len(data['detections']) - 1:
                                        st.markdown("---")
                        
                        except Exception as e:
                            st.error(f"Error displaying image {img_path}: {str(e)}")
                    
                    # Update column index
                    col_idx = (col_idx + 1) % 3
                    
                    # Create new row of columns if needed
                    if col_idx == 0:
                        cols = st.columns(3)
        else:
            st.warning("No detection images available")
    
    with tab4:
        st.subheader("Raw Detection Data")
        st.dataframe(pd.DataFrame(st.session_state.detection_results))
        
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "rb") as f:
                csv_data = f.read()
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="encampment_detections.csv",
                mime="text/csv"
            )
elif not can_run_detection:
    st.error("Cannot run detection: No valid model available. Please add model file(s) to the 'models' directory.")

# Footer
st.markdown("---")
st.markdown("VisionAid - Powered by PyTorch")

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

# Function to geocode address using Google Maps API
def geocode_address(address, api_key):
    """Convert an address to latitude and longitude using Google Maps Geocoding API"""
    if not address or not api_key:
        return None
    
    try:
        # Prepare the request URL
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": address,
            "key": api_key
        }
        
        # Make the request
        response = requests.get(base_url, params=params)
        data = response.json()
        
        # Check if the request was successful
        if data["status"] == "OK":
            location = data["results"][0]["geometry"]["location"]
            return {
                "lat": location["lat"],
                "lng": location["lng"],
                "formatted_address": data["results"][0]["formatted_address"]
            }
        else:
            logger.warning(f"Geocoding error: {data['status']}")
            return None
    except Exception as e:
        logger.error(f"Error geocoding address: {str(e)}")
        return None 