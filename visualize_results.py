import streamlit as st
import pandas as pd
import os
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import base64
from PIL import Image
import glob
import re
# Import utility functions
from homeless_detection.utils import create_detection_map, create_summary_charts

# Set page config
st.set_page_config(
    page_title="Homeless Detection Results Viewer",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load results
def load_results(results_dir):
    csv_path = os.path.join(results_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

# Function to prepare data for detection map
def prepare_map_data(df, predicted_dir):
    if df is None or df.empty:
        st.warning("No results to display")
        return None
    
    # Group detections by location and heading
    df['loc_key'] = df['lat'].astype(str) + '_' + df['lon'].astype(str) + '_' + df['heading'].astype(str)
    
    # Prepare detections in the format expected by create_detection_map
    detections = []
    
    for _, row in df.iterrows():
        filename = row['filename']
        image_path = os.path.join(predicted_dir, filename)
        
        # Check if the image exists or try to find a matching one
        if not os.path.exists(image_path):
            pattern = f"*{row['lat']}_{row['lon']}_heading{row['heading']}.jpg"
            matching_files = glob.glob(os.path.join(predicted_dir, pattern))
            if matching_files:
                image_path = matching_files[0]
                filename = os.path.basename(image_path)
            else:
                continue
        
        # Add to detections list
        detections.append({
            "filename": filename,
            "lat": row['lat'],
            "lon": row['lon'],
            "heading": row['heading'],
            "date": row.get('date', ''),
            "class": row['class'],
            "confidence": row['confidence'],
            "image_path": image_path
        })
    
    return detections if detections else None

# Function to display results on a map
def display_results_map(df, predicted_dir):
    # Prepare data for the map
    detections = prepare_map_data(df, predicted_dir)
    if not detections:
        st.warning("No valid detection locations found")
        return None
    
    # Get center coordinates from first detection
    first_loc = detections[0]
    center_lat = first_loc["lat"]
    center_lon = first_loc["lon"]
    
    # Use the utility function to create the map
    return create_detection_map(detections, center_lat, center_lon)

# Function to display image grid
def display_image_grid(df, predicted_dir, class_filter=None, min_confidence=0.5):
    if df is None or df.empty:
        return
    
    # Filter by class and confidence
    filtered_df = df.copy()
    if class_filter:
        filtered_df = filtered_df[filtered_df['class'] == class_filter]
    
    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    
    if filtered_df.empty:
        st.warning("No images match the selected filters")
        return
    
    # Get unique filenames
    unique_files = filtered_df['filename'].unique()
    
    # Display image grid
    cols = 3
    rows = (len(unique_files) + cols - 1) // cols
    
    for i in range(rows):
        columns = st.columns(cols)
        for j in range(cols):
            idx = i * cols + j
            if idx < len(unique_files):
                filename = unique_files[idx]
                image_path = os.path.join(predicted_dir, filename)
                
                if os.path.exists(image_path):
                    # Get detections for this image
                    image_df = filtered_df[filtered_df['filename'] == filename]
                    class_summary = image_df['class'].value_counts().to_dict()
                    summary_text = ", ".join([f"{cls}: {count}" for cls, count in class_summary.items()])
                    
                    # Extract coordinates from filename
                    match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+)_heading(\d+)', filename)
                    if match:
                        lat, lon, heading = match.groups()
                        location_text = f"Location: {lat}, {lon}, Heading: {heading}°"
                    else:
                        location_text = "Location: Unknown"
                    
                    # Display in column
                    with columns[j]:
                        st.image(image_path, caption=f"{summary_text}\n{location_text}")

# Sidebar
st.sidebar.title("Results Viewer")
st.sidebar.info("Visualize previously generated detection results")

# Results directory selection
results_dir = st.sidebar.text_input("Results Directory", value="results")

# Main content
st.title("Homeless Detection Results Viewer")

# Check if directory exists
if not os.path.exists(results_dir):
    st.error(f"Directory not found: {results_dir}")
    st.info("Please enter a valid directory path containing detection results")
else:
    # Check for predictions CSV
    predicted_dir = os.path.join(results_dir, "predicted")
    df = load_results(results_dir)
    
    if df is not None:
        st.success(f"Loaded {len(df)} detections from {df['filename'].nunique()} images")
        
        # Filter controls
        st.sidebar.subheader("Filters")
        
        # Class filter
        classes = ["All"] + sorted(df['class'].unique().tolist())
        selected_class = st.sidebar.selectbox("Filter by Class", classes)
        class_filter = None if selected_class == "All" else selected_class
        
        # Confidence threshold
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
        
        # Apply filters for statistics
        filtered_df = df.copy()
        if class_filter:
            filtered_df = filtered_df[filtered_df['class'] == class_filter]
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display on tabs
        tab1, tab2, tab3 = st.tabs(["Map", "Statistics", "Image Gallery"])
        
        with tab1:
            st.subheader("Detection Map")
            results_map = display_results_map(filtered_df, predicted_dir)
            if results_map:
                folium_static(results_map, width=1200, height=800)
        
        with tab2:
            st.subheader("Summary Statistics")
            if not filtered_df.empty:
                # Prepare data for charts
                detections = prepare_map_data(filtered_df, predicted_dir)
                if detections:
                    # Use shared utility to create charts
                    fig, fig2, class_counts = create_summary_charts(detections)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Detection Counts")
                        for cls, count in class_counts.items():
                            st.metric(cls, count)
                    
                    st.subheader("Confidence Score Distribution")
                    st.pyplot(fig2)
                    
                    # Additional statistics
                    st.subheader("Additional Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Detections", len(filtered_df))
                        st.metric("Unique Images", filtered_df['filename'].nunique())
                    
                    with col2:
                        st.metric("Average Confidence", f"{filtered_df['confidence'].mean():.2f}")
                        st.metric("Median Confidence", f"{filtered_df['confidence'].median():.2f}")
            else:
                st.warning("No data available for the selected filters")
        
        with tab3:
            st.subheader("Image Gallery")
            display_image_grid(filtered_df, predicted_dir, class_filter, min_confidence)
    else:
        st.error("No predictions found in the specified directory")
        st.info("Please make sure the directory contains a 'predictions.csv' file")

# Footer
st.markdown("---")
st.markdown("Homeless Detection Results Viewer - Powered by Streamlit") 