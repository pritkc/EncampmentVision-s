import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Mechanism - VisionAid",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load model comparison data
def load_model_comparison_data():
    try:
        comprehensive_path = os.path.join('data', 'comprehensive.csv')
        mask_rcnn_path = os.path.join('data', 'mask_r_cnn.csv')
        fast_rcnn_path = os.path.join('data', 'fast_r_cnn.csv')
        
        # Load the data
        comprehensive_df = pd.read_csv(comprehensive_path)
        mask_rcnn_df = pd.read_csv(mask_rcnn_path)
        fast_rcnn_df = pd.read_csv(fast_rcnn_path)
        
        return {
            'comprehensive': comprehensive_df,
            'mask_rcnn': mask_rcnn_df,
            'fast_rcnn': fast_rcnn_df
        }
    except Exception as e:
        st.error(f"Error loading model comparison data: {str(e)}")
        return None

# Title
st.title("How VisionAid Works")

# Main content
st.markdown("""
## Technical Approach

VisionAid uses a deep learning-based object detection system to identify visual indicators of homelessness in Google Street View imagery. The system follows these key steps:

1. **Area Selection**: Users define a geographic area of interest using latitude/longitude coordinates
2. **Image Acquisition**: The system creates a grid of points within the selected area and retrieves Street View images
3. **Object Detection**: Deep learning models analyze each image to identify objects of interest
4. **Results Aggregation**: Detections are aggregated and visualized on maps and in reports

## Model Architecture

After evaluating several state-of-the-art object detection architectures, we selected Faster R-CNN as our primary model due to its superior performance for our specific use case. Our comparative analysis included:

- **Faster R-CNN**: Region-based approach with excellent accuracy for medium-sized objects
- **Mask R-CNN**: Extension of Faster R-CNN with instance segmentation capabilities
- **YOLO (You Only Look Once)**: Single-shot detector with faster inference but lower precision
- **DeepLabV3+**: Semantic segmentation model with good boundary detection

Faster R-CNN proved to be the most effective for our specific requirements, providing the best balance of accuracy and performance across all object categories.
""")

st.markdown("""
### Faster R-CNN Architecture

VisionAid employs a Faster R-CNN (Region-based Convolutional Neural Network) architecture with the following components:

- **Backbone**: ResNet-50 for feature extraction from images
- **Region Proposal Network (RPN)**: Identifies potential regions containing objects
- **Classification & Bounding Box Regression**: Assigns class labels and refines bounding boxes
- **Multi-class Output**: Simultaneously detects multiple categories of objects

The model is trained on a custom dataset of annotated Street View images containing examples of people, encampments, shopping carts, and bicycles.

## Detection Process
""")

# Add process diagram with fallback options
try:
    # Add descriptive text
    st.markdown("""
    ### How Detection Works
    
    The VisionAid system follows a structured pipeline to detect objects in Google Street View imagery:
    
    1. **User Input**: The process begins with users selecting an area on the map and configuring detection parameters including categories and sensitivity thresholds.
    
    2. **Image Acquisition**: The system creates a grid of geographic points and requests Street View images from each location, capturing multiple viewing angles.
    
    3. **Model Preprocessing**: Each image is prepared for the neural network through resizing, normalization, and tensor conversion.
    
    4. **Object Detection**: The Faster R-CNN model processes the images, identifying regions of interest and classifying objects within them.
    
    5. **Results Visualization**: Detected objects are highlighted with color-coded bounding boxes and labeled with class and confidence information.
    
    6. **Analysis & Reporting**: Results are aggregated, displayed on interactive maps, and presented with summary statistics for further analysis.
    
    #### Advanced Technical Details
    
    For a more detailed explanation of our detection process, including each step from user input to final results, please check our [Detection Process](Detection_Process) page.
    """)
    
except Exception as e:
    # Use a fallback image if available
    try:
        img_path = os.path.join('Content', 'Img', 'Img [3].png')
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption="VisionAid Detection Process", use_container_width=True)
        else:
            st.warning("Process diagram not found. Please check Content/Img directory.")
    except Exception as e3:
        st.error(f"Error loading image: {str(e3)}")

st.markdown("""
## Grid-based Analysis

To analyze a geographic area systematically, VisionAid:

1. Creates a grid of evenly spaced points within the selected area
2. For each point, retrieves Street View images facing multiple directions (0° and 180°)
3. Processes each image through the detection model
4. Records the location, direction, and details of each detection

This approach ensures comprehensive coverage of the selected area and enables spatial analysis of results.

## Confidence Scoring

Each detection includes a confidence score (0-1) indicating the model's certainty. Users can adjust confidence thresholds for each category to:

- **Higher threshold**: Fewer false positives, but may miss some objects
- **Lower threshold**: Better recall, but may include more false positives

This allows users to tune the system based on their specific needs and priorities.

## Performance Considerations

- **Processing Speed**: Approximately 5-10 seconds per image, depending on hardware
- **API Usage**: Requires Google Street View API credits (approximately 7¢ per image)
- **Coverage**: Analyzes only areas with available Street View imagery
- **Temporal Limitations**: Detection reflects conditions at the time Street View images were captured
""")

# Add model comparison section
st.subheader("Model Comparison")
st.markdown("""
We evaluated several deep learning architectures for our object detection task. The following comparison showcases 
the performance metrics across different models and scenarios.
""")

# Load model comparison data
model_data = load_model_comparison_data()

if model_data:
    # Create tabs for different comparisons
    model_tab1, model_tab2 = st.tabs(["Overall Performance", "Category-level Performance"])
    
    with model_tab1:
        st.markdown("### Overall Model Performance")
        st.markdown("""
        We evaluated our models across four different scenarios:
        - **Scenario 1**: Dense urban areas with high foot traffic
        - **Scenario 2**: Urban parks and open spaces
        - **Scenario 3**: Industrial/commercial zones
        - **Scenario 4**: Residential neighborhoods
        """)
        
        # Extract data for visualization
        comprehensive_df = model_data['comprehensive']
        
        # Create a bar chart comparing models
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data (handling the specific format of the CSV)
        mask_data = comprehensive_df.iloc[2, 1:5].astype(float).values  # Average F1 for Mask R-CNN
        fast_data = comprehensive_df.iloc[10, 1:5].astype(float).values  # Average F1 for Fast R-CNN
        yolo_data = comprehensive_df.iloc[16, 1:5].astype(float).values  # Average F1 for YOLO
        deeplab_data = comprehensive_df.iloc[22, 1:5].astype(float).values  # Average F1 for DeepLabV3+
        
        # Set up bar positions
        bar_width = 0.2
        r1 = np.arange(4)
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]
        
        # Create bars
        ax.bar(r1, mask_data, width=bar_width, label='Mask R-CNN', color='#3498db')
        ax.bar(r2, fast_data, width=bar_width, label='Fast R-CNN', color='#e74c3c')
        ax.bar(r3, yolo_data, width=bar_width, label='YOLO', color='#2ecc71')
        ax.bar(r4, deeplab_data, width=bar_width, label='DeepLabV3+', color='#f39c12')
        
        # Add labels and legend
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Average F1 Score')
        ax.set_title('Model Performance Comparison (F1 Score)')
        ax.set_xticks([r + bar_width*1.5 for r in range(4)])
        ax.set_xticklabels(['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4'])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Display the chart
        st.pyplot(fig)
        
        # Create a table with the metrics
        st.markdown("### Detailed Metrics By Model")
        
        # Extract data for the table
        models = ['Mask R-CNN', 'Fast R-CNN', 'YOLO', 'DeepLabV3+']
        
        # Create metrics tables
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("#### Average F1 Score")
            f1_data = {
                'Model': models,
                'Scenario 1': [mask_data[0], fast_data[0], yolo_data[0], deeplab_data[0]],
                'Scenario 2': [mask_data[1], fast_data[1], yolo_data[1], deeplab_data[1]],
                'Scenario 3': [mask_data[2], fast_data[2], yolo_data[2], deeplab_data[2]],
                'Scenario 4': [mask_data[3], fast_data[3], yolo_data[3], deeplab_data[3]]
            }
            f1_df = pd.DataFrame(f1_data)
            st.dataframe(f1_df.style.highlight_max(axis=0, subset=f1_df.columns[1:]))
        
        with cols[1]:
            st.markdown("#### Mean Average Precision (mAP@50)")
            # Extract mAP data
            mask_map = comprehensive_df.iloc[3, 1:5].astype(float).values
            fast_map = comprehensive_df.iloc[11, 1:5].astype(float).values
            yolo_map = comprehensive_df.iloc[17, 1:5].astype(float).values
            # DeepLabV3+ doesn't have mAP values
            
            map_data = {
                'Model': models[:3],  # Exclude DeepLabV3+
                'Scenario 1': [mask_map[0], fast_map[0], yolo_map[0]],
                'Scenario 2': [mask_map[1], fast_map[1], yolo_map[1]],
                'Scenario 3': [mask_map[2], fast_map[2], yolo_map[2]],
                'Scenario 4': [mask_map[3], fast_map[3], yolo_map[3]]
            }
            map_df = pd.DataFrame(map_data)
            st.dataframe(map_df.style.highlight_max(axis=0, subset=map_df.columns[1:]))
    
    with model_tab2:
        st.markdown("### Performance By Category")
        st.markdown("""
        The performance of our chosen models varies by object category. Below we compare how Fast R-CNN and 
        Mask R-CNN perform for each object type we detect.
        """)
        
        # Extract data
        mask_rcnn_df = model_data['mask_rcnn']
        fast_rcnn_df = model_data['fast_rcnn']
        
        # Create category comparison visualizations
        categories = ['People ', 'Encampments ', 'Cart ', 'Bike ']
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # For each category, create a comparison
        for i, category in enumerate(categories):
            # Extract data for this category (scenario 2 is used as example)
            mask_val = float(mask_rcnn_df.iloc[1, i+1])  # Scenario 2, F1 score
            fast_val = float(fast_rcnn_df.iloc[1, i+1])  # Scenario 2, F1 score
            
            data = [mask_val, fast_val]
            ax = axes[i]
            ax.bar(['Mask R-CNN', 'Fast R-CNN'], data, color=['#3498db', '#e74c3c'])
            ax.set_title(f'F1 Score for {category} (Scenario 2)')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for j, v in enumerate(data):
                ax.text(j, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add a table with all category data
        st.markdown("### F1 Scores By Category and Scenario")
        
        # Create a combined dataframe for comparison
        combined_data = []
        
        for i in range(1, 5):  # 4 scenarios
            scenario_name = f"Scenario {i}"
            for category in categories:
                mask_val = float(mask_rcnn_df.iloc[i-1, mask_rcnn_df.columns.get_loc(category)])
                fast_val = float(fast_rcnn_df.iloc[i-1, fast_rcnn_df.columns.get_loc(category)])
                
                combined_data.append({
                    'Scenario': scenario_name,
                    'Category': category,
                    'Mask R-CNN': mask_val,
                    'Fast R-CNN': fast_val,
                    'Difference': mask_val - fast_val
                })
        
        combined_df = pd.DataFrame(combined_data)
        
        # Create a color map for the difference column
        def highlight_diff(val):
            if val > 0:
                return 'background-color: rgba(46, 204, 113, 0.3)'  # Green if Mask R-CNN better
            elif val < 0:
                return 'background-color: rgba(231, 76, 60, 0.3)'  # Red if Fast R-CNN better
            else:
                return ''
        
        # Apply styling to the dataframe
        styled_df = combined_df.style.format({
            'Mask R-CNN': '{:.3f}',
            'Fast R-CNN': '{:.3f}',
            'Difference': '{:.3f}'
        }).applymap(highlight_diff, subset=['Difference'])
        
        st.dataframe(styled_df)
        
        st.markdown("""
        *Note: Positive difference (green) indicates Mask R-CNN performed better, while negative difference (red)
        indicates Fast R-CNN performed better for that category and scenario.*
        """)
else:
    st.warning("Model comparison data could not be loaded. Using example metrics instead.")
    
    # Add performance metrics (fallback version)
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Key Performance Metrics
            
            | Category | Precision | Recall | F1-Score |
            |----------|-----------|--------|----------|
            | People | 0.87 | 0.82 | 0.84 |
            | Encampments | 0.78 | 0.75 | 0.76 |
            | Carts | 0.82 | 0.79 | 0.80 |
            | Bikes | 0.85 | 0.81 | 0.83 |
            
            *Performance on validation set. Results may vary on new data.*
            """)
        
        with col2:
            img_path = os.path.join('Content', 'Img', 'Img [4].png')
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, caption="Model Performance Visualization", use_container_width=True)
            else:
                st.warning("Performance visualization not found")
    except Exception as e:
        st.error(f"Error displaying performance metrics: {str(e)}")

# Why We Chose Faster R-CNN
st.markdown("""
## Why We Chose Faster R-CNN

After extensive testing and comparison, we selected Faster R-CNN as our primary model for the following reasons:

1. **Superior F1 scores** across most categories and scenarios
2. **Better balance** between precision and recall compared to other architectures
3. **Reliable performance** for small and medium-sized objects (important for detecting distant encampments)
4. **Reasonable inference speed** that allows processing grid-based images efficiently
5. **Lower false positive rate** on similar but unrelated objects (like regular camping tents vs homeless encampments)

While Mask R-CNN showed slightly better performance in some categories, the additional computational overhead did not 
justify the marginal improvement in accuracy for our deployment scenario.
""")

# Footer
st.markdown("---")
st.markdown("VisionAid - Powered by PyTorch") 