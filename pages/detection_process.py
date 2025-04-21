import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(layout="wide")

def create_example_image(title, width=800, height=400, bg_color=(245, 247, 250)):
    """Create an example image with a title and placeholder content"""
    # Create new image with background
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Use default font since we can't rely on system fonts
    font = ImageFont.load_default()
    
    # Draw title
    title_size = 20
    draw.text((width//2 - len(title)*4, 20), title, fill=(33, 33, 33))
    
    # Draw placeholder content
    draw.rectangle([50, 80, width-50, height-50], outline=(180, 190, 210), width=3)
    placeholder_text = "Example visualization"
    draw.text((width//2 - len(placeholder_text)*4, height//2), placeholder_text, fill=(100, 110, 130))
    
    return image

# Custom CSS for better styling
st.markdown("""
<style>
    .step-container {
        padding: 15px 0;
        margin: 5px 0;
    }
    .icon-container {
        background-color: #4e8cff;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px auto;
        font-size: 1.5em;
        color: white;
    }
    .connector-line {
        background-color: #e0e0e0;
        width: 2px;
        height: 40px;
        margin: 0 auto;
    }
    .step-title {
        color: #1f1f1f;
        font-size: 22px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .image-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px;
        margin: 10px 0;
    }
    .image-caption {
        text-align: center;
        color: #666;
        font-size: 14px;
        margin-top: 5px;
    }
    .main-title {
        color: #1f1f1f;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Apply a cleaner title
st.markdown("<h1 class='main-title'>Detection Process Flow</h1>", unsafe_allow_html=True)
st.markdown("---")

# Add tabs for different views
tabs = st.tabs(["Process Flow", "Urban Comparison"])

with tabs[0]:
    # Define the process steps with detailed information
    steps = [
        {
            "title": "1. User Input & Area Selection",
            "description": """
            The process begins with user interaction:
            
            1. **Map Selection**: Users draw a bounding box on an interactive map to define the area of interest
            2. **Grid Configuration**: Set the density of sampling points (rows × columns)
            3. **Detection Settings**:
                - Select object categories to detect (People, Encampments, Carts, Bikes)
                - Adjust confidence thresholds for each category
                - Choose the appropriate detection model
            """,
            "icon": "🎯",
            "image_title": "Area Selection Interface"
        },
        {
            "title": "2. Image Acquisition",
            "description": """
            The system automatically collects Street View imagery:
            
            1. **Grid Generation**: 
                - Convert the selected area into a grid of latitude/longitude points
                - Calculate optimal spacing for thorough coverage
            
            2. **API Integration**:
                - Make requests to Google Street View API
                - Fetch high-resolution images (640×640 pixels)
                - Capture multiple angles (0° and 180°) at each point
            
            3. **Image Management**:
                - Save images with metadata (location, date, heading)
                - Organize into temporary storage for processing
            """,
            "icon": "📸",
            "image_title": "Street View Image Collection"
        },
        {
            "title": "3. Model Processing",
            "description": """
            Images are prepared and processed through our custom Faster R-CNN model:
            
            1. **Image Preprocessing**:
                - Resize images to model input size (800×800)
                - Normalize pixel values
                - Convert to PyTorch tensors
            
            2. **Model Architecture**:
                - Backbone: ResNet-50 with FPN
                - Custom detection heads for each category
                - Optimized anchor sizes for street-level imagery
            """,
            "icon": "🤖",
            "image_title": "Model Architecture & Processing"
        },
        {
            "title": "4. Object Detection",
            "description": """
            The model performs detection and filtering:
            
            1. **Model Inference**:
                - Forward pass through the network
                - Generate region proposals
                - Classify and refine bounding boxes
            
            2. **Post-processing**:
                - Apply Non-Maximum Suppression (NMS)
                - Filter by confidence thresholds
                - Scale predictions back to original image size
            """,
            "icon": "🔍",
            "image_title": "Object Detection Results"
        },
        {
            "title": "5. Results Visualization",
            "description": """
            Detections are visualized for user interpretation:
            
            1. **Bounding Box Drawing**:
                - Color-coded boxes by category
                - Confidence scores display
                - Clear labeling
            
            2. **Image Processing**:
                - Save original and annotated versions
                - Maintain high resolution for detail
                - Organize by location and timestamp
            """,
            "icon": "🎨",
            "image_title": "Detection Visualization"
        },
        {
            "title": "6. Analysis & Reporting",
            "description": """
            Results are aggregated and presented:
            
            1. **Data Organization**:
                - Create structured detection records
                - Link images with metadata
                - Generate CSV exports
            
            2. **Interactive Visualization**:
                - Plot detections on interactive maps
                - Generate statistical summaries
                - Create detailed reports
            
            3. **Result Storage**:
                - Save all processed data
                - Maintain organized file structure
                - Enable easy access and sharing
            """,
            "icon": "📊",
            "image_title": "Analysis & Reporting"
        }
    ]

    # Create a vertical timeline-style layout
    for i, step in enumerate(steps):
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            
            # Create two columns: one narrow for icon, one wide for content
            col1, col2 = st.columns([0.1, 0.9])
            
            # Icon column
            with col1:
                st.markdown(f"""
                <div class="icon-container">
                    {step['icon']}
                </div>
                """, unsafe_allow_html=True)
                
                # Vertical line connector (except for last item)
                if i < len(steps) - 1:
                    st.markdown('<div class="connector-line"></div>', unsafe_allow_html=True)
            
            # Content column
            with col2:
                st.markdown(f"<h3 class='step-title'>{step['title']}</h3>", unsafe_allow_html=True)
                # Use standard Streamlit markdown instead of HTML to prevent rendering issues
                st.markdown(step['description'])
                
                # Create and display example image
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                image = create_example_image(step['image_title'])
                st.image(image, use_container_width=True)
                st.markdown(f"<p class='image-caption'>{step['image_title']}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Urban Comparison Analysis")
    
    st.markdown("""
    ### Los Angeles vs. San Francisco Detection Performance
    
    Our system was deployed in both Los Angeles and San Francisco to evaluate detection performance
    across different urban environments. This analysis provides insights into how the model performs
    in these distinct urban settings.
    """)
    
    # Hard-coded data for LA and SF (based on actual model results from the CSV)
    # These values are taken directly from the fast_r_cnn_-_(la_and_sf_data).csv file
    categories = ['People', 'Encampments', 'Carts', 'Bikes']
    la_f1 = [0.9245, 0.8807, 0.9811, 0.9811]  # LA F1 scores (Scenario 1)
    sf_f1 = [0.9057, 0.5422, 0.9434, 0.9434]  # SF F1 scores (Scenario 2)
    
    # Create comparison visualization
    import matplotlib.pyplot as plt
    import numpy as np
    
    # F1 Score comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    bar_width = 0.35
    r1 = np.arange(len(categories))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    ax.bar(r1, la_f1, width=bar_width, label='Los Angeles', color='#3498db')
    ax.bar(r2, sf_f1, width=bar_width, label='San Francisco', color='#e74c3c')
    
    # Add labels and legend
    ax.set_xlabel('Category')
    ax.set_ylabel('F1 Score')
    ax.set_title('Detection Performance: LA vs SF')
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the chart
    st.pyplot(fig)
    
    # Create a metrics comparison
    st.markdown("### Key Metrics Comparison")
    
    metric_cols = st.columns(4)
    
    for i, category in enumerate(categories):
        with metric_cols[i]:
            la_val = la_f1[i]
            sf_val = sf_f1[i]
            difference = ((sf_val - la_val) / la_val) * 100
            
            st.metric(
                label=category, 
                value=f"LA: {la_val:.2f}",
                delta=f"SF: {difference:.1f}% {'higher' if difference > 0 else 'lower'}"
            )
    
    # Add analysis insights
    st.markdown("""
    ### Regional Analysis Insights
    
    Our comparison reveals several key differences in detection performance between Los Angeles and San Francisco:
    
    1. **Consistent person detection** with slightly better performance in Los Angeles
    2. **Encampment detection significantly varies** with much better results in Los Angeles (+62.4%)
    3. **Cart and bike detection maintains high F1 scores** in both cities
    4. **Overall model performance is more consistent in Los Angeles** across all categories
    
    These variations likely reflect differences in urban density, built environment characteristics,
    and the distinct visual presentations of homelessness indicators in each city.
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style="margin-top: 20px; text-align: center;">
    <p style="color: #666;">This visualization represents the complete pipeline of our encampment detection system.</p>
</div>
""", unsafe_allow_html=True) 