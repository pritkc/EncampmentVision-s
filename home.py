import streamlit as st
import os
import base64
from PIL import Image

# Set page config
st.set_page_config(
    page_title="VisionAid - Home",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Welcome to VisionAid")
st.write("""
## Overview

VisionAid is an application that uses artificial intelligence to detect encampment-related objects
in Google Street View images. It provides visualization tools and analytics for understanding
the spatial distribution of encampment indicators.

## Key Features

- Area selection using latitude/longitude coordinates
- Automatic retrieval of Google Street View images
- Detecting multiple categories of objects related to encampments
- Interactive map visualization of detection results
- Statistical analysis and reporting capabilities

## Getting Started

Select a page from the sidebar to begin:

- **About**: Learn more about the project and its goals
- **Mechanism**: Understand how the detection system works
- **Results**: View example results and analyses
- **Meet the Team**: Information about the project contributors
""")

# Load and display example image if available
try:
    img_path = os.path.join('Content', 'Img', 'Img [4].png')
    if os.path.exists(img_path):
        image = Image.open(img_path)
        st.image(image, caption="VisionAid Detection Example", use_container_width=True)
    else:
        st.info("Example image not found. Navigate to the 'Results' page to see detection examples.")
except Exception as e:
    st.error(f"Error loading example image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2023 BDA600 Project Team | Powered by Streamlit") 