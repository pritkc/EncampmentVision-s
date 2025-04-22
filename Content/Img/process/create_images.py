import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_example_image(title, width=800, height=400, bg_color=(240, 242, 246)):
    """Create an example image with a title and placeholder content"""
    # Create new image with background
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a system font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw title
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (width - text_width) // 2
    y = 20
    draw.text((x, y), title, fill=(33, 33, 33), font=font)
    
    # Draw placeholder content
    draw.rectangle([50, 80, width-50, height-50], outline=(200, 200, 200), width=2)
    placeholder_text = "Example visualization - Replace with actual screenshot"
    text_bbox = draw.textbbox((0, 0), placeholder_text, font=small_font)
    text_width = text_bbox[2] - text_bbox[0]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), placeholder_text, fill=(150, 150, 150), font=small_font)
    
    return image

def main():
    # Create process directory if it doesn't exist
    os.makedirs(".", exist_ok=True)
    
    # Create example images for each step
    images = {
        "area_selection.png": "Area Selection Interface",
        "image_acquisition.png": "Street View Image Collection",
        "model_processing.png": "Model Architecture & Processing",
        "object_detection.png": "Object Detection Results",
        "visualization.png": "Detection Visualization",
        "analysis.png": "Analysis & Reporting"
    }
    
    for filename, title in images.items():
        image = create_example_image(title)
        image.save(filename)
        print(f"Created {filename}")

if __name__ == "__main__":
    main() 