import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont

# Set page config
st.set_page_config(
    page_title="Team - VisionAid",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to create a placeholder profile image
def create_placeholder_image(initials, size=(150, 150), bg_color=(70, 130, 180), text_color=(255, 255, 255)):
    """Create a placeholder profile image with initials"""
    # Create image with background color
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Use default font
    font = ImageFont.load_default()
    
    # Draw text in the center
    w, h = draw.textbbox((0, 0), initials, font=font)[2:4]
    draw.text(((size[0]-w)/2, (size[1]-h)/2), initials, font=font, fill=text_color)
    
    return image

# Title
st.title("Meet the Team")

st.markdown("""
## The People Behind VisionAid

VisionAid was developed by a team of data scientists and engineers committed to using 
technology to address social challenges. Meet our core team members below.
""")

# Create team members with clean, professional information
team_members = [
    {
        'name': 'Xinyu Du',
        'title': 'Project Coordinator + Data',
        'bio': 'Xinyu coordinates project activities and ensures seamless collaboration across teams. With strong organizational skills and technical knowledge, she effectively bridges communication between stakeholders and development teams.',
        'contributions': ['Project management', 'Technical direction', 'Mapping systems']
    },

    {
        'name': 'Ethan',
        'title': 'Machine Learning',
        'bio': 'Ethan spearheaded the data processing pipeline and model experimentation. His expertise in data augmentation and transformation techniques significantly improved model performance and reliability.',
        'contributions': ['Augmentation process', 'Data transformation', 'Experimental setup', 'Results analysis']
    },

    {
        'name': 'Prit',
        'title': 'Software',
        'bio': 'Prit architected and developed the VisionAid Streamlit application from the ground up, creating an intuitive interface for visualizing homeless encampment data on maps',
        'contributions': ['App Development', 'Deployment', 'Data curation', 'Map integration']
    },
]

# Display team members in two rows of three
# First row
st.markdown("### Leadership & Development")
cols1 = st.columns(3)

# Second row
cols2 = st.columns(3)

# Combine columns for iteration
all_cols = cols1 + cols2

# Display each team member
for i, member in enumerate(team_members):
    with all_cols[i]:
        # Generate initials for the placeholder image
        name_parts = member['name'].split()
        initials = name_parts[0][0] + name_parts[-1][0]
        
        # Generate a color based on name
        import hashlib
        name_hash = int(hashlib.md5(member['name'].encode('utf-8')).hexdigest(), 16)
        r = ((name_hash & 0xFF0000) >> 16) % 200 + 56
        g = ((name_hash & 0x00FF00) >> 8) % 200 + 56
        b = (name_hash & 0x0000FF) % 200 + 56
        bg_color = (r, g, b)
        
        # Create and display the profile image
        profile_img = create_placeholder_image(initials, bg_color=bg_color)
        st.image(profile_img, width=150)
        
        # Display member information
        st.markdown(f"### {member['name']}")
        st.markdown(f"**{member['title']}**")
        st.markdown(member['bio'])
        
        # Display contributions
        st.markdown("**Key Contributions:**")
        for contribution in member['contributions']:
            st.markdown(f"- {contribution}")

# Acknowledgments section
st.markdown("---")
st.markdown("### Acknowledgments")
st.markdown("""
We would also like to thank our partners and supporters:

- Research institutions and universities that provided expertise
- Community organizations working on homelessness issues
- Open-source contributors to the libraries and tools used in this project
- Google for access to the Street View API
""")

# Footer
st.markdown("---")
st.markdown("© 2023 VisionAid Team") 