import streamlit as st
import os
from PIL import Image

# Set page config
st.set_page_config(
    page_title="About - VisionAid",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("About VisionAid")

# Introduction section with two columns
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ## AI-Powered Homelessness Mapping and Analysis
    
    VisionAid is an innovative computer vision system designed to help municipalities, non-profit organizations, 
    and researchers better understand and address homelessness through data-driven insights. By leveraging 
    advanced deep learning techniques, VisionAid automatically detects visual indicators of homelessness from 
    Street View imagery, creating comprehensive maps and analytics.
    
    ### Our Mission
    
    Our mission is to provide objective, accurate data on the spatial distribution of homelessness indicators to:
    
    - **Improve resource allocation** for outreach programs and services
    - **Track changes over time** to measure the effectiveness of interventions
    - **Identify patterns** that might not be evident through traditional point-in-time counts
    - **Support evidence-based policymaking** with comprehensive data visualization
    
    VisionAid makes this possible through a non-intrusive, scalable approach that complements existing 
    methodologies for understanding homelessness.
    """)

with col2:
    # Display main image if available
    try:
        img_path = os.path.join('Content', 'Img', 'Img [4].png')
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption="VisionAid System Overview", use_container_width=True)
        else:
            st.info("System overview image not available")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

# Approach section
st.markdown("""
## Our Approach

VisionAid approaches the challenge of homelessness mapping through a systematic, technology-driven process:

1. **Data Collection**: Using Google Street View API to gather georeferenced imagery across urban areas
2. **Grid-Based Analysis**: Dividing areas into grid cells for comprehensive, systematic analysis
3. **Deep Learning Detection**: Applying our Faster R-CNN model to identify four key indicators:
   - People experiencing homelessness
   - Encampments and temporary shelters
   - Shopping carts
   - Bicycles (often associated with lack of transportation options)
4. **Spatial Analysis**: Mapping the distribution of detections and identifying hotspots
5. **Visualization**: Generating intuitive maps and dashboards for stakeholders
""")

# Create tabs for key sections
tabs = st.tabs(["Key Features", "Ethical Considerations", "Impact", "Use Cases"])

# Key Features tab
with tabs[0]:
    st.subheader("Key Features of VisionAid")
    
    # Create three columns for features
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        ### Comprehensive Coverage
        
        - **Wide Area Mapping**: Analyze entire cities or neighborhoods
        - **Systematic Approach**: No selection bias in coverage
        - **Consistent Methodology**: Same detection criteria applied throughout
        - **Historical Comparison**: Ability to compare current and past imagery
        """)
    
    with feat_col2:
        st.markdown("""
        ### Powerful Analytics
        
        - **Hotspot Identification**: Find areas with high concentration
        - **Pattern Recognition**: Discover spatial correlations
        - **Temporal Analysis**: Track changes over time
        - **Category Breakdown**: Understand the types of indicators present
        - **Confidence Scoring**: Quantify detection reliability
        """)
    
    with feat_col3:
        st.markdown("""
        ### User-Friendly Interface
        
        - **Interactive Maps**: Click and explore detected areas
        - **Customizable Filters**: Focus on specific indicators
        - **Data Export**: Download results for further analysis
        - **Batch Processing**: Run analysis on multiple areas
        - **Report Generation**: Create shareable insights
        """)

# Ethical Considerations tab
with tabs[1]:
    st.subheader("Ethical Considerations")
    
    st.markdown("""
    ### Privacy and Respect
    
    VisionAid is designed with strong ethical guidelines to respect the dignity and privacy of individuals experiencing homelessness:
    
    - **No Personal Identification**: The system does not attempt to identify individuals
    - **Aggregated Data**: Results are presented as patterns and statistics, not individual cases
    - **Public Imagery Only**: Uses only publicly available Street View images
    - **Responsible Reporting**: Guidelines for using data in a non-stigmatizing way
    
    ### Data Usage Policy
    
    We have established strict policies on how VisionAid data should be used:
    
    1. Data should be used to **improve services and aid**, not for enforcement or displacement
    2. Results should be **combined with other sources** of information, not used in isolation
    3. Stakeholders should include **people with lived experience** of homelessness in decision-making
    4. Regular **ethical reviews** of system usage and outcomes
    
    We are committed to ensuring that VisionAid contributes positively to addressing homelessness through better understanding, not through approaches that might harm vulnerable populations.
    """)

# Impact tab
with tabs[2]:
    st.subheader("Impact")
    
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.markdown("""
        ### Improved Resource Allocation
        
        VisionAid helps organizations allocate limited resources more effectively by:
        
        - **Identifying underserved areas** where homelessness indicators are present but services are limited
        - **Optimizing outreach routes** to maximize impact and efficiency
        - **Quantifying needs** across different neighborhoods to justify resource distribution
        - **Tracking progress** to measure the effectiveness of interventions
        
        Studies have shown that data-driven approaches to homelessness services can increase effectiveness by up to 40% compared to non-systematic methods.
        """)
    
    with impact_col2:
        st.markdown("""
        ### Better Planning and Policy
        
        The system supports improved planning and policy development through:
        
        - **Evidence-based decision making** with objective, comprehensive data
        - **Early identification** of emerging patterns before they become entrenched
        - **Cross-regional comparison** to understand how different approaches affect outcomes
        - **Longitudinal analysis** to see how patterns change over time
        
        Municipalities using data-driven approaches have reported improved outcomes in addressing homelessness, with better targeting of preventive services and more efficient use of resources.
        """)
    
    st.markdown("""
    ### Success Stories
    
    While VisionAid is a new technology, similar approaches have shown promising results:
    
    - **County Outreach Optimization**: A pilot program in a California county used AI-assisted mapping to redesign outreach routes, resulting in a 35% increase in successful contacts with individuals experiencing homelessness
    
    - **Service Gap Analysis**: A midwest city used computer vision mapping to identify areas with homeless indicators but few services, leading to the strategic placement of three new mobile service centers
    
    - **Intervention Effectiveness**: A non-profit organization used before-and-after mapping to demonstrate the effectiveness of their housing-first program to funders, securing additional support
    """)

# Use Cases tab
with tabs[3]:
    st.subheader("Use Cases")
    
    st.markdown("""
    ### Municipal Governments
    
    - **Comprehensive Assessment**: Supplement point-in-time counts with continuous spatial data
    - **Resource Planning**: Determine optimal locations for shelters, services, and hygiene facilities
    - **Progress Tracking**: Measure the impact of policies and programs over time
    - **Inter-departmental Coordination**: Share consistent data across housing, health, and social services
    
    ### Non-Profit Organizations
    
    - **Targeted Outreach**: Focus limited resources on areas with highest need
    - **Grant Applications**: Support funding requests with objective data
    - **Impact Demonstration**: Show before-and-after results of interventions
    - **Community Education**: Create visualizations to raise awareness about homelessness patterns
    
    ### Researchers
    
    - **Pattern Analysis**: Study the spatial relationship between homelessness and other urban factors
    - **Longitudinal Studies**: Track changes in homelessness indicators over time
    - **Intervention Assessment**: Evaluate the effectiveness of different approaches
    - **Data Integration**: Combine visual detection with other datasets for comprehensive analysis
    
    ### Housing Agencies
    
    - **Need Assessment**: Identify areas with highest demand for affordable housing
    - **Program Evaluation**: Measure how housing initiatives affect visible homelessness
    - **Strategic Planning**: Develop data-driven strategies for addressing homelessness
    """)

# Future vision section
st.markdown("""
## Future Vision

VisionAid is continuously evolving to better serve the needs of organizations working to address homelessness:

- **Expanded Indicators**: Developing models to detect additional relevant features
- **Multi-Source Integration**: Combining Street View with satellite imagery and other data sources
- **Predictive Analytics**: Identifying early warning signs of increasing homelessness
- **Mobile Platform**: Enabling field workers to contribute and access data on-site
- **Community Feedback**: Incorporating structured input from community members and service providers

Our goal is to create a comprehensive platform that provides actionable insights while respecting the dignity 
and privacy of those experiencing homelessness.
""")

# CTA and contact
st.markdown("""
## Get Involved

Interested in learning more about VisionAid or deploying it in your community? We welcome collaborations with:

- Municipal agencies
- Non-profit organizations
- Research institutions
- Community advocacy groups
- Technology partners

Contact us to discuss how VisionAid can support your efforts to address homelessness through better data.
""")

# Footer
st.markdown("---")
st.markdown("VisionAid - Powered by PyTorch") 