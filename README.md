# Homeless Detection System

This application uses a trained Faster R-CNN model to detect homeless-related objects in Google Street View images.

## Prerequisites

- Python 3.8 or later
- A valid Google Street View API key
- PyTorch and other dependencies (see requirements.txt)

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your trained model file(s) in the `models` directory (`.pth` files)

## Running the Application

To avoid PyTorch-related errors with Streamlit, use the provided launcher scripts:

### Main Detection App

```bash
python run_app.py --browser
```

Options:
- `--browser`: Open the browser automatically
- `--port <number>`: Run on a specific port (default: 8501)
- `--debug`: Enable debug mode (includes file watching)

### Results Visualization

To view previously generated results:

```bash
python run_visualizer.py --browser
```

Options:
- `--browser`: Open the browser automatically
- `--port <number>`: Run on a specific port (default: 8502)
- `--debug`: Enable debug mode

## Troubleshooting

### PyTorch Path Errors

If you see errors related to `torch._classes.__path__._path` or similar, make sure to:
1. Use the provided launcher scripts instead of running `streamlit run app.py` directly
2. Update to the latest version of PyTorch
3. Try running with the `--debug` flag disabled

### Map Display Issues

If the map doesn't display correctly:
1. Check that you have internet connectivity
2. Verify that all image paths are correct
3. Try refreshing the page

## Features

- Select specific categories to detect (People, Encampments, Bikes, Carts)
- Set individual confidence thresholds per category
- Draw bounding boxes on maps to select areas
- View detection results on an interactive map
- Generate statistics and visualizations

## Directory Structure

- `app.py`: Main application file
- `visualize_results.py`: Results viewer
- `run_app.py` & `run_visualizer.py`: Launcher scripts
- `models/`: Directory for model files
- `results/`: Directory for detection results
- `homeless_detection/`: Utility functions package

## Overview

This application uses a trained Faster R-CNN model to analyze Google Street View images from a specified area and detect:
- Homeless People
- Homeless Encampments
- Homeless Carts
- Homeless Bikes

The system processes images from a grid of locations within a specified area, runs the detection model, and visualizes the results on an interactive map.

## Features

- Interactive area selection using latitude/longitude coordinates
- Configurable grid density for detailed area coverage
- Adjustable confidence threshold for detections
- Map visualization of detected objects
- Statistical analysis of detection results
- CSV export of detection data
- Interactive results exploration

## Requirements

This project requires Python 3.10 or higher and Streamlit 1.31.0 or higher. If you encounter any UI-related errors, please ensure you have the correct Streamlit version installed:

```bash
pip install --upgrade streamlit>=1.31.0
```

Or install all requirements:

```bash
pip install -r requirements.txt --upgrade
```

## Installation

### Using Virtual Environment (Recommended)

1. Clone the repository:
```
git clone <repository-url>
cd homeless-detection-system
```

2. Create a virtual environment:
```
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

1. Make sure to park your model inside the `models` directory. This app will provide dropdown option to select any one model if parked more than one.

## API Setup

This application requires a Google Street View API key to fetch images. To obtain a key:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to APIs & Services > Credentials
4. Click "Create credentials" and select "API key"
5. Enable the "Street View Static API" in the API Library
6. Set up billing for your Google Cloud project (required for API usage)

Once you have your API key, you can enter it in the application's sidebar when running the app.

## Usage

### Running the Main Application

1. Run the Streamlit application using the provided script:
```
python run_app.py --browser
```

2. In the web interface:
   - Enter your Google Street View API key
   - Define the bounding box for the area you want to analyze
   - Adjust the grid dimensions and confidence threshold
   - Click "Run Detection" to start the process

3. View and explore the results in the Map, Statistics, and Raw Data tabs

### Visualizing Existing Results

To view previously generated results:
```
python run_app.py --app visualize_results.py --browser
```

## Project Structure

The project consists of the following components:

### Core Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application for homeless detection |
| `visualize_results.py` | Standalone app for viewing previously generated results |
| `area_splitter.py` | Utility for dividing large areas into smaller chunks |
| `merge_results.py` | Tool for combining results from multiple detection runs |
| `run_app.py` | Helper script to run the applications |
| `setup.py` | Package installation configuration |
| `requirements.txt` | List of project dependencies |
| `model_final_2.pth` | Pre-trained Faster R-CNN model for homeless detection |

### Package Structure

The `homeless_detection` package contains reusable components:

| File | Description |
|------|-------------|
| `homeless_detection/__init__.py` | Package initialization |
| `homeless_detection/utils.py` | Utility functions for model loading, image processing, and visualization |

## Detailed Component Descriptions

### Main Application (`app.py`)

The main application provides a user interface for configuring and running the homeless detection process.

**Key Functions:**
- `load_model()`: Loads the pre-trained detection model
- `draw_predictions()`: Draws bounding boxes on detected objects
- `process_image()`: Processes a single image through the detection model
- `run_detection()`: Main function that fetches and processes images from a grid
- `display_results_map()`: Visualizes detection results on an interactive map
- `display_summary_stats()`: Generates statistical visualizations of detections

### Results Viewer (`visualize_results.py`)

A standalone application for exploring detection results after they've been generated.

**Key Functions:**
- `load_results()`: Loads detection data from CSV files
- `display_results_map()`: Creates an interactive map with detection markers
- `display_summary_stats()`: Generates statistical charts from detection data
- `display_image_grid()`: Shows a grid of detected images with filtering capabilities

### Area Splitting (`area_splitter.py`)

A utility for dividing large geographic areas into smaller, manageable chunks for processing.

**Key Functions:**
- `split_area()`: Divides a bounding box into smaller chunks
- `visualize_chunks()`: Creates a visualization of the area divisions
- Command-line interface for use in scripts or automation

### Results Merging (`merge_results.py`)

A tool for combining detection results from multiple runs, useful when processing large areas in chunks.

**Key Functions:**
- `merge_csv_files()`: Combines multiple CSV files into one
- `copy_image_files()`: Copies images from multiple directories
- Command-line interface for use in scripts or automation

### Runner Script (`run_app.py`)

A helper script that simplifies launching the applications with proper configuration.

**Features:**
- Checks for the model file before starting
- Option to automatically install dependencies
- Configurable port and browser settings
- Support for running different app components

### Utilities (`homeless_detection/utils.py`)

Shared utility functions used across the applications.

**Key Functions:**
- `load_model()`: Loads the detection model with proper configuration
- `draw_predictions()`: Common function for drawing bounding boxes
- `process_image()`: Shared image processing logic
- `create_detection_map()`: Creates interactive maps with consistent styling
- `create_summary_charts()`: Generates consistent statistical visualizations

## Project Flow

1. **Setup Phase**:
   - User installs dependencies and prepares the model
   - User obtains a Google Street View API key

2. **Configuration Phase**:
   - User selects an area using coordinates
   - User configures grid density and detection threshold
   - For large areas, `area_splitter.py` can be used to create manageable chunks

3. **Detection Phase**:
   - System queries Google Street View API for images
   - Images are processed through the detection model
   - Detection results are saved to files and CSV

4. **Visualization Phase**:
   - Results are displayed on an interactive map
   - Statistical summaries and charts are generated
   - Raw data is available for export

5. **Optional Post-Processing**:
   - Results from multiple runs can be merged using `merge_results.py`
   - Saved results can be explored using `visualize_results.py`

## Notes

- The application requires a valid Google Street View API key with proper quota
- Processing may take some time depending on the grid size (number of locations)
- Temporary files are stored in the `temp_images` directory
- Results are saved in the `results` directory
- The model can detect four classes of objects related to homelessness

## Credits

This application was developed as part of BDA600 project, based on a pre-trained model for homeless detection using street view imagery.

## License

[MIT License](LICENSE)
