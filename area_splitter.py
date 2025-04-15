import numpy as np
import pandas as pd
import os
import argparse
from shapely.geometry import Polygon, box
import geopandas as gpd
from matplotlib import pyplot as plt

def split_area(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, 
               rows=2, cols=2, output_file="area_chunks.csv"):
    """
    Split a large bounding box into smaller chunks.
    
    Args:
        top_left_lat (float): Latitude of the top-left corner
        top_left_lon (float): Longitude of the top-left corner
        bottom_right_lat (float): Latitude of the bottom-right corner
        bottom_right_lon (float): Longitude of the bottom-right corner
        rows (int): Number of rows to split into
        cols (int): Number of columns to split into
        output_file (str): Path to save the CSV file with chunk coordinates
    """
    # Calculate steps for latitude and longitude
    lat_step = (bottom_right_lat - top_left_lat) / rows
    lon_step = (bottom_right_lon - top_left_lon) / cols
    
    chunks = []
    
    # Generate chunks
    for i in range(rows):
        for j in range(cols):
            chunk_tl_lat = top_left_lat + i * lat_step
            chunk_tl_lon = top_left_lon + j * lon_step
            chunk_br_lat = top_left_lat + (i + 1) * lat_step
            chunk_br_lon = top_left_lon + (j + 1) * lon_step
            
            chunks.append({
                'chunk_id': f"chunk_{i}_{j}",
                'top_left_lat': chunk_tl_lat,
                'top_left_lon': chunk_tl_lon,
                'bottom_right_lat': chunk_br_lat,
                'bottom_right_lon': chunk_br_lon
            })
    
    # Create DataFrame
    chunks_df = pd.DataFrame(chunks)
    
    # Save to CSV
    chunks_df.to_csv(output_file, index=False)
    print(f"Created {len(chunks)} chunks and saved to {output_file}")
    
    return chunks_df

def visualize_chunks(chunks_df, output_file="area_chunks_visualization.png"):
    """
    Create a visualization of the area chunks
    
    Args:
        chunks_df (pd.DataFrame): DataFrame with chunk coordinates
        output_file (str): Path to save the visualization
    """
    # Create GeoDataFrame with polygons
    polygons = []
    for _, row in chunks_df.iterrows():
        poly = box(row['top_left_lon'], row['bottom_right_lat'], 
                   row['bottom_right_lon'], row['top_left_lat'])
        polygons.append(poly)
    
    gdf = gpd.GeoDataFrame(chunks_df, geometry=polygons)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(ax=ax, alpha=0.5, edgecolor='k')
    
    # Add chunk IDs as labels
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['chunk_id'], 
                fontsize=10, ha='center', va='center')
    
    plt.title("Area Split into Chunks")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Split a large area into smaller chunks')
    parser.add_argument('--top_left_lat', type=float, required=True, 
                        help='Latitude of the top-left corner')
    parser.add_argument('--top_left_lon', type=float, required=True, 
                        help='Longitude of the top-left corner')
    parser.add_argument('--bottom_right_lat', type=float, required=True, 
                        help='Latitude of the bottom-right corner')
    parser.add_argument('--bottom_right_lon', type=float, required=True, 
                        help='Longitude of the bottom-right corner')
    parser.add_argument('--rows', type=int, default=2, 
                        help='Number of rows to split into')
    parser.add_argument('--cols', type=int, default=2, 
                        help='Number of columns to split into')
    parser.add_argument('--output', type=str, default='area_chunks.csv', 
                        help='Output CSV file path')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualization of chunks')
    
    args = parser.parse_args()
    
    chunks_df = split_area(
        args.top_left_lat, args.top_left_lon,
        args.bottom_right_lat, args.bottom_right_lon,
        args.rows, args.cols, args.output
    )
    
    if args.visualize:
        visualize_chunks(chunks_df)

if __name__ == "__main__":
    main() 