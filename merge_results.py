import os
import pandas as pd
import argparse
import glob
import shutil
from datetime import datetime

def merge_csv_files(csv_files, output_file):
    """
    Merge multiple CSV files into one
    
    Args:
        csv_files (list): List of CSV file paths to merge
        output_file (str): Path to save the merged CSV file
    """
    if not csv_files:
        print("No CSV files found to merge")
        return None
    
    # Read all CSV files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if not dfs:
        print("No valid data found in CSV files")
        return None
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(dfs)} CSV files into {output_file}")
    
    return merged_df

def copy_image_files(image_dirs, output_dir):
    """
    Copy image files from multiple directories to one output directory
    
    Args:
        image_dirs (list): List of directories containing images
        output_dir (str): Directory to copy images to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    copied_count = 0
    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        image_files = glob.glob(os.path.join(dir_path, "*.jpg"))
        for image_file in image_files:
            filename = os.path.basename(image_file)
            output_path = os.path.join(output_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                continue
                
            try:
                shutil.copy2(image_file, output_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_file}: {str(e)}")
    
    print(f"Copied {copied_count} image files to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Merge results from multiple detection runs')
    parser.add_argument('--base_dirs', type=str, nargs='+', required=True,
                       help='Base directories containing results (e.g., results1 results2)')
    parser.add_argument('--output_dir', type=str, default=f'merged_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Output directory for merged results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories
    merged_original_dir = os.path.join(args.output_dir, "original")
    merged_predicted_dir = os.path.join(args.output_dir, "predicted")
    os.makedirs(merged_original_dir, exist_ok=True)
    os.makedirs(merged_predicted_dir, exist_ok=True)
    
    # Find CSV files
    csv_files = []
    original_dirs = []
    predicted_dirs = []
    
    for base_dir in args.base_dirs:
        # Find CSV file
        csv_path = os.path.join(base_dir, "predictions.csv")
        if os.path.exists(csv_path):
            csv_files.append(csv_path)
        
        # Find image directories
        original_dir = os.path.join(base_dir, "original")
        predicted_dir = os.path.join(base_dir, "predicted")
        
        if os.path.exists(original_dir):
            original_dirs.append(original_dir)
        if os.path.exists(predicted_dir):
            predicted_dirs.append(predicted_dir)
    
    # Merge CSV files
    merged_csv_path = os.path.join(args.output_dir, "predictions.csv")
    merged_df = merge_csv_files(csv_files, merged_csv_path)
    
    # Copy image files
    copy_image_files(original_dirs, merged_original_dir)
    copy_image_files(predicted_dirs, merged_predicted_dir)
    
    print(f"\nMerge completed. Results saved to {args.output_dir}")
    if merged_df is not None:
        print(f"Total detections: {len(merged_df)}")
        print(f"Detection classes: {merged_df['class'].unique()}")
        print(f"Number of unique images: {merged_df['filename'].nunique()}")

if __name__ == "__main__":
    main() 