import pandas as pd
import os

def convert_excel_to_csv():
    # Paths
    excel_path = 'Content/HomelessDetection_Results.xlsx'
    output_dir = 'data'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Excel file
    xls = pd.ExcelFile(excel_path)
    
    # Convert each sheet to a CSV file
    for sheet_name in xls.sheet_names:
        if not sheet_name.startswith('Sheet'):  # Skip unnamed sheets
            # Read the sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Create CSV filename based on sheet name
            csv_filename = f"{sheet_name.replace(' ', '_').lower()}.csv"
            output_path = os.path.join(output_dir, csv_filename)
            
            # Save as CSV
            df.to_csv(output_path, index=False)
            print(f"Converted sheet '{sheet_name}' to {output_path}")

if __name__ == "__main__":
    convert_excel_to_csv() 