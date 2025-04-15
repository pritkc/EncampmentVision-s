#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the Homeless Detection System app')
    parser.add_argument('--app', type=str, default='app.py', 
                        choices=['app.py', 'visualize_results.py'],
                        help='Which app to run (default: app.py)')
    parser.add_argument('--port', type=int, default=8501,
                        help='Port to run the app on (default: 8501)')
    parser.add_argument('--browser', action='store_true',
                        help='Open browser automatically')
    parser.add_argument('--install', action='store_true',
                        help='Install dependencies before running')
    
    args = parser.parse_args()
    
    # Check if model files exist in models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"[WARNING] Models directory '{models_dir}' not found. Please create this directory.")
        cont = input("Continue anyway? (y/n): ")
        if cont.lower() != 'y':
            return
    else:
        # Find all .pth files in the models directory
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        if not model_files:
            print(f"[WARNING] No model files (.pth) found in '{models_dir}' directory. Please add a model file.")
            cont = input("Continue anyway? (y/n): ")
            if cont.lower() != 'y':
                return
        elif len(model_files) > 1:
            print(f"[INFO] Multiple model files found in '{models_dir}': {', '.join(model_files)}")
            print(f"[INFO] The application will allow you to select which model to use.")
        else:
            print(f"[INFO] Using model: {model_files[0]}")
            if model_files[0] != "model_final_2.pth":
                print(f"[INFO] Note: The recommended filename is 'model_final_2.pth'")
    
    # Install dependencies if requested
    if args.install:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the app
    print(f"Starting {args.app} on port {args.port}...")
    cmd = [
        "streamlit", "run", 
        args.app,
        "--server.port", str(args.port)
    ]
    
    if not args.browser:
        cmd.extend(["--server.headless", "true"])
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 