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
    
    # Check if model file exists
    model_path = "model_final_2.pth"
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file {model_path} not found. Application may not work correctly.")
        cont = input("Continue anyway? (y/n): ")
        if cont.lower() != 'y':
            return
    
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