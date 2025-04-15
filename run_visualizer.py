#!/usr/bin/env python3
"""
Launcher script for the Homeless Detection visualization tool that handles PyTorch errors
by configuring Streamlit environment variables before launch.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the Homeless Detection Visualizer with error handling')
    parser.add_argument('--browser', action='store_true', help='Open browser automatically')
    parser.add_argument('--port', type=int, default=8502, help='Port to run Streamlit on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Set environment variables to prevent PyTorch errors with Streamlit
    os.environ["STREAMLIT_SERVER_WATCH_EXCLUDE_DIRS"] = "torch,torchvision,PIL,numpy"
    
    # Optional: Disable file watcher entirely for better stability
    if not args.debug:
        os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
    
    # Command to run Streamlit
    cmd = [
        "streamlit", "run", "visualize_results.py",
        "--server.port", str(args.port)
    ]
    
    if not args.browser:
        cmd.extend(["--server.headless", "true"])
    
    print(f"\n[INFO] Starting visualize_results.py on port {args.port}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 