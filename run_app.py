#!/usr/bin/env python3
"""
Launcher script for the Encampment Detection app that handles PyTorch errors
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Encampment Detection app with error handling')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the Streamlit app on')
    parser.add_argument('--browser', action='store_true', help='Open browser automatically')
    parser.add_argument('--app', type=str, default='app.py', help='App file to run (defaults to app.py)')
    args = parser.parse_args()
    
    # Set environment variables to prevent PyTorch errors with Streamlit
    os.environ["STREAMLIT_SERVER_WATCH_EXCLUDE_DIRS"] = "torch,torchvision,PIL,numpy"
     
    # Optional: Disable file watcher entirely for better stability
    if not args.debug:
        os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
    
    # Construct the command
    cmd = [sys.executable, "-m", "streamlit", "run", args.app]
    
    # Add arguments
    if args.debug:
        cmd.append("--debug")
    
    if args.port != 8501:
        cmd.extend(["--server.port", str(args.port)])
    
    if not args.browser:
        cmd.extend(["--server.headless", "true"])
    
    # Run the command
    try:
        print(f"\n[INFO] Starting {args.app} on port {args.port}...")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {str(e)}")
    
if __name__ == "__main__":
    main() 