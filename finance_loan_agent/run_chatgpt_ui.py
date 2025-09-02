"""
Run ChatGPT-style UI Module

This module provides the main entry point for running the loan processing ChatGPT-style UI.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_streamlit_installed():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit package."""
    print("Streamlit is not installed. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install Streamlit. Please install it manually with 'pip install streamlit'.")
        return False

def main():
    """Run the loan processing ChatGPT-style UI."""
    print("\n===== Loan Processing ChatGPT-style UI =====")
    
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        if not install_streamlit():
            return
    
    # Check if Google API key is configured
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == "your_google_api_key_here":
        print("⚠️ WARNING: Google API key not found or is default value.")
        print("Please set the GOOGLE_API_KEY environment variable in the .env file.")
        print("The UI will still start, but functionality will be limited.")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the path to the ChatGPT UI script
    chatgpt_ui_path = os.path.join(script_dir, "chatgpt_ui.py")
    
    # Run Streamlit
    print("\nStarting Streamlit server...")
    print("You can access the ChatGPT-style UI at http://localhost:8501")
    print("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            chatgpt_ui_path, 
            "--server.headless", "true",
            "--theme.base", "dark"
        ])
    except KeyboardInterrupt:
        print("\nStopping Streamlit server...")
    except Exception as e:
        print(f"\nError running Streamlit: {str(e)}")

if __name__ == "__main__":
    main()

