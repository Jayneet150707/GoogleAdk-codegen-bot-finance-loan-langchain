"""
Run Agent Module with Python 3.13.2 compatibility

This module provides the main entry point for running the loan processing agent.
"""

import os
import sys
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Check if we're using Python 3.13+
PY_VERSION = sys.version_info
IS_PY_313_PLUS = PY_VERSION.major == 3 and PY_VERSION.minor >= 13

# Import agent with proper error handling
try:
    from .agent import create_loan_agent
except ImportError:
    # Handle relative import error when running as script
    from agent import create_loan_agent

# Load environment variables
load_dotenv()

def check_environment() -> Dict[str, Any]:
    """
    Check if the environment is properly configured.
    
    Returns:
        Dict: Status of environment checks
    """
    status = {
        "python_version": f"{PY_VERSION.major}.{PY_VERSION.minor}.{PY_VERSION.micro}",
        "is_py_313_plus": IS_PY_313_PLUS,
        "google_api_key": False,
        "mongodb_connection": False
    }
    
    # Check Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key and google_api_key != "your_google_api_key_here":
        status["google_api_key"] = True
    
    # Check MongoDB connection string
    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    if mongodb_connection_string and mongodb_connection_string != "your_mongodb_connection_string_here":
        status["mongodb_connection"] = True
    
    return status

def main() -> None:
    """Run the loan processing agent."""
    print("\n===== Loan Processing Agent =====")
    print("Starting environment checks...")
    
    # Check environment
    env_status = check_environment()
    print(f"Python version: {env_status['python_version']}")
    
    if env_status["is_py_313_plus"]:
        print("⚠️ Running on Python 3.13+. Some features may use fallback implementations.")
    
    # Check if Google API key is configured
    if not env_status["google_api_key"]:
        print("❌ ERROR: Google API key not found or is default value.")
        print("Please set the GOOGLE_API_KEY environment variable in the .env file.")
        return
    
    # Check MongoDB connection
    if not env_status["mongodb_connection"]:
        print("⚠️ WARNING: MongoDB connection string not found or is default value.")
        print("Vector search functionality will not be available.")
    
    # Create the agent with the API key
    print("\nInitializing loan processing agent...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    try:
        loan_agent = create_loan_agent(api_key=google_api_key)
        print("✅ Loan Processing Agent initialized successfully!")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize agent: {str(e)}")
        return
    
    print("\nLoan Processing Agent is ready! Type 'quit' to exit.")
    print("Example: 'I'd like to apply for a loan. My annual income is $65,000, my credit score is 720, and my debt-to-income ratio is 0.35. I'm looking for a $25,000 loan for 48 months.'")
    
    while True:
        # Get user input
        try:
            user_input = input("\nYou > ")
        except KeyboardInterrupt:
            print("\nExiting loan processing agent.")
            break
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting loan processing agent.")
            break
        
        # Run the agent
        try:
            response = loan_agent.invoke({"input": user_input})
            print("\nAgent > ", end="")
            print(response["output"])
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()

