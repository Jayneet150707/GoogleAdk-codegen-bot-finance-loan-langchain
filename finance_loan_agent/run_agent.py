import os
from dotenv import load_dotenv
from agent import create_loan_agent

load_dotenv()

def main():
    """Run the loan processing agent."""
    print("Starting loan processing agent...")
    
    # Check if Google API key is configured
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key is None or google_api_key == "your_google_api_key_here":
        print("ERROR: Google API key not found or is default value.")
        print("Please set the GOOGLE_API_KEY environment variable in the .env file.")
        return
    
    # Create the agent with the API key
    loan_agent = create_loan_agent(api_key=google_api_key)
    
    print("\nLoan Processing Agent is ready! Type 'quit' to exit.")
    print("Example: 'I'd like to apply for a loan. My annual income is $65,000, my credit score is 720, and my debt-to-income ratio is 0.35. I'm looking for a $25,000 loan for 48 months.'")
    
    while True:
        # Get user input
        user_input = input("\nYou > ")
        
        # Check if user wants to quit
        if user_input.lower() == 'quit':
            print("Exiting loan processing agent.")
            break
        
        # Run the agent
        try:
            response = loan_agent.invoke({"input": user_input})
            print("\nAgent > ", end="")
            print(response["output"])
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()

