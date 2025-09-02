"""
UI Bot for Finance Loan Agent

This module provides a Streamlit-based web interface for the loan processing agent.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

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

# Set page configuration
st.set_page_config(
    page_title="Loan Processing Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'loan_agent' not in st.session_state:
        # Get Google API key from environment
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key or google_api_key == "your_google_api_key_here":
            st.session_state.api_key_error = True
        else:
            st.session_state.api_key_error = False
            try:
                st.session_state.loan_agent = create_loan_agent(api_key=google_api_key)
            except Exception as e:
                st.error(f"Error initializing loan agent: {str(e)}")
                st.session_state.api_key_error = True
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_header():
    """Display the application header."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/loan.png", width=80)
    with col2:
        st.title("Loan Processing Assistant")
        st.markdown("*Powered by AI to help you with loan applications*")
    
    st.markdown("---")

def display_sidebar():
    """Display the sidebar with information and settings."""
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This application uses AI to analyze loan applications, 
        assess credit risk, and provide recommendations for loan approval.
        
        It can help you:
        - Analyze loan applications
        - Calculate loan terms
        - Recommend interest rates
        - Find similar loan applications
        """
    )
    
    st.sidebar.header("Environment")
    st.sidebar.text(f"Python version: {PY_VERSION.major}.{PY_VERSION.minor}.{PY_VERSION.micro}")
    if IS_PY_313_PLUS:
        st.sidebar.warning("Running on Python 3.13+. Some features may use fallback implementations.")
    
    st.sidebar.header("Sample Queries")
    st.sidebar.markdown(
        """
        - *"I'd like to apply for a loan. My annual income is $65,000, my credit score is 720, and my debt-to-income ratio is 0.35. I'm looking for a $25,000 loan for 48 months."*
        
        - *"What would be the monthly payment for a $30,000 loan at 5.2% interest for 60 months?"*
        
        - *"What interest rate would you recommend for someone with a 680 credit score looking for a $15,000 loan for 36 months?"*
        """
    )

def display_loan_calculator():
    """Display a simple loan calculator."""
    st.header("Loan Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=25000, step=1000)
    
    with col2:
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    
    with col3:
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=48, step=12)
    
    if st.button("Calculate"):
        # Convert annual interest rate to monthly
        monthly_rate = interest_rate / 12 / 100
        
        # Calculate monthly payment
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term) / ((1 + monthly_rate) ** loan_term - 1)
        
        # Calculate total payment and interest
        total_payment = monthly_payment * loan_term
        total_interest = total_payment - loan_amount
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Payment", f"${monthly_payment:.2f}")
        
        with col2:
            st.metric("Total Payment", f"${total_payment:.2f}")
        
        with col3:
            st.metric("Total Interest", f"${total_interest:.2f}")

def display_loan_application_form():
    """Display a loan application form."""
    st.header("Loan Application Form")
    
    with st.form("loan_application_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=65000, step=5000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=10)
            debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=25000, step=1000)
            loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=48, step=12)
            loan_purpose = st.selectbox("Loan Purpose", ["Home Improvement", "Debt Consolidation", "Major Purchase", "Education", "Medical Expenses", "Other"])
        
        submitted = st.form_submit_button("Analyze Application")
        
        if submitted:
            if 'api_key_error' in st.session_state and st.session_state.api_key_error:
                st.error("Google API key not configured. Please set the GOOGLE_API_KEY environment variable.")
                return
            
            # Prepare the query
            query = (
                f"I'd like to apply for a loan. My annual income is ${annual_income}, "
                f"my credit score is {credit_score}, and my debt-to-income ratio is {debt_to_income_ratio}. "
                f"I'm looking for a ${loan_amount} loan for {loan_term} months for {loan_purpose}."
            )
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Process with the agent
            try:
                with st.spinner("Analyzing loan application..."):
                    response = st.session_state.loan_agent.invoke({"input": query})
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                st.error(f"Error processing loan application: {str(e)}")

def display_chat_interface():
    """Display a chat interface for interacting with the loan agent."""
    st.header("Chat with Loan Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
    
    # Chat input
    user_input = st.text_area("Your message:", height=100)
    
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Please enter a message.")
            return
        
        if 'api_key_error' in st.session_state and st.session_state.api_key_error:
            st.error("Google API key not configured. Please set the GOOGLE_API_KEY environment variable.")
            return
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process with the agent
        try:
            with st.spinner("Processing..."):
                response = st.session_state.loan_agent.invoke({"input": user_input})
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            
        # Rerun to update the display
        st.experimental_rerun()

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Check if API key is configured
    if 'api_key_error' in st.session_state and st.session_state.api_key_error:
        st.error("""
        Google API key not configured. Please set the GOOGLE_API_KEY environment variable.
        
        1. Create a `.env` file in the project directory
        2. Add the following line: `GOOGLE_API_KEY=your_google_api_key_here`
        3. Restart the application
        """)
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Loan Application", "Loan Calculator"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_loan_application_form()
    
    with tab3:
        display_loan_calculator()

if __name__ == "__main__":
    main()

