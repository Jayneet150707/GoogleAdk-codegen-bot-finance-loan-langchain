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
    from .tools.finance_tools import (
        calculate_loan_terms,
        generate_amortization_schedule,
        calculate_effective_annual_rate,
        calculate_apr,
        recommend_interest_rate
    )
except ImportError:
    # Handle relative import error when running as script
    from agent import create_loan_agent
    from tools.finance_tools import (
        calculate_loan_terms,
        generate_amortization_schedule,
        calculate_effective_annual_rate,
        calculate_apr,
        recommend_interest_rate
    )

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
        st.markdown("*Powered by AI to help you with loan applications in ₹*")
    
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
        - Calculate loan terms with flat interest
        - Recommend interest rates with CIBIL score relaxation
        - Find similar loan applications
        - Generate amortization schedules
        """
    )
    
    st.sidebar.header("Environment")
    st.sidebar.text(f"Python version: {PY_VERSION.major}.{PY_VERSION.minor}.{PY_VERSION.micro}")
    if IS_PY_313_PLUS:
        st.sidebar.warning("Running on Python 3.13+. Some features may use fallback implementations.")
    
    st.sidebar.header("Interest Rate Information")
    st.sidebar.info(
        """
        **Flat Interest System**
        
        Interest is calculated on the full principal throughout the loan term.
        
        **CIBIL Score Relaxation**
        - Score > 800: -3.0% reduction
        - Score 775-799: -2.0% reduction
        - Score 751-774: -1.0% reduction
        - Score ≤ 750: Standard adjustments apply
        """
    )
    
    st.sidebar.header("Sample Queries")
    st.sidebar.markdown(
        """
        - *"I'd like to apply for a loan. My annual income is ₹6,50,000, my CIBIL score is 780, and my debt-to-income ratio is 0.35. I'm looking for a ₹2,50,000 loan for 48 months."*
        
        - *"What would be the monthly payment for a ₹3,00,000 loan at 10.5% flat interest for 60 months?"*
        
        - *"What interest rate would you recommend for someone with a 760 CIBIL score looking for a ₹5,00,000 loan for 36 months?"*
        """
    )

def display_loan_calculator():
    """Display a simple loan calculator."""
    st.header("Loan Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=250000, step=10000)
    
    with col2:
        interest_rate = st.number_input("Interest Rate (%)", min_value=7.0, max_value=18.0, value=12.0, step=0.1)
    
    with col3:
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=48, step=12)
    
    # Interest type selection
    interest_type = st.radio("Interest Type", ["Flat", "Reducing Balance"], horizontal=True, index=0)
    
    if st.button("Calculate"):
        # Calculate loan terms
        terms = calculate_loan_terms(
            loan_amount, 
            loan_term, 
            interest_rate, 
            interest_type.lower(), 
            "INR"
        )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Payment", f"₹{terms['monthly_payment']:,.2f}")
        
        with col2:
            st.metric("Total Payment", f"₹{terms['total_payment']:,.2f}")
        
        with col3:
            st.metric("Total Interest", f"₹{terms['total_interest']:,.2f}")
        
        # Display interest type
        st.info(f"Interest Type: {terms['interest_type']}")
        
        # Calculate effective annual rate
        ear = calculate_effective_annual_rate(interest_rate)
        st.info(f"Effective Annual Rate (EAR): {ear:.2f}%")
        
        # Option to show amortization schedule
        if st.checkbox("Show Amortization Schedule"):
            schedule = generate_amortization_schedule(
                loan_amount, 
                loan_term, 
                interest_rate, 
                interest_type.lower(), 
                "INR"
            )
            
            # Convert to DataFrame for display
            df = pd.DataFrame(schedule)
            
            # Format columns
            df['payment'] = df['payment'].map('₹{:,.2f}'.format)
            df['principal'] = df['principal'].map('₹{:,.2f}'.format)
            df['interest'] = df['interest'].map('₹{:,.2f}'.format)
            df['remaining_balance'] = df['remaining_balance'].map('₹{:,.2f}'.format)
            
            # Rename columns
            df = df.rename(columns={
                'period': 'Payment #',
                'payment': 'Payment',
                'principal': 'Principal',
                'interest': 'Interest',
                'remaining_balance': 'Remaining Balance'
            })
            
            # Drop currency columns for display
            if 'currency' in df.columns:
                df = df.drop(columns=['currency', 'currency_symbol'])
            
            # Display the schedule
            st.subheader("Amortization Schedule")
            st.dataframe(df, use_container_width=True)
            
            # Display a chart of principal vs interest over time
            st.subheader("Principal vs Interest Over Time")
            
            # Convert back to numeric for charting
            chart_data = pd.DataFrame(schedule)
            
            # Create a stacked bar chart
            chart_data = chart_data.set_index('period')
            st.bar_chart(chart_data[['principal', 'interest']])

def display_interest_rate_calculator():
    """Display an interest rate calculator with CIBIL score relaxation."""
    st.header("Interest Rate Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=10)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=250000, step=10000, key="ir_loan_amount")
    
    with col2:
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=48, step=12, key="ir_loan_term")
        interest_type = st.radio("Interest Type", ["Flat", "Reducing Balance"], horizontal=True, index=0, key="ir_interest_type")
    
    if st.button("Calculate Recommended Rate"):
        # Calculate recommended interest rate
        recommendation = recommend_interest_rate(
            credit_score, 
            loan_term, 
            loan_amount, 
            interest_type.lower()
        )
        
        # Display results
        st.metric("Recommended Interest Rate", f"{recommendation['interest_rate']}%")
        
        # Display details
        st.subheader("Rate Calculation Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Base Rate: {recommendation['base_rate']}%")
            st.write(f"Credit Score Adjustment: {recommendation['credit_adjustment']}%")
            st.write(f"Term Adjustment: {recommendation['term_adjustment']}%")
        
        with col2:
            st.write(f"Amount Adjustment: {recommendation['amount_adjustment']}%")
            st.write(f"Interest Type: {recommendation['interest_type']}")
            
            # Display CIBIL relaxation info
            if recommendation['relaxation_applied']:
                st.success("✅ CIBIL Score Relaxation Applied!")
            else:
                st.info("CIBIL Score Relaxation not applicable (score ≤ 750)")

def display_apr_calculator():
    """Display an APR calculator."""
    st.header("APR Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=250000, step=10000, key="apr_loan_amount")
        interest_rate = st.number_input("Interest Rate (%)", min_value=7.0, max_value=18.0, value=12.0, step=0.1, key="apr_interest_rate")
    
    with col2:
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=48, step=12, key="apr_loan_term")
        fees = st.number_input("Processing Fees (₹)", min_value=0, max_value=100000, value=5000, step=1000)
    
    # Interest type selection
    interest_type = st.radio("Interest Type", ["Flat", "Reducing Balance"], horizontal=True, index=0, key="apr_interest_type")
    
    if st.button("Calculate APR"):
        # Calculate APR
        apr = calculate_apr(loan_amount, interest_rate, loan_term, fees, interest_type.lower())
        
        # Display results
        st.metric("Annual Percentage Rate (APR)", f"{apr:.2f}%")
        
        # Display explanation
        st.info(
            """
            The Annual Percentage Rate (APR) includes both the interest rate and any fees charged.
            It represents the true cost of borrowing and is typically higher than the stated interest rate.
            """
        )
        
        # Display processing fee percentage
        fee_percentage = (fees / loan_amount) * 100
        st.write(f"Processing Fee: ₹{fees:,.2f} ({fee_percentage:.2f}% of loan amount)")

def display_loan_application_form():
    """Display a loan application form."""
    st.header("Loan Application Form")
    
    with st.form("loan_application_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            annual_income = st.number_input("Annual Income (₹)", min_value=0, max_value=10000000, value=650000, step=50000)
            credit_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=10, key="app_credit_score")
            debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        
        with col2:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=250000, step=10000, key="app_loan_amount")
            loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=48, step=12, key="app_loan_term")
            loan_purpose = st.selectbox("Loan Purpose", ["Home Improvement", "Debt Consolidation", "Major Purchase", "Education", "Medical Expenses", "Other"])
        
        # Interest type selection
        interest_type = st.radio("Interest Type", ["Flat", "Reducing Balance"], horizontal=True, index=0, key="app_interest_type")
        
        submitted = st.form_submit_button("Analyze Application")
        
        if submitted:
            if 'api_key_error' in st.session_state and st.session_state.api_key_error:
                st.error("Google API key not configured. Please set the GOOGLE_API_KEY environment variable.")
                return
            
            # Prepare the query
            query = (
                f"I'd like to apply for a loan. My annual income is ₹{annual_income:,}, "
                f"my CIBIL score is {credit_score}, and my debt-to-income ratio is {debt_to_income_ratio}. "
                f"I'm looking for a ₹{loan_amount:,} loan for {loan_term} months for {loan_purpose} "
                f"with {interest_type.lower()} interest."
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
        try:
            # Try the new method first (Streamlit >= 1.27.0)
            st.rerun()
        except AttributeError:
            # Fall back to the old method for older Streamlit versions
            try:
                st.experimental_rerun()
            except AttributeError:
                # If both fail, just inform the user to refresh
                st.info("Please refresh the page to see your message and response.")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Chat", 
        "Loan Application", 
        "Loan Calculator", 
        "Interest Rate Calculator", 
        "APR Calculator"
    ])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_loan_application_form()
    
    with tab3:
        display_loan_calculator()
    
    with tab4:
        display_interest_rate_calculator()
    
    with tab5:
        display_apr_calculator()

if __name__ == "__main__":
    main()

