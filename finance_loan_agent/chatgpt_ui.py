"""
ChatGPT-style UI for Finance Loan Agent

This module provides a Streamlit-based web interface for the loan processing agent
with a ChatGPT-like user experience.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import random

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
    page_title="Loan Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like interface
def load_css():
    """Load custom CSS for ChatGPT-like interface."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #343541;
        color: #ECECF1;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #202123;
    }
    
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 0px;
        padding-bottom: 100px;
    }
    
    /* User message styling */
    .user-message {
        background-color: #343541;
        padding: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        width: 100%;
    }
    
    .user-message-content {
        max-width: 800px;
        margin: 0 auto;
        width: 100%;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #444654;
        padding: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        width: 100%;
    }
    
    .assistant-message-content {
        max-width: 800px;
        margin: 0 auto;
        width: 100%;
    }
    
    /* Avatar styling */
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 0.125rem;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .user-avatar {
        background-color: #5436DA;
        color: white;
    }
    
    .assistant-avatar {
        background-color: #19C37D;
        color: white;
    }
    
    /* Input area styling */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #343541;
        padding: 1rem 1rem 2rem 1rem;
        display: flex;
        justify-content: center;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    .input-box {
        max-width: 800px;
        width: 100%;
        border-radius: 0.375rem;
        border: 1px solid rgba(255,255,255,0.1);
        background-color: #40414F;
        padding: 0.75rem 1rem;
        color: white;
        font-size: 1rem;
        line-height: 1.5;
        resize: none;
        height: 52px;
        max-height: 200px;
        overflow-y: auto;
    }
    
    /* Button styling */
    .send-button {
        position: absolute;
        right: 1rem;
        bottom: 1rem;
        background-color: transparent;
        border: none;
        color: rgba(255,255,255,0.5);
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    
    .send-button:hover {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: rgba(255,255,255,0.5);
        border-radius: 50%;
        animation: typing-dot 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) {
        animation-delay: 0s;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing-dot {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.5;
        }
        30% {
            transform: translateY(-5px);
            opacity: 1;
        }
    }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9em;
        width: 100%;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    .dataframe thead tr {
        background-color: #19C37D;
        color: white;
        text-align: left;
    }
    
    .dataframe th,
    .dataframe td {
        padding: 12px 15px;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #3E3F4B;
    }
    
    .dataframe tbody tr:last-of-type {
        border-bottom: 2px solid #19C37D;
    }
    
    /* Code block styling */
    code {
        background-color: #2D2E3A;
        border-radius: 0.25rem;
        padding: 0.2rem 0.4rem;
        font-family: monospace;
    }
    
    pre {
        background-color: #2D2E3A;
        border-radius: 0.5rem;
        padding: 1rem;
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #202123;
        color: white;
    }
    
    /* Sidebar button */
    .new-chat-button {
        background-color: #19C37D;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .new-chat-button:hover {
        background-color: #15A36B;
    }
    
    /* Chat history buttons */
    .chat-history-button {
        background-color: transparent;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        cursor: pointer;
        width: 100%;
        text-align: left;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .chat-history-button:hover {
        background-color: #2A2B32;
    }
    
    /* Welcome message */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        color: #ECECF1;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: rgba(255,255,255,0.7);
    }
    
    .examples-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        width: 100%;
        margin-top: 2rem;
    }
    
    .example-card {
        background-color: #3E3F4B;
        border-radius: 0.5rem;
        padding: 1rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .example-card:hover {
        background-color: #4A4B59;
    }
    
    .example-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .example-text {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
    }
    
    /* Markdown styling */
    .stMarkdown a {
        color: #19C37D;
        text-decoration: none;
    }
    
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ECECF1;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #19C37D transparent transparent transparent;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #343541;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #19C37D;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    # Initialize chat history
    if 'chat_histories' not in st.session_state:
        st.session_state.chat_histories = {}
    
    # Initialize current chat ID
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = generate_chat_id()
    
    # Ensure current chat exists in histories
    if st.session_state.current_chat_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[st.session_state.current_chat_id] = []
    
    # Initialize loan agent
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
    
    # Initialize typing indicator
    if 'typing' not in st.session_state:
        st.session_state.typing = False

def generate_chat_id():
    """Generate a unique chat ID."""
    return f"chat_{int(time.time())}_{random.randint(1000, 9999)}"

def get_chat_title(messages):
    """Generate a title for the chat based on the first user message."""
    if not messages:
        return "New chat"
    
    first_message = messages[0]["content"]
    # Truncate to first 30 characters
    title = first_message[:30]
    if len(first_message) > 30:
        title += "..."
    
    return title

def display_sidebar():
    """Display the sidebar with chat history."""
    with st.sidebar:
        st.markdown('<div style="text-align: center; margin-bottom: 20px;"><h1>Loan Assistant</h1></div>', unsafe_allow_html=True)
        
        # New chat button
        if st.button("+ New Chat", key="new_chat", help="Start a new chat"):
            st.session_state.current_chat_id = generate_chat_id()
            st.session_state.chat_histories[st.session_state.current_chat_id] = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Chat History")
        
        # Display chat history
        for chat_id, messages in st.session_state.chat_histories.items():
            if messages:  # Only show non-empty chats
                title = get_chat_title(messages)
                if st.button(title, key=f"history_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This is a ChatGPT-style interface for the Loan Assistant.
        
        It can help you with:
        - Loan calculations
        - Interest rate recommendations
        - Loan application analysis
        - Amortization schedules
        - APR calculations
        
        **Features:**
        - Flat interest calculation
        - CIBIL score relaxation
        - Indian Rupee (₹) support
        """)
        
        st.markdown("---")
        st.markdown(f"Python version: {PY_VERSION.major}.{PY_VERSION.minor}.{PY_VERSION.micro}")
        if IS_PY_313_PLUS:
            st.warning("Running on Python 3.13+. Some features may use fallback implementations.")

def display_welcome_screen():
    """Display the welcome screen when no messages are present."""
    st.markdown("""
    <div class="welcome-container">
        <h1 class="welcome-title">Loan Assistant</h1>
        <p class="welcome-subtitle">Your AI-powered financial advisor for loan calculations and recommendations</p>
        
        <div class="examples-container">
            <div class="example-card" onclick="document.getElementById('user-input').value = 'I\'d like to apply for a loan. My annual income is ₹6,50,000, my CIBIL score is 780, and my debt-to-income ratio is 0.35. I\'m looking for a ₹2,50,000 loan for 48 months.'; document.getElementById('user-input').focus();">
                <div class="example-title">Apply for a loan</div>
                <div class="example-text">I'd like to apply for a loan. My annual income is ₹6,50,000, my CIBIL score is 780, and my debt-to-income ratio is 0.35...</div>
            </div>
            
            <div class="example-card" onclick="document.getElementById('user-input').value = 'What would be the monthly payment for a ₹3,00,000 loan at 10.5% flat interest for 60 months?'; document.getElementById('user-input').focus();">
                <div class="example-title">Calculate loan payment</div>
                <div class="example-text">What would be the monthly payment for a ₹3,00,000 loan at 10.5% flat interest for 60 months?</div>
            </div>
            
            <div class="example-card" onclick="document.getElementById('user-input').value = 'What interest rate would you recommend for someone with a 760 CIBIL score looking for a ₹5,00,000 loan for 36 months?'; document.getElementById('user-input').focus();">
                <div class="example-title">Get interest rate recommendation</div>
                <div class="example-text">What interest rate would you recommend for someone with a 760 CIBIL score looking for a ₹5,00,000 loan for 36 months?</div>
            </div>
            
            <div class="example-card" onclick="document.getElementById('user-input').value = 'Generate an amortization schedule for a ₹2,00,000 loan at 12% interest for 24 months.'; document.getElementById('user-input').focus();">
                <div class="example-title">Generate amortization schedule</div>
                <div class="example-text">Generate an amortization schedule for a ₹2,00,000 loan at 12% interest for 24 months.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_chat_messages():
    """Display the chat messages."""
    messages = st.session_state.chat_histories[st.session_state.current_chat_id]
    
    if not messages:
        display_welcome_screen()
        return
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-message-content">
                    <div style="display: flex;">
                        <div class="avatar user-avatar">👤</div>
                        <div style="flex-grow: 1;">{message["content"]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="assistant-message-content">
                    <div style="display: flex;">
                        <div class="avatar assistant-avatar">🤖</div>
                        <div style="flex-grow: 1;">{message["content"]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display typing indicator if needed
    if st.session_state.typing:
        st.markdown(f"""
        <div class="assistant-message">
            <div class="assistant-message-content">
                <div style="display: flex;">
                    <div class="avatar assistant-avatar">🤖</div>
                    <div style="flex-grow: 1;">
                        <div class="typing-indicator">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_input_area():
    """Display the input area for user messages."""
    st.markdown("""
    <div class="input-container">
        <textarea id="user-input" class="input-box" placeholder="Message Loan Assistant..." onkeydown="if(event.keyCode==13 && !event.shiftKey){document.getElementById('send-button').click(); return false;}"></textarea>
        <button id="send-button" class="send-button" onclick="handleSend()">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
            </svg>
        </button>
    </div>
    
    <script>
    function handleSend() {
        const userInput = document.getElementById('user-input').value.trim();
        if (userInput) {
            document.getElementById('user-input').value = '';
            window.parent.postMessage({type: 'streamlit:sendMessage', message: userInput}, '*');
        }
    }
    </script>
    """, unsafe_allow_html=True)

def handle_user_input():
    """Handle user input from the frontend."""
    # Create a container for the user input
    input_container = st.container()
    
    # Get the user input from the query parameters
    query_params = st.experimental_get_query_params()
    user_input = query_params.get("message", [""])[0]
    
    # Clear the query parameters
    if user_input:
        st.experimental_set_query_params()
        
        # Process the user input
        process_user_input(user_input)

def process_user_input(user_input):
    """Process the user input and generate a response."""
    if not user_input.strip():
        return
    
    # Add user message to chat history
    st.session_state.chat_histories[st.session_state.current_chat_id].append({
        "role": "user",
        "content": user_input
    })
    
    # Set typing indicator
    st.session_state.typing = True
    st.rerun()
    
    # Process with the agent
    try:
        if 'api_key_error' in st.session_state and st.session_state.api_key_error:
            response_text = """
            ⚠️ Google API key not configured. Please set the GOOGLE_API_KEY environment variable.
            
            1. Create a `.env` file in the project directory
            2. Add the following line: `GOOGLE_API_KEY=your_google_api_key_here`
            3. Restart the application
            """
        else:
            # Add a small delay to simulate typing
            time.sleep(1)
            
            # Get response from agent
            response = st.session_state.loan_agent.invoke({"input": user_input})
            response_text = response["output"]
    except Exception as e:
        response_text = f"Error processing message: {str(e)}"
    
    # Add assistant message to chat history
    st.session_state.chat_histories[st.session_state.current_chat_id].append({
        "role": "assistant",
        "content": response_text
    })
    
    # Clear typing indicator
    st.session_state.typing = False
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    # Load custom CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Display chat messages
    display_chat_messages()
    
    # Display input area
    display_input_area()
    
    # Handle user input
    handle_user_input()

if __name__ == "__main__":
    main()

