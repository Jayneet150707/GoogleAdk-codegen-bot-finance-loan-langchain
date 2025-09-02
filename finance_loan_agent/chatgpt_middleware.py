"""
ChatGPT Middleware Module

This module provides middleware functionality for the ChatGPT-style UI,
handling communication between the frontend and the agent.
"""

import streamlit as st
import time
import json
from typing import Dict, Any, List, Optional

class ChatGPTMiddleware:
    """Middleware for handling communication between the frontend and the agent."""
    
    def __init__(self, agent):
        """
        Initialize the middleware.
        
        Args:
            agent: The loan processing agent
        """
        self.agent = agent
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'processing' not in st.session_state:
            st.session_state.processing = False
    
    def add_user_message(self, message: str):
        """
        Add a user message to the chat history.
        
        Args:
            message: The user message
        """
        st.session_state.messages.append({
            "role": "user",
            "content": message
        })
    
    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the chat history.
        
        Args:
            message: The assistant message
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": message
        })
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get the chat history.
        
        Returns:
            List of message dictionaries
        """
        return st.session_state.messages
    
    def clear_messages(self):
        """Clear the chat history."""
        st.session_state.messages = []
    
    def is_processing(self) -> bool:
        """
        Check if a message is being processed.
        
        Returns:
            True if a message is being processed, False otherwise
        """
        return st.session_state.processing
    
    def set_processing(self, processing: bool):
        """
        Set the processing state.
        
        Args:
            processing: The processing state
        """
        st.session_state.processing = processing
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: The user message
            
        Returns:
            The assistant response
        """
        self.set_processing(True)
        
        try:
            # Process the message with the agent
            response = self.agent.invoke({"input": message})
            response_text = response["output"]
        except Exception as e:
            response_text = f"Error processing message: {str(e)}"
        
        self.set_processing(False)
        return response_text
    
    def process_message_streaming(self, message: str, callback=None):
        """
        Process a user message and generate a streaming response.
        
        Args:
            message: The user message
            callback: Callback function for each token
        """
        self.set_processing(True)
        
        try:
            # Process the message with the agent
            response = self.agent.invoke({"input": message})
            response_text = response["output"]
            
            # Simulate streaming by splitting the response into tokens
            tokens = response_text.split()
            partial_response = ""
            
            for token in tokens:
                partial_response += token + " "
                if callback:
                    callback(partial_response)
                time.sleep(0.05)  # Simulate typing delay
        except Exception as e:
            response_text = f"Error processing message: {str(e)}"
            if callback:
                callback(response_text)
        
        self.set_processing(False)
        return response_text

