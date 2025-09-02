/**
 * ChatGPT-style UI JavaScript
 * 
 * This file contains JavaScript functions for the ChatGPT-style UI.
 */

// Function to handle sending a message
function handleSend() {
    const userInput = document.getElementById('user-input').value.trim();
    if (userInput) {
        document.getElementById('user-input').value = '';
        
        // Send the message to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setQueryParam',
            queryParams: { message: userInput }
        }, '*');
    }
}

// Function to handle example clicks
function useExample(example) {
    document.getElementById('user-input').value = example;
    document.getElementById('user-input').focus();
}

// Function to adjust textarea height based on content
function adjustTextareaHeight(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
    
    // Limit height to 200px
    if (textarea.scrollHeight > 200) {
        textarea.style.height = '200px';
        textarea.style.overflowY = 'auto';
    } else {
        textarea.style.overflowY = 'hidden';
    }
}

// Function to scroll to bottom of chat
function scrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up textarea auto-resize
    const textarea = document.getElementById('user-input');
    if (textarea) {
        textarea.addEventListener('input', function() {
            adjustTextareaHeight(this);
        });
        
        // Handle Enter key (send) and Shift+Enter (new line)
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });
    }
    
    // Set up send button
    const sendButton = document.getElementById('send-button');
    if (sendButton) {
        sendButton.addEventListener('click', handleSend);
    }
    
    // Set up example cards
    const exampleCards = document.querySelectorAll('.example-card');
    exampleCards.forEach(card => {
        card.addEventListener('click', function() {
            const example = this.getAttribute('data-example');
            useExample(example);
        });
    });
    
    // Scroll to bottom of chat
    scrollToBottom();
});

// Listen for messages from Streamlit
window.addEventListener('message', function(e) {
    if (e.data.type === 'streamlit:render') {
        // Scroll to bottom after render
        setTimeout(scrollToBottom, 100);
    }
});

