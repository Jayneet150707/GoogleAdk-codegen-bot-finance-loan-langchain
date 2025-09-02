# ChatGPT-Style UI Guide

This document provides information on using the ChatGPT-style UI for the Finance Loan Agent.

## Overview

The ChatGPT-style UI provides a user-friendly interface for interacting with the loan processing agent, similar to the popular ChatGPT interface. It offers a clean, intuitive chat experience with features like:

1. Chat history management
2. Dark mode interface
3. Typing indicators
4. Example suggestions
5. Multi-chat support

## Getting Started

### Prerequisites

- Python 3.13.2 or later
- Streamlit (installed automatically if not present)
- Google API key (for the Generative AI model)

### Installation

1. Make sure you have installed all the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment variables by creating a `.env` file:
   ```
   # Google API Configuration
   GOOGLE_API_KEY=your_google_api_key_here
   
   # MongoDB Configuration (optional)
   MONGODB_CONNECTION_STRING=your_mongodb_connection_string_here
   ```

### Running the ChatGPT-Style UI

Run the ChatGPT-style UI using the provided script:

```bash
python finance_loan_agent/run_chatgpt_ui.py
```

This will start a Streamlit server, and you can access the ChatGPT-style UI at http://localhost:8501.

## Features

### Chat Interface

The chat interface allows you to interact with the loan assistant in a conversational manner. You can ask questions about loans, submit loan applications, and get recommendations.

Example queries:
- "I'd like to apply for a loan. My annual income is ₹6,50,000, my CIBIL score is 780, and my debt-to-income ratio is 0.35. I'm looking for a ₹2,50,000 loan for 48 months."
- "What would be the monthly payment for a ₹3,00,000 loan at 10.5% flat interest for 60 months?"
- "What interest rate would you recommend for someone with a 760 CIBIL score looking for a ₹5,00,000 loan for 36 months?"

### Chat History

The UI maintains a history of your conversations, allowing you to:
- Start new chats
- Switch between previous chats
- Continue conversations where you left off

Each chat is automatically titled based on the first message you send.

### Example Suggestions

When you start a new chat, the UI provides example suggestions to help you get started. These examples cover common use cases like:
- Applying for a loan
- Calculating loan payments
- Getting interest rate recommendations
- Generating amortization schedules

Click on any example to automatically fill the input field with that query.

## UI Components

### Sidebar

The sidebar provides access to:
- New chat button
- Chat history
- Information about the assistant
- Environment details

### Main Chat Area

The main chat area displays:
- User messages (with user avatar)
- Assistant responses (with assistant avatar)
- Typing indicator when the assistant is generating a response

### Input Area

The input area at the bottom of the screen allows you to:
- Type your message
- Send by clicking the send button or pressing Enter
- Use Shift+Enter for multi-line messages

## Customization

### Styling

The ChatGPT-style UI uses custom CSS for styling. You can modify the styles by editing the CSS in the `chatgpt_ui.py` file.

### Adding New Features

To add new features to the ChatGPT-style UI, you can modify the `chatgpt_ui.py` file. The UI is built using Streamlit, which provides a simple way to create web applications with Python.

## Troubleshooting

### API Key Issues

If you see an error message about the Google API key, make sure you have:
1. Created a `.env` file in the project directory
2. Added your Google API key to the `.env` file
3. Restarted the application

### Streamlit Installation Issues

If Streamlit fails to install automatically, you can install it manually:

```bash
pip install streamlit
```

### Python 3.13.2 Compatibility

When running with Python 3.13.2, you may see a warning message indicating that some features may use fallback implementations. This is normal and indicates that the code is using the compatibility layer for TensorFlow.

## Differences from Standard UI

The ChatGPT-style UI differs from the standard UI in several ways:

1. **Focus on Conversation**: The ChatGPT-style UI focuses on the conversation experience, making it more natural to interact with the assistant.

2. **No Separate Calculators**: Instead of having separate calculator tabs, all functionality is accessed through the chat interface.

3. **Dark Mode**: The ChatGPT-style UI uses a dark mode theme for reduced eye strain.

4. **Multi-Chat Support**: You can maintain multiple separate conversations and switch between them.

5. **Simplified Interface**: The interface is cleaner and more focused on the conversation, without additional UI elements.

## Conclusion

The ChatGPT-style UI provides a modern, intuitive interface for interacting with the loan processing agent. It combines the power of the loan processing agent with the familiar experience of ChatGPT, making it easy for users to get the information they need.

