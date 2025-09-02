# UI Bot Guide

This document provides information on using the UI Bot for the Finance Loan Agent.

## Overview

The UI Bot provides a user-friendly web interface for interacting with the loan processing agent. It allows users to:

1. Chat with the loan assistant
2. Submit loan applications through a form
3. Calculate loan terms using a simple calculator

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

### Running the UI Bot

Run the UI Bot using the provided script:

```bash
python finance_loan_agent/run_ui_bot.py
```

This will start a Streamlit server, and you can access the UI Bot at http://localhost:8501.

## Features

### Chat Interface

The chat interface allows you to interact with the loan assistant in a conversational manner. You can ask questions about loans, submit loan applications, and get recommendations.

Example queries:
- "I'd like to apply for a loan. My annual income is $65,000, my credit score is 720, and my debt-to-income ratio is 0.35. I'm looking for a $25,000 loan for 48 months."
- "What would be the monthly payment for a $30,000 loan at 5.2% interest for 60 months?"
- "What interest rate would you recommend for someone with a 680 credit score looking for a $15,000 loan for 36 months?"

### Loan Application Form

The loan application form provides a structured way to submit loan applications. Fill in the required information and click "Analyze Application" to get a risk assessment and loan recommendation.

Fields include:
- Annual Income
- Credit Score
- Debt-to-Income Ratio
- Loan Amount
- Loan Term
- Loan Purpose

### Loan Calculator

The loan calculator allows you to quickly calculate loan terms without submitting a full application. Enter the loan amount, interest rate, and loan term to calculate:
- Monthly Payment
- Total Payment
- Total Interest

## Customization

### Styling

The UI Bot uses a custom CSS file for styling. You can modify the `finance_loan_agent/ui_styles.css` file to change the appearance of the UI.

### Adding New Features

To add new features to the UI Bot, you can modify the `finance_loan_agent/ui_bot.py` file. The UI is built using Streamlit, which provides a simple way to create web applications with Python.

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

## Advanced Usage

### Embedding in Existing Web Applications

The UI Bot can be embedded in existing web applications using Streamlit's Component API or by running it as a separate service and embedding it using an iframe.

### Customizing the Agent

The UI Bot uses the same loan processing agent as the command-line interface. You can customize the agent by modifying the `finance_loan_agent/agent.py` file.

