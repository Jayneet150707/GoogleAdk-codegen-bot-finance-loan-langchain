# Finance Loan Agent with LangChain and MongoDB

This project demonstrates how to build an AI-powered loan processing agent using LangChain and MongoDB Atlas Vector Search. The agent can analyze loan applications, assess credit risk, and provide recommendations for loan approval.

## Features

- **Credit Risk Assessment**: Deep learning model to predict default risk
- **Vector Search**: Find similar loan applications using MongoDB Atlas Vector Search
- **Loan Term Calculation**: Calculate monthly payments and total interest
- **Interest Rate Recommendation**: Suggest interest rates based on applicant profile
- **Explainable AI**: Provide explanations for risk assessments

## Project Structure

```
finance_loan_agent/
├── __init__.py
├── agent.py
├── models/
│   ├── __init__.py
│   ├── credit_scoring.py
│   └── risk_assessment.py
├── tools/
│   ├── __init__.py
│   ├── mongodb_tools.py
│   └── finance_tools.py
├── data/
│   └── sample_loan_data.csv
├── train_model.py
├── run_agent.py
└── .env.example
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and update the values:

```bash
cp finance_loan_agent/.env.example finance_loan_agent/.env
```

Edit the `.env` file to add your Google API key and MongoDB connection string:

```
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# MongoDB Configuration
MONGODB_CONNECTION_STRING=your_mongodb_connection_string_here
```

### 3. Train the Model

```bash
cd finance_loan_agent
python train_model.py
```

This will:
- Train the credit scoring model using the sample data
- Create text embeddings for vector search
- Store the embeddings in MongoDB (if configured)

### 4. Set Up MongoDB Atlas Vector Search

If you're using MongoDB Atlas Vector Search:

1. Go to the MongoDB Atlas UI
2. Navigate to your cluster
3. Go to the "Search" tab
4. Create a new index with the following configuration:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 128,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

### 5. Run the Agent

```bash
python run_agent.py
```

## Example Usage

Once the agent is running, you can interact with it by typing queries like:

```
I'd like to apply for a loan. My annual income is $65,000, my credit score is 720, and my debt-to-income ratio is 0.35. I'm looking for a $25,000 loan for 48 months.
```

The agent will analyze the application and provide a risk assessment, recommended interest rate, and monthly payment information.

## Advanced Features

- **Multi-Agent System**: Create specialized agents for different aspects of loan processing
- **Document Processing**: Extract information from loan documents using OCR and NLP
- **Fraud Detection**: Identify potentially fraudulent applications
- **Customer Communication**: Generate personalized communications to loan applicants
- **Portfolio Management**: Analyze and manage the overall loan portfolio risk

## License

This project is licensed under the MIT License - see the LICENSE file for details.

