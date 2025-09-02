from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import importlib.util
import sys
import re
from typing import Dict, Any, Union, List, Optional

# Check if we're using Python 3.13+
PY_VERSION = sys.version_info
IS_PY_313_PLUS = PY_VERSION.major == 3 and PY_VERSION.minor >= 13

# Import tools with proper error handling
try:
    from .tools.finance_tools import (
        analyze_loan_application, 
        get_similar_loan_applications,
        calculate_loan_terms,
        recommend_interest_rate
    )
except ImportError:
    # Handle relative import error when running as script
    from tools.finance_tools import (
        analyze_loan_application, 
        get_similar_loan_applications,
        calculate_loan_terms,
        recommend_interest_rate
    )

# --- Utility to safely parse inputs ---
def safe_parse_input(x: Any) -> Dict[str, Any]:
    """
    Ensure input is always a dictionary.
    
    Args:
        x: Input that might be a string or dictionary
        
    Returns:
        Dict: Parsed dictionary from input
        
    Raises:
        ValueError: If input is a string but not valid JSON
    """
    if isinstance(x, str):
        # Remove markdown code block formatting if present
        x = re.sub(r'^```(?:json)?\s*', '', x)
        x = re.sub(r'\s*```$', '', x)
        
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            raise ValueError(f"❌ Invalid JSON input: {x}")
    return x

def create_loan_agent(api_key: Optional[str] = None) -> AgentExecutor:
    """
    Create a LangChain agent for loan processing.
    
    Args:
        api_key: Google API key for the LLM
        
    Returns:
        AgentExecutor: The configured loan processing agent
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    
    # Define tools (always wrap input in dict with safe_parse_input)
    tools = [
        Tool(
            name="AnalyzeLoanApplication",
            func=lambda x: analyze_loan_application(safe_parse_input(x)),
            description="Analyze a loan application and return a risk assessment. Input should be a JSON dictionary with applicant data."
        ),
        Tool(
            name="GetSimilarLoanApplications",
            func=lambda x: get_similar_loan_applications(safe_parse_input(x)),
            description="Find similar loan applications in the database using vector search. Input should be a JSON dictionary with applicant data and embedding."
        ),
        Tool(
            name="CalculateLoanTerms",
            func=lambda x: calculate_loan_terms(**safe_parse_input(x)),
            description="Calculate monthly payment and total interest for a loan. Input should be a JSON dictionary with loan_amount, loan_term, and interest_rate."
        ),
        Tool(
            name="RecommendInterestRate",
            func=lambda x: recommend_interest_rate(**safe_parse_input(x)),
            description="Recommend an interest rate based on credit score and loan details. Input should be a JSON dictionary with credit_score, loan_term, and loan_amount."
        )
    ]

    # ReAct style prompt
    prompt = PromptTemplate.from_template(
        """You are a financial loan processing assistant.
You have access to the following tools:
{tools}

When answering, always follow this format:

Question: the input question
Thought: reasoning about what to do
Action: the tool name to use, one of [{tool_names}]
Action Input: the input to the tool (provide a valid JSON object without markdown formatting)
Observation: the result of the action
... (this Thought/Action/Observation can repeat)
Final Answer: the answer to the original question

Begin!

Question: {input}
{agent_scratchpad}"""
    )

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

