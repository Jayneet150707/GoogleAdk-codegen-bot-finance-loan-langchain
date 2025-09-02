# from langchain.agents import Tool, AgentExecutor, create_react_agent
# from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from tools.finance_tools import(
#    analyze_loan_application, 
#     get_similar_loan_applications,
#     calculate_loan_terms,
#     recommend_interest_rate
# )
# import json

# # from  .tools.finance_tools import (
# #     analyze_loan_application, 
# #     get_similar_loan_applications,
# #     calculate_loan_terms,
# #     recommend_interest_rate
# # )
# def safe_parse_input(x):
#     """Ensure input is always a dictionary."""
#     if isinstance(x, str):
#         try:
#             return json.loads(x)
#         except json.JSONDecodeError:
#             raise ValueError(f"Invalid JSON input: {x}")
#     return x
# def create_loan_agent(api_key=None):
#     """Create a LangChain agent for loan processing."""
#     # Initialize the LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         google_api_key=api_key,
#         temperature=0.2
#     )
    
#     # Define tools
#     tools = [
#         Tool(
#             name="AnalyzeLoanApplication",
#             func=analyze_loan_application,
#             description="Analyze a loan application and return a risk assessment. Input should be a dictionary with applicant data."
#         ),
#         Tool(
#             name="GetSimilarLoanApplications",
#             func=get_similar_loan_applications,
#             description="Find similar loan applications in the database using vector search. Input should be a dictionary with applicant data and embedding."
#         ),
#         Tool(
#             name="CalculateLoanTerms",
#             func=calculate_loan_terms,
#             description="Calculate monthly payment and total interest for a loan. Input should be a dictionary with loan_amount, loan_term, and interest_rate."
#         ),
#         Tool(
#             name="RecommendInterestRate",
#             func=recommend_interest_rate,
#             description="Recommend an interest rate based on credit score and loan details. Input should be a dictionary with credit_score, loan_term, and loan_amount."
#         )
#     ]

# #     tools = [
# #     Tool(
# #         name="AnalyzeLoanApplication",
# #         func=lambda x: analyze_loan_application(**safe_parse_input(x)),
# #         description="Analyze a loan application and return a risk assessment. Input should be a dictionary with applicant data."
# #     ),
# #     Tool(
# #         name="GetSimilarLoanApplications",
# #         func=lambda x: get_similar_loan_applications(**safe_parse_input(x)),
# #         description="Find similar loan applications in the database using vector search. Input should be a dictionary with applicant data and embedding."
# #     ),
# #     Tool(
# #         name="CalculateLoanTerms",
# #         func=lambda x: calculate_loan_terms(**safe_parse_input(x)),
# #         description="Calculate monthly payment and total interest for a loan. Input should be a dictionary with loan_amount, loan_term, and interest_rate."
# #     ),
# #     Tool(
# #         name="RecommendInterestRate",
# #         func=lambda x: recommend_interest_rate(**safe_parse_input(x)),
# #         description="Recommend an interest rate based on credit score and loan details. Input should be a dictionary with credit_score, loan_term, and loan_amount."
# #     )
# # ]
    
#     # # Create prompt template
#     # prompt = PromptTemplate.from_template(
#     #     """You are a helpful agent who can process loan applications, assess credit risk, and provide recommendations for loan approval.
#     #     You can analyze applicant data, find similar past applications, and explain the factors affecting the risk assessment.
        
#     #     {chat_history}
#     #     Human: {input}
#     #     {agent_scratchpad}
#     #     """
#     # )
#       # ReAct style prompt
#     prompt = PromptTemplate.from_template(
#         """You are a financial loan processing assistant.
# You have access to the following tools:
# {tools}

# When answering, always follow this format:

# Question: the input question
# Thought: reasoning about what to do
# Action: the tool name to use, one of [{tool_names}]
# Action Input: the input to the tool
# Observation: the result of the action
# ... (this Thought/Action/Observation can repeat)
# Final Answer: the answer to the original question

# Begin!

# Question: {input}
# {agent_scratchpad}"""
#     )
#     # Create the agent
#     agent = create_react_agent(llm, tools, prompt)
    
#     # Create the agent executor
#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         verbose=True,
#         handle_parsing_errors=True
#     )
    
#     return agent_executor

# # This will be initialized when needed with the API key
# loan_agent = None

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json

from tools.finance_tools import (
    analyze_loan_application, 
    get_similar_loan_applications,
    calculate_loan_terms,
    recommend_interest_rate
)

# --- Utility to safely parse inputs ---
def safe_parse_input(x):
    """Ensure input is always a dictionary."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            raise ValueError(f"❌ Invalid JSON input: {x}")
    return x
def create_loan_agent(api_key=None):
    """Create a LangChain agent for loan processing."""
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
            func=lambda x: calculate_loan_terms(safe_parse_input(x)),
            description="Calculate monthly payment and total interest for a loan. Input should be a JSON dictionary with loan_amount, loan_term, and interest_rate."
        ),
        Tool(
            name="RecommendInterestRate",
            func=lambda x: recommend_interest_rate(safe_parse_input(x)),
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
Action Input: the input to the tool
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
