import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
import sys
import warnings

# Check if we're using Python 3.13+
PY_VERSION = sys.version_info
IS_PY_313_PLUS = PY_VERSION.major == 3 and PY_VERSION.minor >= 13

# Import models with proper error handling
try:
    from ..models.credit_scoring import CreditScoringModel
    from ..models.risk_assessment import RiskAssessmentModel
except ImportError:
    # Handle relative import error when running as script
    from models.credit_scoring import CreditScoringModel
    from models.risk_assessment import RiskAssessmentModel

# Import MongoDB tools with proper error handling
try:
    from .mongodb_tools import get_collection
except ImportError:
    # Handle relative import error when running as script
    from mongodb_tools import get_collection


def analyze_loan_application(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a loan application and return a risk assessment.
    
    Args:
        applicant_data: Dictionary containing applicant information
        
    Returns:
        Dict: Risk assessment results including score, level, recommendation, and explanation
    """
    # Map input field names to expected model field names
    field_mapping = {
        'annual_income': 'income',
        'debt_to_income_ratio': 'debt_to_income',
    }
    
    # Create a new dictionary with mapped field names
    mapped_data = {}
    for key, value in applicant_data.items():
        # Use the mapped field name if available, otherwise use the original
        mapped_key = field_mapping.get(key, key)
        mapped_data[mapped_key] = value
    
    # Convert applicant data to DataFrame
    df = pd.DataFrame([mapped_data])
    
    # Load the credit scoring model
    model = CreditScoringModel()
    try:
        model.load_model('models/credit_scoring_model.h5')
    except Exception as e:
        # If model doesn't exist or can't be loaded, return a default assessment
        warnings.warn(f"Could not load credit scoring model: {str(e)}. Using default assessment.")
        return {
            "risk_score": 0.5,
            "risk_level": "Medium Risk",
            "recommendation": "Review Manually",
            "explanation": "Model not available or not trained yet. This is a default assessment."
        }
    
    # Make prediction
    try:
        risk_score = model.predict(df)[0][0]
    except Exception as e:
        warnings.warn(f"Error making prediction: {str(e)}. Using default assessment.")
        risk_score = 0.5
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = "Low Risk"
        recommendation = "Approve"
    elif risk_score < 0.7:
        risk_level = "Medium Risk"
        recommendation = "Review Manually"
    else:
        risk_level = "High Risk"
        recommendation = "Deny"
    
    return {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "explanation": generate_explanation(df, risk_score)
    }


def generate_explanation(applicant_data: pd.DataFrame, risk_score: float) -> str:
    """
    Generate an explanation for the risk assessment.
    
    Args:
        applicant_data: DataFrame containing applicant information
        risk_score: Calculated risk score
        
    Returns:
        str: Explanation of risk factors
    """
    # This would be more sophisticated in a real application
    factors = []
    
    if 'income' in applicant_data.columns and applicant_data['income'].values[0] < 30000:
        factors.append("Low income")
    
    if 'credit_score' in applicant_data.columns and applicant_data['credit_score'].values[0] < 650:
        factors.append("Below average credit score")
    
    if 'debt_to_income' in applicant_data.columns and applicant_data['debt_to_income'].values[0] > 0.4:
        factors.append("High debt-to-income ratio")
    
    if len(factors) == 0:
        return "No significant risk factors identified."
    else:
        return "Risk factors include: " + ", ".join(factors)


def get_similar_loan_applications(applicant_data: Dict[str, Any], limit: int = 5) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Find similar loan applications in the database using vector search.
    
    Args:
        applicant_data: Dictionary containing applicant information and embedding
        limit: Maximum number of similar applications to return
        
    Returns:
        List of similar loan applications or error message
    """
    try:
        # Get the collection
        collection = get_collection("loan_applications")
        
        # Verify embedding exists
        if "embedding" not in applicant_data:
            return {"error": "Missing embedding in applicant data", 
                    "message": "The applicant data must include an 'embedding' field for vector search."}
        
        # Create a pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "loan_application_index",
                    "path": "embedding",
                    "queryVector": applicant_data["embedding"],
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "income": 1,
                    "credit_score": 1,
                    "debt_to_income": 1,
                    "loan_amount": 1,
                    "loan_term": 1,
                    "interest_rate": 1,
                    "approval_status": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Execute the pipeline
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        return {
            "error": str(e), 
            "message": "Could not perform vector search. Database may not be configured correctly."
        }


def calculate_loan_terms(loan_amount: float, loan_term: int, interest_rate: float) -> Dict[str, float]:
    """
    Calculate monthly payment and total interest for a loan.
    
    Args:
        loan_amount: Principal loan amount
        loan_term: Loan term in months
        interest_rate: Annual interest rate as a percentage
        
    Returns:
        Dict: Monthly payment, total payment, and total interest
    """
    # Convert annual interest rate to monthly
    monthly_rate = interest_rate / 12 / 100
    
    # Calculate monthly payment using the loan formula
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term) / ((1 + monthly_rate) ** loan_term - 1)
    
    # Calculate total payment and interest
    total_payment = monthly_payment * loan_term
    total_interest = total_payment - loan_amount
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2)
    }


def recommend_interest_rate(credit_score: int, loan_term: int, loan_amount: float) -> float:
    """
    Recommend an interest rate based on credit score and loan details.
    
    Args:
        credit_score: Applicant's credit score
        loan_term: Loan term in months
        loan_amount: Principal loan amount
        
    Returns:
        float: Recommended interest rate as a percentage
    """
    # Base rate
    base_rate = 5.0
    
    # Adjust for credit score
    if credit_score >= 800:
        credit_adjustment = -1.5
    elif credit_score >= 750:
        credit_adjustment = -1.0
    elif credit_score >= 700:
        credit_adjustment = -0.5
    elif credit_score >= 650:
        credit_adjustment = 0.0
    elif credit_score >= 600:
        credit_adjustment = 1.0
    else:
        credit_adjustment = 2.0
    
    # Adjust for loan term
    if loan_term <= 36:
        term_adjustment = -0.25
    elif loan_term <= 60:
        term_adjustment = 0.0
    else:
        term_adjustment = 0.5
    
    # Adjust for loan amount
    if loan_amount >= 100000:
        amount_adjustment = -0.25
    elif loan_amount >= 50000:
        amount_adjustment = 0.0
    else:
        amount_adjustment = 0.25
    
    # Calculate recommended rate
    recommended_rate = base_rate + credit_adjustment + term_adjustment + amount_adjustment
    
    # Ensure rate is within reasonable bounds
    recommended_rate = max(2.0, min(recommended_rate, 15.0))
    
    return round(recommended_rate, 2)

