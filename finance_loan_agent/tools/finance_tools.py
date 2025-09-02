import pandas as pd
import numpy as np
from models.credit_scoring import CreditScoringModel
from models.risk_assessment import RiskAssessmentModel
from .mongodb_tools import get_collection

 

def analyze_loan_application(applicant_data):
    """Analyze a loan application and return a risk assessment."""
    # Convert applicant data to DataFrame
    df = pd.DataFrame([applicant_data])
    
    # Load the credit scoring model
    model = CreditScoringModel()
    try:
        model.load_model('models/credit_scoring_model.h5')
    except:
        # If model doesn't exist, return a default assessment
        return {
            "risk_score": 0.5,
            "risk_level": "Medium Risk",
            "recommendation": "Review Manually",
            "explanation": "Model not trained yet. This is a default assessment."
        }
    
    # Make prediction
    risk_score = model.predict(df)[0][0]
    
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

def generate_explanation(applicant_data, risk_score):
    """Generate an explanation for the risk assessment."""
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

def get_similar_loan_applications(applicant_data, limit=5):
    """Find similar loan applications in the database using vector search."""
    try:
        # Get the collection
        collection = get_collection("loan_applications")
        
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
        return {"error": str(e), "message": "Could not perform vector search. Database may not be configured correctly."}

def calculate_loan_terms(loan_amount, loan_term, interest_rate):
    """Calculate monthly payment and total interest for a loan."""
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

def recommend_interest_rate(credit_score, loan_term, loan_amount):
    """Recommend an interest rate based on credit score and loan details."""
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

