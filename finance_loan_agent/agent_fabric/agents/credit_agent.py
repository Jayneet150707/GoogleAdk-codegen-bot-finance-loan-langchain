"""
Credit Agent Module

This module implements the Credit Agent for the finance loan agent fabric.
The Credit Agent is responsible for assessing creditworthiness, calculating
risk scores, and making credit decisions.
"""

import logging
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import hashlib
import random

class CreditAgent:
    """
    Credit Agent for assessing creditworthiness and making credit decisions.
    
    This agent is responsible for:
    1. Evaluating credit risk
    2. Calculating credit scores
    3. Determining loan eligibility
    4. Setting loan terms and conditions
    5. Making credit decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Credit Agent.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load credit scoring model
        self.credit_model = self._load_credit_scoring_model()
        
        # Load credit policy rules
        self.credit_policy = self._load_credit_policy()
        
        self.logger.info("Credit Agent initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and assess creditworthiness.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state with credit assessment
        """
        self.logger.info("Assessing creditworthiness in Credit Agent")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract enrichment results if available
        enrichment_results = state.get("enrichment_results", {})
        
        # Extract applicant information
        applicant = application_data.get("applicant", {})
        
        # Extract loan details
        loan_details = application_data.get("loan_details", {})
        
        # Prepare features for credit scoring
        features = self._prepare_credit_features(applicant, loan_details, enrichment_results)
        
        # Calculate credit score
        credit_score = self._calculate_credit_score(features)
        state["credit_results"] = {"credit_score": credit_score}
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(features, credit_score)
        state["credit_results"]["risk_metrics"] = risk_metrics
        
        # Determine loan eligibility
        eligibility = self._determine_loan_eligibility(credit_score, risk_metrics, loan_details)
        state["credit_results"]["eligibility"] = eligibility
        
        # Set loan terms and conditions
        terms_and_conditions = self._set_loan_terms(credit_score, risk_metrics, loan_details, eligibility)
        state["credit_results"]["terms_and_conditions"] = terms_and_conditions
        
        # Make credit decision
        decision, reason = self._make_credit_decision(credit_score, risk_metrics, eligibility)
        state["credit_results"]["decision"] = decision
        state["credit_results"]["reason"] = reason
        
        # Add credit metadata
        state["credit_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "credit_agent_version": "1.0.0",
            "credit_model_version": "1.0.0",
            "credit_policy_version": "1.0.0"
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "Credit",
            "timestamp": datetime.now().isoformat(),
            "action": "Assessed creditworthiness",
            "details": {
                "credit_score": credit_score,
                "decision": decision,
                "reason": reason
            }
        })
        
        self.logger.info(f"Credit Agent processing complete with decision: {decision}")
        return state
    
    def _prepare_credit_features(
        self,
        applicant: Dict[str, Any],
        loan_details: Dict[str, Any],
        enrichment_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare features for credit scoring.
        
        Args:
            applicant: Applicant information
            loan_details: Loan details
            enrichment_results: Enrichment results
            
        Returns:
            Dict[str, Any]: Credit features
        """
        features = {}
        
        # Extract credit information
        credit_info = applicant.get("credit", {})
        credit_score = credit_info.get("credit_score", 0)
        credit_utilization = credit_info.get("credit_utilization", 0)
        
        # Extract financial information
        financial_info = applicant.get("financial_information", {})
        annual_income = financial_info.get("annual_income", 0)
        
        # Extract employment information
        employment_info = applicant.get("employment", {})
        employment_status = employment_info.get("verification_status", "unknown")
        
        # Extract loan information
        loan_amount = loan_details.get("amount", 0)
        loan_term = loan_details.get("term", 0)
        loan_purpose = loan_details.get("purpose", "")
        
        # Extract derived metrics if available
        derived_metrics = loan_details.get("derived_metrics", {})
        dti_ratio = derived_metrics.get("debt_to_income_ratio", 0)
        ltv_ratio = derived_metrics.get("loan_to_value_ratio", 0)
        
        # Set features
        features["credit_score"] = credit_score
        features["credit_utilization"] = credit_utilization
        features["annual_income"] = annual_income
        features["employment_verified"] = 1 if employment_status == "verified" else 0
        features["loan_amount"] = loan_amount
        features["loan_term"] = loan_term
        features["loan_purpose"] = loan_purpose
        features["debt_to_income_ratio"] = dti_ratio
        features["loan_to_value_ratio"] = ltv_ratio
        
        # Extract additional features from enrichment results if available
        if "credit_bureau_data" in enrichment_results:
            credit_bureau_data = enrichment_results["credit_bureau_data"]
            if credit_bureau_data.get("status") in ["found", "generated"] and "credit_data" in credit_bureau_data:
                credit_data = credit_bureau_data["credit_data"]
                
                # Add credit bureau features
                features["inquiries_last_6_months"] = credit_data.get("inquiries_last_6_months", 0)
                features["derogatory_marks"] = credit_data.get("derogatory_marks", 0)
                
                # Add payment history features
                payment_history = credit_data.get("payment_history", {})
                features["on_time_payments_percentage"] = payment_history.get("on_time_payments_percentage", 0)
                features["late_payments_30_days"] = payment_history.get("late_payments_30_days", 0)
                features["late_payments_60_days"] = payment_history.get("late_payments_60_days", 0)
                features["late_payments_90_days"] = payment_history.get("late_payments_90_days", 0)
        
        return features
    
    def _calculate_credit_score(self, features: Dict[str, Any]) -> int:
        """
        Calculate credit score based on features.
        
        Args:
            features: Credit features
            
        Returns:
            int: Credit score
        """
        # In a real system, this would use a trained machine learning model
        # For demonstration, we'll use a simplified scoring approach
        
        # Start with base score
        base_score = 600
        
        # Adjust based on credit bureau score if available
        if "credit_score" in features and features["credit_score"] > 0:
            # Weight the existing credit score heavily
            base_score = features["credit_score"]
        
        # Apply adjustments based on features
        adjustments = 0
        
        # Income adjustment (higher income = higher score)
        if "annual_income" in features:
            income = features["annual_income"]
            if income > 100000:
                adjustments += 30
            elif income > 75000:
                adjustments += 20
            elif income > 50000:
                adjustments += 10
            elif income > 30000:
                adjustments += 0
            else:
                adjustments -= 10
        
        # DTI ratio adjustment (lower DTI = higher score)
        if "debt_to_income_ratio" in features:
            dti = features["debt_to_income_ratio"]
            if dti < 0.2:
                adjustments += 30
            elif dti < 0.3:
                adjustments += 20
            elif dti < 0.4:
                adjustments += 10
            elif dti < 0.5:
                adjustments += 0
            else:
                adjustments -= 20
        
        # LTV ratio adjustment (lower LTV = higher score)
        if "loan_to_value_ratio" in features:
            ltv = features["loan_to_value_ratio"]
            if ltv < 0.5:
                adjustments += 20
            elif ltv < 0.7:
                adjustments += 10
            elif ltv < 0.8:
                adjustments += 0
            elif ltv < 0.9:
                adjustments -= 10
            else:
                adjustments -= 20
        
        # Credit utilization adjustment (lower utilization = higher score)
        if "credit_utilization" in features:
            utilization = features["credit_utilization"]
            if utilization < 0.1:
                adjustments += 20
            elif utilization < 0.3:
                adjustments += 10
            elif utilization < 0.5:
                adjustments += 0
            elif utilization < 0.7:
                adjustments -= 10
            else:
                adjustments -= 20
        
        # Payment history adjustment
        if "on_time_payments_percentage" in features:
            on_time = features["on_time_payments_percentage"]
            if on_time > 0.99:
                adjustments += 30
            elif on_time > 0.97:
                adjustments += 20
            elif on_time > 0.95:
                adjustments += 10
            elif on_time > 0.9:
                adjustments += 0
            else:
                adjustments -= 20
        
        # Derogatory marks adjustment
        if "derogatory_marks" in features:
            derogatory = features["derogatory_marks"]
            if derogatory == 0:
                adjustments += 0
            elif derogatory == 1:
                adjustments -= 30
            elif derogatory == 2:
                adjustments -= 60
            else:
                adjustments -= 100
        
        # Recent inquiries adjustment
        if "inquiries_last_6_months" in features:
            inquiries = features["inquiries_last_6_months"]
            if inquiries == 0:
                adjustments += 10
            elif inquiries == 1:
                adjustments += 0
            elif inquiries == 2:
                adjustments -= 5
            elif inquiries == 3:
                adjustments -= 10
            else:
                adjustments -= 20
        
        # Employment verification adjustment
        if "employment_verified" in features:
            if features["employment_verified"] == 1:
                adjustments += 10
            else:
                adjustments -= 10
        
        # Calculate final score
        final_score = base_score + adjustments
        
        # Ensure score is within valid range (300-850)
        final_score = max(300, min(850, final_score))
        
        return final_score
    
    def _calculate_risk_metrics(self, features: Dict[str, Any], credit_score: int) -> Dict[str, Any]:
        """
        Calculate risk metrics based on features and credit score.
        
        Args:
            features: Credit features
            credit_score: Calculated credit score
            
        Returns:
            Dict[str, Any]: Risk metrics
        """
        risk_metrics = {}
        
        # Calculate probability of default (PD)
        # In a real system, this would use a sophisticated model
        # For demonstration, we'll use a simplified approach
        if credit_score >= 800:
            pd = random.uniform(0.001, 0.005)
        elif credit_score >= 750:
            pd = random.uniform(0.005, 0.01)
        elif credit_score >= 700:
            pd = random.uniform(0.01, 0.02)
        elif credit_score >= 650:
            pd = random.uniform(0.02, 0.05)
        elif credit_score >= 600:
            pd = random.uniform(0.05, 0.1)
        elif credit_score >= 550:
            pd = random.uniform(0.1, 0.15)
        elif credit_score >= 500:
            pd = random.uniform(0.15, 0.25)
        else:
            pd = random.uniform(0.25, 0.4)
        
        risk_metrics["probability_of_default"] = round(pd, 4)
        
        # Calculate loss given default (LGD)
        # This is typically based on collateral value and loan amount
        ltv = features.get("loan_to_value_ratio", 0)
        
        if ltv > 0:
            # Higher LTV means higher loss given default
            lgd = min(1.0, ltv * 1.2)  # Add a buffer for foreclosure costs
        else:
            # Default LGD for unsecured loans
            lgd = 0.6
        
        risk_metrics["loss_given_default"] = round(lgd, 4)
        
        # Calculate exposure at default (EAD)
        # For simplicity, we'll use the loan amount
        loan_amount = features.get("loan_amount", 0)
        risk_metrics["exposure_at_default"] = loan_amount
        
        # Calculate expected loss (EL)
        el = pd * lgd * loan_amount
        risk_metrics["expected_loss"] = round(el, 2)
        
        # Calculate risk grade
        if pd < 0.01:
            risk_grade = "A"
        elif pd < 0.02:
            risk_grade = "B"
        elif pd < 0.05:
            risk_grade = "C"
        elif pd < 0.1:
            risk_grade = "D"
        elif pd < 0.2:
            risk_grade = "E"
        else:
            risk_grade = "F"
        
        risk_metrics["risk_grade"] = risk_grade
        
        return risk_metrics
    
    def _determine_loan_eligibility(
        self,
        credit_score: int,
        risk_metrics: Dict[str, Any],
        loan_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine loan eligibility based on credit score and risk metrics.
        
        Args:
            credit_score: Calculated credit score
            risk_metrics: Risk metrics
            loan_details: Loan details
            
        Returns:
            Dict[str, Any]: Loan eligibility
        """
        # Extract loan details
        loan_amount = loan_details.get("amount", 0)
        loan_term = loan_details.get("term", 0)
        
        # Extract derived metrics
        derived_metrics = loan_details.get("derived_metrics", {})
        dti_ratio = derived_metrics.get("debt_to_income_ratio", 0)
        ltv_ratio = derived_metrics.get("loan_to_value_ratio", 0)
        
        # Extract risk metrics
        pd = risk_metrics.get("probability_of_default", 0)
        risk_grade = risk_metrics.get("risk_grade", "")
        
        # Check eligibility criteria
        eligibility_checks = []
        
        # Credit score check
        min_credit_score = self.credit_policy.get("min_credit_score", 620)
        if credit_score >= min_credit_score:
            eligibility_checks.append({"check": "credit_score", "result": "pass"})
        else:
            eligibility_checks.append({
                "check": "credit_score",
                "result": "fail",
                "reason": f"Credit score {credit_score} below minimum required {min_credit_score}"
            })
        
        # DTI ratio check
        max_dti_ratio = self.credit_policy.get("max_dti_ratio", 0.43)
        if dti_ratio <= max_dti_ratio:
            eligibility_checks.append({"check": "dti_ratio", "result": "pass"})
        else:
            eligibility_checks.append({
                "check": "dti_ratio",
                "result": "fail",
                "reason": f"DTI ratio {dti_ratio:.2f} exceeds maximum allowed {max_dti_ratio:.2f}"
            })
        
        # LTV ratio check
        max_ltv_ratio = self.credit_policy.get("max_ltv_ratio", 0.8)
        if ltv_ratio <= max_ltv_ratio:
            eligibility_checks.append({"check": "ltv_ratio", "result": "pass"})
        elif ltv_ratio > 0:  # Only check if LTV is available
            eligibility_checks.append({
                "check": "ltv_ratio",
                "result": "fail",
                "reason": f"LTV ratio {ltv_ratio:.2f} exceeds maximum allowed {max_ltv_ratio:.2f}"
            })
        
        # Probability of default check
        max_pd = self.credit_policy.get("max_probability_of_default", 0.1)
        if pd <= max_pd:
            eligibility_checks.append({"check": "probability_of_default", "result": "pass"})
        else:
            eligibility_checks.append({
                "check": "probability_of_default",
                "result": "fail",
                "reason": f"Probability of default {pd:.4f} exceeds maximum allowed {max_pd:.4f}"
            })
        
        # Risk grade check
        acceptable_risk_grades = self.credit_policy.get("acceptable_risk_grades", ["A", "B", "C", "D"])
        if risk_grade in acceptable_risk_grades:
            eligibility_checks.append({"check": "risk_grade", "result": "pass"})
        else:
            eligibility_checks.append({
                "check": "risk_grade",
                "result": "fail",
                "reason": f"Risk grade {risk_grade} not in acceptable grades {acceptable_risk_grades}"
            })
        
        # Determine overall eligibility
        failed_checks = [check for check in eligibility_checks if check["result"] == "fail"]
        
        if not failed_checks:
            eligibility_status = "eligible"
            eligibility_reason = "All eligibility criteria met"
        else:
            eligibility_status = "ineligible"
            eligibility_reason = "Failed eligibility checks: " + ", ".join([check["check"] for check in failed_checks])
        
        return {
            "status": eligibility_status,
            "reason": eligibility_reason,
            "checks": eligibility_checks
        }
    
    def _set_loan_terms(
        self,
        credit_score: int,
        risk_metrics: Dict[str, Any],
        loan_details: Dict[str, Any],
        eligibility: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set loan terms and conditions based on credit assessment.
        
        Args:
            credit_score: Calculated credit score
            risk_metrics: Risk metrics
            loan_details: Loan details
            eligibility: Loan eligibility
            
        Returns:
            Dict[str, Any]: Loan terms and conditions
        """
        # If ineligible, return empty terms
        if eligibility["status"] == "ineligible":
            return {
                "status": "not_available",
                "reason": eligibility["reason"]
            }
        
        # Extract loan details
        loan_amount = loan_details.get("amount", 0)
        loan_term = loan_details.get("term", 0)
        loan_purpose = loan_details.get("purpose", "").lower()
        
        # Extract risk metrics
        risk_grade = risk_metrics.get("risk_grade", "")
        
        # Determine loan type
        if "home" in loan_purpose or "house" in loan_purpose or "mortgage" in loan_purpose:
            loan_type = "mortgage"
        elif "car" in loan_purpose or "auto" in loan_purpose or "vehicle" in loan_purpose:
            loan_type = "auto"
        else:
            loan_type = "personal"
        
        # Set base interest rate based on loan type
        if loan_type == "mortgage":
            base_rate = 0.045  # 4.5%
        elif loan_type == "auto":
            base_rate = 0.055  # 5.5%
        else:
            base_rate = 0.095  # 9.5%
        
        # Adjust rate based on credit score
        if credit_score >= 800:
            rate_adjustment = -0.01  # -1.0%
        elif credit_score >= 750:
            rate_adjustment = -0.0075  # -0.75%
        elif credit_score >= 700:
            rate_adjustment = -0.005  # -0.5%
        elif credit_score >= 650:
            rate_adjustment = -0.0025  # -0.25%
        elif credit_score >= 600:
            rate_adjustment = 0  # 0%
        elif credit_score >= 550:
            rate_adjustment = 0.01  # +1.0%
        elif credit_score >= 500:
            rate_adjustment = 0.02  # +2.0%
        else:
            rate_adjustment = 0.03  # +3.0%
        
        # Adjust rate based on risk grade
        if risk_grade == "A":
            grade_adjustment = -0.005  # -0.5%
        elif risk_grade == "B":
            grade_adjustment = -0.0025  # -0.25%
        elif risk_grade == "C":
            grade_adjustment = 0  # 0%
        elif risk_grade == "D":
            grade_adjustment = 0.005  # +0.5%
        elif risk_grade == "E":
            grade_adjustment = 0.01  # +1.0%
        else:
            grade_adjustment = 0.015  # +1.5%
        
        # Adjust rate based on loan term
        if loan_term <= 60:  # 5 years
            term_adjustment = -0.0025  # -0.25%
        elif loan_term <= 120:  # 10 years
            term_adjustment = 0  # 0%
        elif loan_term <= 180:  # 15 years
            term_adjustment = 0.0025  # +0.25%
        elif loan_term <= 240:  # 20 years
            term_adjustment = 0.005  # +0.5%
        elif loan_term <= 360:  # 30 years
            term_adjustment = 0.0075  # +0.75%
        else:
            term_adjustment = 0.01  # +1.0%
        
        # Calculate final interest rate
        interest_rate = base_rate + rate_adjustment + grade_adjustment + term_adjustment
        
        # Ensure rate is within reasonable bounds
        interest_rate = max(0.01, min(0.25, interest_rate))
        
        # Calculate monthly payment
        monthly_rate = interest_rate / 12
        num_payments = loan_term
        
        if monthly_rate > 0 and num_payments > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        else:
            monthly_payment = loan_amount / num_payments if num_payments > 0 else 0
        
        # Calculate total interest
        total_interest = monthly_payment * num_payments - loan_amount
        
        # Calculate APR (simplified)
        apr = interest_rate  # In a real system, this would include fees
        
        # Set loan terms
        terms = {
            "status": "available",
            "loan_type": loan_type,
            "interest_rate": round(interest_rate * 100, 2),  # Convert to percentage
            "apr": round(apr * 100, 2),  # Convert to percentage
            "term_months": loan_term,
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
            "total_repayment": round(loan_amount + total_interest, 2),
            "origination_fee": round(loan_amount * 0.01, 2),  # 1% origination fee
            "other_fees": 0,
            "prepayment_penalty": False
        }
        
        return terms
    
    def _make_credit_decision(
        self,
        credit_score: int,
        risk_metrics: Dict[str, Any],
        eligibility: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Make credit decision based on credit assessment.
        
        Args:
            credit_score: Calculated credit score
            risk_metrics: Risk metrics
            eligibility: Loan eligibility
            
        Returns:
            Tuple[str, str]: Decision and reason
        """
        # If ineligible, reject the application
        if eligibility["status"] == "ineligible":
            return "rejected", eligibility["reason"]
        
        # Extract risk metrics
        pd = risk_metrics.get("probability_of_default", 0)
        risk_grade = risk_metrics.get("risk_grade", "")
        
        # Check automatic approval criteria
        if credit_score >= 720 and pd < 0.02 and risk_grade in ["A", "B"]:
            return "approved", "Strong credit profile meets all approval criteria"
        
        # Check automatic rejection criteria
        if credit_score < 580 or pd > 0.15 or risk_grade in ["F"]:
            return "rejected", f"Credit profile does not meet minimum requirements (Score: {credit_score}, Risk Grade: {risk_grade})"
        
        # For borderline cases, refer to manual review
        return "manual_review", f"Borderline case requires manual review (Score: {credit_score}, Risk Grade: {risk_grade})"
    
    def _load_credit_scoring_model(self) -> Any:
        """
        Load the credit scoring model.
        
        Returns:
            Any: Credit scoring model
        """
        # In a real system, this would load a trained machine learning model
        # For demonstration, we'll return a placeholder
        return "credit_scoring_model_placeholder"
    
    def _load_credit_policy(self) -> Dict[str, Any]:
        """
        Load credit policy rules.
        
        Returns:
            Dict[str, Any]: Credit policy rules
        """
        # In a real system, these would be loaded from a database or configuration file
        return {
            "min_credit_score": 620,
            "max_dti_ratio": 0.43,
            "max_ltv_ratio": 0.8,
            "max_probability_of_default": 0.1,
            "acceptable_risk_grades": ["A", "B", "C", "D"],
            "auto_approval_criteria": {
                "min_credit_score": 720,
                "max_dti_ratio": 0.36,
                "max_ltv_ratio": 0.7,
                "max_probability_of_default": 0.02,
                "acceptable_risk_grades": ["A", "B"]
            },
            "auto_rejection_criteria": {
                "max_credit_score": 580,
                "min_dti_ratio": 0.5,
                "min_ltv_ratio": 0.9,
                "min_probability_of_default": 0.15,
                "unacceptable_risk_grades": ["F"]
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the agent
    agent = CreditAgent()
    
    # Example state with application data and enrichment results
    state = {
        "application_data": {
            "applicant": {
                "name": "John Smith",
                "contact_info": {
                    "email": "john.smith@example.com",
                    "phone": "+1234567890"
                },
                "credit": {
                    "credit_score": 720,
                    "credit_utilization": 0.25
                },
                "financial_information": {
                    "annual_income": 85000
                },
                "employment": {
                    "verification_status": "verified"
                }
            },
            "loan_details": {
                "amount": 250000,
                "term": 360,
                "purpose": "Home purchase",
                "derived_metrics": {
                    "debt_to_income_ratio": 0.32,
                    "loan_to_value_ratio": 0.75
                }
            }
        },
        "enrichment_results": {
            "credit_bureau_data": {
                "status": "found",
                "credit_data": {
                    "inquiries_last_6_months": 2,
                    "derogatory_marks": 0,
                    "payment_history": {
                        "on_time_payments_percentage": 0.98,
                        "late_payments_30_days": 1,
                        "late_payments_60_days": 0,
                        "late_payments_90_days": 0
                    }
                }
            }
        }
    }
    
    # Process the state
    updated_state = agent.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

