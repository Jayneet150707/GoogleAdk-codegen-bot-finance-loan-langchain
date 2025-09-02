"""
Intake Agent Module

This module implements the Intake Agent for the finance loan agent fabric.
The Intake Agent is responsible for processing initial loan applications,
validating input data, and preparing it for further processing.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

class IntakeAgent:
    """
    Intake Agent for processing initial loan applications.
    
    This agent is responsible for:
    1. Validating application data completeness
    2. Standardizing data formats
    3. Assigning application IDs
    4. Initial risk flagging
    5. Preparing data for downstream agents
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Intake Agent.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Intake Agent initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state
        """
        self.logger.info("Processing application in Intake Agent")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Validate application data
        validation_results = self._validate_application(application_data)
        
        # Update state with validation results
        state["validation_results"] = validation_results
        
        # If validation failed, mark as requiring human review
        if not validation_results["is_valid"]:
            state["requires_human_review"] = True
            state["human_review_reason"] = "Validation failed in Intake Agent"
            self.logger.warning(f"Application validation failed: {validation_results['errors']}")
        
        # Standardize data formats
        standardized_data = self._standardize_data(application_data)
        state["application_data"] = standardized_data
        
        # Assign application ID if not present
        if "application_id" not in standardized_data:
            state["application_data"]["application_id"] = self._generate_application_id()
        
        # Perform initial risk assessment
        risk_flags = self._initial_risk_assessment(standardized_data)
        state["risk_flags"] = risk_flags
        
        # Add intake metadata
        state["intake_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "intake_agent_version": "1.0.0",
            "initial_risk_level": "high" if risk_flags else "standard"
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "Intake",
            "timestamp": datetime.now().isoformat(),
            "action": "Processed application",
            "details": {
                "validation_status": "valid" if validation_results["is_valid"] else "invalid",
                "risk_flags": risk_flags
            }
        })
        
        self.logger.info("Intake Agent processing complete")
        return state
    
    def _validate_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the application data for completeness and correctness.
        
        Args:
            application_data: The application data to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = [
            "applicant.name",
            "applicant.contact_info",
            "loan_details.amount",
            "loan_details.term"
        ]
        
        for field in required_fields:
            parts = field.split(".")
            current = application_data
            valid = True
            
            for part in parts:
                if part not in current:
                    errors.append(f"Missing required field: {field}")
                    valid = False
                    break
                current = current[part]
            
            if valid and current is None:
                errors.append(f"Required field {field} cannot be null")
        
        # Validate loan amount
        if "loan_details" in application_data and "amount" in application_data["loan_details"]:
            amount = application_data["loan_details"]["amount"]
            if not isinstance(amount, (int, float)) or amount <= 0:
                errors.append("Loan amount must be a positive number")
            elif amount > 1000000:  # Example threshold
                warnings.append("Loan amount exceeds standard threshold")
        
        # Validate loan term
        if "loan_details" in application_data and "term" in application_data["loan_details"]:
            term = application_data["loan_details"]["term"]
            if not isinstance(term, int) or term <= 0:
                errors.append("Loan term must be a positive integer")
            elif term > 360:  # 30 years in months
                warnings.append("Loan term exceeds maximum allowed (360 months)")
        
        # Validate contact information
        if "applicant" in application_data and "contact_info" in application_data["applicant"]:
            contact_info = application_data["applicant"]["contact_info"]
            
            # Check email if present
            if "email" in contact_info:
                email = contact_info["email"]
                if not self._is_valid_email(email):
                    errors.append("Invalid email format")
            
            # Check phone if present
            if "phone" in contact_info:
                phone = contact_info["phone"]
                if not self._is_valid_phone(phone):
                    errors.append("Invalid phone number format")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _standardize_data(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize data formats for consistency.
        
        Args:
            application_data: The application data to standardize
            
        Returns:
            Dict[str, Any]: Standardized application data
        """
        # Create a copy to avoid modifying the original
        standardized = application_data.copy()
        
        # Standardize applicant name (capitalize)
        if "applicant" in standardized and "name" in standardized["applicant"]:
            name = standardized["applicant"]["name"]
            if isinstance(name, str):
                standardized["applicant"]["name"] = name.title()
        
        # Standardize contact information
        if "applicant" in standardized and "contact_info" in standardized["applicant"]:
            contact_info = standardized["applicant"]["contact_info"]
            
            # Standardize email (lowercase)
            if "email" in contact_info:
                email = contact_info["email"]
                if isinstance(email, str):
                    standardized["applicant"]["contact_info"]["email"] = email.lower()
            
            # Standardize phone (E.164 format)
            if "phone" in contact_info:
                phone = contact_info["phone"]
                if isinstance(phone, str):
                    # Simple E.164 conversion (just an example)
                    phone = re.sub(r'[^0-9+]', '', phone)
                    if phone and not phone.startswith('+'):
                        phone = '+' + phone
                    standardized["applicant"]["contact_info"]["phone"] = phone
        
        # Standardize loan amount (ensure float)
        if "loan_details" in standardized and "amount" in standardized["loan_details"]:
            amount = standardized["loan_details"]["amount"]
            if isinstance(amount, (int, float, str)):
                try:
                    standardized["loan_details"]["amount"] = float(amount)
                except ValueError:
                    pass  # Keep original if conversion fails
        
        # Standardize loan term (ensure int)
        if "loan_details" in standardized and "term" in standardized["loan_details"]:
            term = standardized["loan_details"]["term"]
            if isinstance(term, (int, float, str)):
                try:
                    standardized["loan_details"]["term"] = int(float(term))
                except ValueError:
                    pass  # Keep original if conversion fails
        
        # Standardize dates (ISO format)
        if "submission_time" in standardized:
            submission_time = standardized["submission_time"]
            if isinstance(submission_time, str):
                try:
                    # Parse and reformat to ensure ISO format
                    dt = datetime.fromisoformat(submission_time.replace('Z', '+00:00'))
                    standardized["submission_time"] = dt.isoformat()
                except ValueError:
                    pass  # Keep original if conversion fails
        
        return standardized
    
    def _generate_application_id(self) -> str:
        """
        Generate a unique application ID.
        
        Returns:
            str: Unique application ID
        """
        # Generate a timestamp-based ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Add a random component
        import random
        random_component = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=6))
        
        return f"LOAN-{timestamp}-{random_component}"
    
    def _initial_risk_assessment(self, application_data: Dict[str, Any]) -> List[str]:
        """
        Perform initial risk assessment and flag potential issues.
        
        Args:
            application_data: The application data to assess
            
        Returns:
            List[str]: List of risk flags
        """
        risk_flags = []
        
        # Check loan amount thresholds
        if "loan_details" in application_data and "amount" in application_data["loan_details"]:
            amount = application_data["loan_details"]["amount"]
            if amount > 500000:
                risk_flags.append("high_value_loan")
        
        # Check loan term thresholds
        if "loan_details" in application_data and "term" in application_data["loan_details"]:
            term = application_data["loan_details"]["term"]
            if term > 240:  # 20 years in months
                risk_flags.append("extended_term_loan")
        
        # Check for missing documentation
        if "documents" in application_data:
            documents = application_data["documents"]
            required_docs = ["id_proof", "income_proof", "address_proof"]
            
            for doc in required_docs:
                if doc not in documents or not documents[doc]:
                    risk_flags.append(f"missing_{doc}")
        else:
            risk_flags.append("no_documentation")
        
        return risk_flags
    
    def _is_valid_email(self, email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Simple regex for email validation
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _is_valid_phone(self, phone: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Simple regex for phone validation (international format)
        pattern = r'^\+?[0-9]{10,15}$'
        return bool(re.match(pattern, phone))

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the agent
    agent = IntakeAgent()
    
    # Example state with application data
    state = {
        "application_data": {
            "applicant": {
                "name": "john doe",
                "contact_info": {
                    "email": "JOHN.DOE@EXAMPLE.COM",
                    "phone": "123-456-7890"
                }
            },
            "loan_details": {
                "amount": "50000",
                "term": "60",
                "purpose": "Home renovation"
            },
            "documents": {
                "id_proof": "id_proof.pdf",
                "income_proof": "income_proof.pdf"
                # Missing address_proof
            }
        }
    }
    
    # Process the state
    updated_state = agent.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

