"""
Policy Guardrails Module

This module implements policy guardrails for the finance loan agent fabric.
Policy guardrails ensure that all agent actions comply with regulatory requirements,
internal policies, and ethical guidelines.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import re

class PolicyGuardrails:
    """
    Policy Guardrails for ensuring compliance with regulations and policies.
    
    This component is responsible for:
    1. Enforcing regulatory compliance
    2. Applying internal policy rules
    3. Ensuring ethical AI behavior
    4. Preventing discriminatory practices
    5. Maintaining audit trails
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Policy Guardrails.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load policy rules
        self.policy_rules = self._load_policy_rules()
        
        # Load regulatory requirements
        self.regulatory_requirements = self._load_regulatory_requirements()
        
        # Load ethical guidelines
        self.ethical_guidelines = self._load_ethical_guidelines()
        
        self.logger.info("Policy Guardrails initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and apply policy guardrails.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state with policy decisions
        """
        self.logger.info("Applying policy guardrails")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract results from previous agents
        kyc_results = state.get("kyc_results", {})
        credit_results = state.get("credit_results", {})
        fraud_results = state.get("fraud_results", {})
        
        # Check regulatory compliance
        regulatory_compliance = self._check_regulatory_compliance(
            application_data, kyc_results, credit_results, fraud_results
        )
        state["policy_results"] = {"regulatory_compliance": regulatory_compliance}
        
        # Check internal policy compliance
        policy_compliance = self._check_policy_compliance(
            application_data, kyc_results, credit_results, fraud_results
        )
        state["policy_results"]["policy_compliance"] = policy_compliance
        
        # Check ethical compliance
        ethical_compliance = self._check_ethical_compliance(
            application_data, kyc_results, credit_results, fraud_results
        )
        state["policy_results"]["ethical_compliance"] = ethical_compliance
        
        # Make policy decision
        decision, reason = self._make_policy_decision(
            regulatory_compliance,
            policy_compliance,
            ethical_compliance
        )
        state["policy_results"]["decision"] = decision
        state["policy_results"]["reason"] = reason
        
        # Set the policy decision for the workflow
        state["policy_decision"] = decision
        
        # Add policy metadata
        state["policy_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "policy_version": "1.0.0",
            "decision": decision,
            "reason": reason
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "PolicyGuardrails",
            "timestamp": datetime.now().isoformat(),
            "action": "Applied policy guardrails",
            "details": {
                "decision": decision,
                "reason": reason
            }
        })
        
        self.logger.info(f"Policy guardrails applied with decision: {decision}")
        return state
    
    def _check_regulatory_compliance(
        self,
        application_data: Dict[str, Any],
        kyc_results: Dict[str, Any],
        credit_results: Dict[str, Any],
        fraud_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check compliance with regulatory requirements.
        
        Args:
            application_data: Application data
            kyc_results: KYC verification results
            credit_results: Credit assessment results
            fraud_results: Fraud detection results
            
        Returns:
            Dict[str, Any]: Regulatory compliance results
        """
        # Extract relevant data
        loan_details = application_data.get("loan_details", {})
        loan_amount = loan_details.get("amount", 0)
        loan_term = loan_details.get("term", 0)
        
        # Check for violations
        violations = []
        warnings = []
        
        # Check KYC compliance
        kyc_status = kyc_results.get("status", "")
        if kyc_status == "failed":
            violations.append("kyc_verification_failed")
        elif kyc_status == "review":
            warnings.append("kyc_verification_needs_review")
        
        # Check for high-risk flags in fraud results
        if fraud_results:
            fraud_risk = fraud_results.get("risk_level", "")
            if fraud_risk == "high":
                violations.append("high_fraud_risk")
            elif fraud_risk == "medium":
                warnings.append("medium_fraud_risk")
        
        # Check loan amount limits
        if loan_amount > self.regulatory_requirements.get("max_loan_amount", 1000000):
            violations.append("exceeds_max_loan_amount")
        
        # Check loan term limits
        if loan_term > self.regulatory_requirements.get("max_loan_term", 360):
            violations.append("exceeds_max_loan_term")
        
        # Determine compliance status
        if violations:
            status = "non_compliant"
        elif warnings:
            status = "conditionally_compliant"
        else:
            status = "compliant"
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "checked_at": datetime.now().isoformat()
        }
    
    def _check_policy_compliance(
        self,
        application_data: Dict[str, Any],
        kyc_results: Dict[str, Any],
        credit_results: Dict[str, Any],
        fraud_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check compliance with internal policy rules.
        
        Args:
            application_data: Application data
            kyc_results: KYC verification results
            credit_results: Credit assessment results
            fraud_results: Fraud detection results
            
        Returns:
            Dict[str, Any]: Policy compliance results
        """
        # Extract relevant data
        loan_details = application_data.get("loan_details", {})
        loan_amount = loan_details.get("amount", 0)
        loan_purpose = loan_details.get("purpose", "")
        
        # Check for violations
        violations = []
        warnings = []
        
        # Check credit score requirements
        if credit_results:
            credit_score = credit_results.get("credit_score", 0)
            min_credit_score = self.policy_rules.get("min_credit_score", 600)
            
            if credit_score < min_credit_score:
                violations.append("below_min_credit_score")
        
        # Check debt-to-income ratio
        if credit_results:
            dti_ratio = credit_results.get("debt_to_income_ratio", 0)
            max_dti_ratio = self.policy_rules.get("max_dti_ratio", 0.43)
            
            if dti_ratio > max_dti_ratio:
                violations.append("exceeds_max_dti_ratio")
        
        # Check loan purpose restrictions
        restricted_purposes = self.policy_rules.get("restricted_purposes", [])
        if loan_purpose.lower() in [p.lower() for p in restricted_purposes]:
            violations.append("restricted_loan_purpose")
        
        # Check for high-risk applications
        if "risk_flags" in application_data and application_data["risk_flags"]:
            warnings.append("application_has_risk_flags")
        
        # Determine compliance status
        if violations:
            status = "non_compliant"
        elif warnings:
            status = "conditionally_compliant"
        else:
            status = "compliant"
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "checked_at": datetime.now().isoformat()
        }
    
    def _check_ethical_compliance(
        self,
        application_data: Dict[str, Any],
        kyc_results: Dict[str, Any],
        credit_results: Dict[str, Any],
        fraud_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check compliance with ethical guidelines.
        
        Args:
            application_data: Application data
            kyc_results: KYC verification results
            credit_results: Credit assessment results
            fraud_results: Fraud detection results
            
        Returns:
            Dict[str, Any]: Ethical compliance results
        """
        # Check for violations
        violations = []
        warnings = []
        
        # Check for potential discrimination
        if self._check_for_discrimination(application_data):
            violations.append("potential_discrimination")
        
        # Check for predatory lending
        if self._check_for_predatory_lending(application_data, credit_results):
            violations.append("potential_predatory_lending")
        
        # Check for transparency issues
        if self._check_for_transparency_issues(application_data):
            warnings.append("transparency_issues")
        
        # Determine compliance status
        if violations:
            status = "non_compliant"
        elif warnings:
            status = "conditionally_compliant"
        else:
            status = "compliant"
        
        return {
            "status": status,
            "violations": violations,
            "warnings": warnings,
            "checked_at": datetime.now().isoformat()
        }
    
    def _make_policy_decision(
        self,
        regulatory_compliance: Dict[str, Any],
        policy_compliance: Dict[str, Any],
        ethical_compliance: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Make a policy decision based on compliance results.
        
        Args:
            regulatory_compliance: Regulatory compliance results
            policy_compliance: Policy compliance results
            ethical_compliance: Ethical compliance results
            
        Returns:
            Tuple[str, str]: Decision and reason
        """
        # Extract compliance statuses
        regulatory_status = regulatory_compliance.get("status", "")
        policy_status = policy_compliance.get("status", "")
        ethical_status = ethical_compliance.get("status", "")
        
        # Collect all violations
        all_violations = []
        all_violations.extend(regulatory_compliance.get("violations", []))
        all_violations.extend(policy_compliance.get("violations", []))
        all_violations.extend(ethical_compliance.get("violations", []))
        
        # Collect all warnings
        all_warnings = []
        all_warnings.extend(regulatory_compliance.get("warnings", []))
        all_warnings.extend(policy_compliance.get("warnings", []))
        all_warnings.extend(ethical_compliance.get("warnings", []))
        
        # Make decision
        if regulatory_status == "non_compliant":
            return "rejected", f"Regulatory compliance violations: {', '.join(regulatory_compliance.get('violations', []))}"
        
        if ethical_status == "non_compliant":
            return "rejected", f"Ethical compliance violations: {', '.join(ethical_compliance.get('violations', []))}"
        
        if policy_status == "non_compliant":
            return "rejected", f"Policy compliance violations: {', '.join(policy_compliance.get('violations', []))}"
        
        if all_warnings:
            return "human_review", f"Compliance warnings require review: {', '.join(all_warnings)}"
        
        return "approved", "All compliance checks passed"
    
    def _check_for_discrimination(self, application_data: Dict[str, Any]) -> bool:
        """
        Check for potential discrimination in the application processing.
        
        Args:
            application_data: Application data
            
        Returns:
            bool: True if potential discrimination detected, False otherwise
        """
        # In a real system, this would involve sophisticated analysis
        # For demonstration, we'll return False (no discrimination detected)
        return False
    
    def _check_for_predatory_lending(
        self,
        application_data: Dict[str, Any],
        credit_results: Dict[str, Any]
    ) -> bool:
        """
        Check for potential predatory lending practices.
        
        Args:
            application_data: Application data
            credit_results: Credit assessment results
            
        Returns:
            bool: True if potential predatory lending detected, False otherwise
        """
        # Extract relevant data
        loan_details = application_data.get("loan_details", {})
        interest_rate = loan_details.get("interest_rate", 0)
        
        # Check for high interest rates for vulnerable borrowers
        if credit_results:
            credit_score = credit_results.get("credit_score", 0)
            if credit_score < 600 and interest_rate > 20:
                return True
        
        # In a real system, this would involve more sophisticated analysis
        return False
    
    def _check_for_transparency_issues(self, application_data: Dict[str, Any]) -> bool:
        """
        Check for potential transparency issues in the application.
        
        Args:
            application_data: Application data
            
        Returns:
            bool: True if potential transparency issues detected, False otherwise
        """
        # In a real system, this would involve checking for clear disclosures
        # For demonstration, we'll return False (no transparency issues detected)
        return False
    
    def _load_policy_rules(self) -> Dict[str, Any]:
        """
        Load internal policy rules.
        
        Returns:
            Dict[str, Any]: Policy rules
        """
        # In a real system, these would be loaded from a database or configuration file
        return {
            "min_credit_score": 620,
            "max_dti_ratio": 0.43,
            "min_income": 25000,
            "max_loan_to_value": 0.8,
            "restricted_purposes": [
                "gambling",
                "cryptocurrency",
                "speculative investments"
            ]
        }
    
    def _load_regulatory_requirements(self) -> Dict[str, Any]:
        """
        Load regulatory requirements.
        
        Returns:
            Dict[str, Any]: Regulatory requirements
        """
        # In a real system, these would be loaded from a database or configuration file
        return {
            "max_loan_amount": 1000000,
            "max_loan_term": 360,  # 30 years in months
            "required_disclosures": [
                "interest_rate",
                "annual_percentage_rate",
                "total_cost_of_credit"
            ],
            "kyc_requirements": [
                "identity_verification",
                "address_verification",
                "income_verification"
            ]
        }
    
    def _load_ethical_guidelines(self) -> Dict[str, Any]:
        """
        Load ethical guidelines.
        
        Returns:
            Dict[str, Any]: Ethical guidelines
        """
        # In a real system, these would be loaded from a database or configuration file
        return {
            "non_discrimination": {
                "protected_attributes": [
                    "race",
                    "color",
                    "religion",
                    "sex",
                    "national_origin",
                    "disability",
                    "familial_status"
                ]
            },
            "fair_lending": {
                "max_interest_rate_for_vulnerable": 18.0,
                "vulnerable_credit_score_threshold": 600
            },
            "transparency": {
                "required_explanations": [
                    "interest_calculation",
                    "fee_structure",
                    "prepayment_penalties"
                ]
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the policy guardrails
    policy_guardrails = PolicyGuardrails()
    
    # Example state with application data and agent results
    state = {
        "application_data": {
            "applicant": {
                "name": "John Smith",
                "contact_info": {
                    "email": "john.smith@example.com",
                    "phone": "+1234567890"
                }
            },
            "loan_details": {
                "amount": 250000,
                "term": 360,
                "purpose": "Home purchase",
                "interest_rate": 4.5
            }
        },
        "kyc_results": {
            "status": "passed",
            "risk_score": 0.2
        },
        "credit_results": {
            "credit_score": 720,
            "debt_to_income_ratio": 0.35,
            "recommendation": "approve"
        },
        "fraud_results": {
            "risk_level": "low",
            "score": 0.15
        }
    }
    
    # Process the state
    updated_state = policy_guardrails.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

