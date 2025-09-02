"""
KYC (Know Your Customer) Agent Module

This module implements the KYC Agent for the finance loan agent fabric.
The KYC Agent is responsible for identity verification, background checks,
and regulatory compliance verification.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import hashlib
import random

class KYCAgent:
    """
    KYC Agent for identity verification and compliance checks.
    
    This agent is responsible for:
    1. Identity verification
    2. Document validation
    3. Background checks
    4. Regulatory compliance verification
    5. Anti-money laundering (AML) screening
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the KYC Agent.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default risk thresholds
        self.risk_thresholds = self.config.get("risk_thresholds", {
            "low": 0.3,
            "medium": 0.7,
            "high": 0.9
        })
        
        # Mock databases for demonstration
        self._setup_mock_databases()
        
        self.logger.info("KYC Agent initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state
        """
        self.logger.info("Processing application in KYC Agent")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract applicant information
        applicant = application_data.get("applicant", {})
        documents = application_data.get("documents", {})
        
        # Perform identity verification
        identity_verification = self._verify_identity(applicant, documents)
        state["kyc_results"] = {"identity_verification": identity_verification}
        
        # Perform document validation
        document_validation = self._validate_documents(documents)
        state["kyc_results"]["document_validation"] = document_validation
        
        # Perform background check
        background_check = self._perform_background_check(applicant)
        state["kyc_results"]["background_check"] = background_check
        
        # Perform AML screening
        aml_screening = self._perform_aml_screening(applicant)
        state["kyc_results"]["aml_screening"] = aml_screening
        
        # Calculate overall KYC risk score
        kyc_risk_score = self._calculate_kyc_risk_score(
            identity_verification,
            document_validation,
            background_check,
            aml_screening
        )
        state["kyc_results"]["risk_score"] = kyc_risk_score
        
        # Determine KYC status
        kyc_status, reason = self._determine_kyc_status(kyc_risk_score)
        state["kyc_results"]["status"] = kyc_status
        state["kyc_results"]["reason"] = reason
        
        # If KYC failed, mark as requiring human review
        if kyc_status == "failed":
            state["requires_human_review"] = True
            state["human_review_reason"] = f"KYC verification failed: {reason}"
            self.logger.warning(f"KYC verification failed: {reason}")
        
        # Add KYC metadata
        state["kyc_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "kyc_agent_version": "1.0.0",
            "risk_score": kyc_risk_score,
            "status": kyc_status
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "KYC",
            "timestamp": datetime.now().isoformat(),
            "action": "Performed KYC verification",
            "details": {
                "status": kyc_status,
                "risk_score": kyc_risk_score
            }
        })
        
        self.logger.info(f"KYC Agent processing complete with status: {kyc_status}")
        return state
    
    def _verify_identity(self, applicant: Dict[str, Any], documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the applicant's identity using provided information and documents.
        
        Args:
            applicant: Applicant information
            documents: Submitted documents
            
        Returns:
            Dict[str, Any]: Identity verification results
        """
        # Extract applicant details
        name = applicant.get("name", "")
        contact_info = applicant.get("contact_info", {})
        email = contact_info.get("email", "")
        phone = contact_info.get("phone", "")
        
        # Check if ID proof document exists
        id_proof_exists = "id_proof" in documents and documents["id_proof"]
        
        # In a real system, we would:
        # 1. Extract information from ID documents using OCR
        # 2. Compare extracted information with provided applicant details
        # 3. Verify ID document authenticity
        # 4. Check for tampering or forgery
        
        # For demonstration, we'll use a mock verification
        verification_score = 0.0
        verification_flags = []
        
        # Check name against blacklist
        if self._is_in_blacklist(name):
            verification_flags.append("name_blacklisted")
            verification_score += 0.5
        
        # Check email validity
        if not self._is_valid_email(email):
            verification_flags.append("invalid_email")
            verification_score += 0.2
        
        # Check phone validity
        if not self._is_valid_phone(phone):
            verification_flags.append("invalid_phone")
            verification_score += 0.2
        
        # Check if ID proof exists
        if not id_proof_exists:
            verification_flags.append("missing_id_proof")
            verification_score += 0.4
        
        # Generate a deterministic but random-looking verification score
        # In a real system, this would be based on actual verification results
        base_score = self._generate_deterministic_score(name + email + phone, 0.2, 0.9)
        
        # Combine base score with flag-based adjustments
        final_score = min(0.99, max(0.01, base_score + verification_score))
        
        # Determine verification status
        if final_score < self.risk_thresholds["low"]:
            status = "verified"
        elif final_score < self.risk_thresholds["medium"]:
            status = "needs_review"
        else:
            status = "failed"
        
        return {
            "status": status,
            "score": final_score,
            "flags": verification_flags,
            "verified_at": datetime.now().isoformat()
        }
    
    def _validate_documents(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the submitted documents for authenticity and completeness.
        
        Args:
            documents: Submitted documents
            
        Returns:
            Dict[str, Any]: Document validation results
        """
        # Required documents
        required_docs = ["id_proof", "income_proof", "address_proof"]
        
        # Check for missing documents
        missing_docs = [doc for doc in required_docs if doc not in documents or not documents[doc]]
        
        # In a real system, we would:
        # 1. Check document format and quality
        # 2. Verify document authenticity
        # 3. Extract and validate information
        # 4. Check for tampering or forgery
        
        # For demonstration, we'll use mock validation
        validation_score = 0.0
        validation_flags = []
        
        # Penalize for missing documents
        if missing_docs:
            validation_flags.append(f"missing_documents: {', '.join(missing_docs)}")
            validation_score += 0.3 * len(missing_docs)
        
        # Generate a deterministic but random-looking validation score
        # In a real system, this would be based on actual validation results
        doc_string = json.dumps(documents)
        base_score = self._generate_deterministic_score(doc_string, 0.1, 0.8)
        
        # Combine base score with flag-based adjustments
        final_score = min(0.99, max(0.01, base_score + validation_score))
        
        # Determine validation status
        if final_score < self.risk_thresholds["low"]:
            status = "valid"
        elif final_score < self.risk_thresholds["medium"]:
            status = "needs_review"
        else:
            status = "invalid"
        
        return {
            "status": status,
            "score": final_score,
            "flags": validation_flags,
            "missing_documents": missing_docs,
            "validated_at": datetime.now().isoformat()
        }
    
    def _perform_background_check(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a background check on the applicant.
        
        Args:
            applicant: Applicant information
            
        Returns:
            Dict[str, Any]: Background check results
        """
        # Extract applicant details
        name = applicant.get("name", "")
        
        # In a real system, we would:
        # 1. Check criminal records
        # 2. Check credit history
        # 3. Check employment history
        # 4. Check education history
        
        # For demonstration, we'll use mock checks
        check_flags = []
        
        # Check if in criminal database
        if self._is_in_criminal_database(name):
            check_flags.append("criminal_record")
        
        # Check if in fraud database
        if self._is_in_fraud_database(name):
            check_flags.append("fraud_history")
        
        # Generate a deterministic but random-looking check score
        # In a real system, this would be based on actual check results
        check_score = self._generate_deterministic_score(name, 0.1, 0.9)
        
        # Adjust score based on flags
        if "criminal_record" in check_flags:
            check_score = max(0.7, check_score)
        
        if "fraud_history" in check_flags:
            check_score = max(0.8, check_score)
        
        # Determine check status
        if check_score < self.risk_thresholds["low"]:
            status = "clear"
        elif check_score < self.risk_thresholds["medium"]:
            status = "minor_issues"
        else:
            status = "major_issues"
        
        return {
            "status": status,
            "score": check_score,
            "flags": check_flags,
            "checked_at": datetime.now().isoformat()
        }
    
    def _perform_aml_screening(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform anti-money laundering (AML) screening on the applicant.
        
        Args:
            applicant: Applicant information
            
        Returns:
            Dict[str, Any]: AML screening results
        """
        # Extract applicant details
        name = applicant.get("name", "")
        
        # In a real system, we would:
        # 1. Check against sanctions lists
        # 2. Check against politically exposed persons (PEP) lists
        # 3. Check against watchlists
        # 4. Check for suspicious activity patterns
        
        # For demonstration, we'll use mock screening
        screening_flags = []
        
        # Check if in sanctions database
        if self._is_in_sanctions_database(name):
            screening_flags.append("sanctioned")
        
        # Check if in PEP database
        if self._is_in_pep_database(name):
            screening_flags.append("politically_exposed")
        
        # Generate a deterministic but random-looking screening score
        # In a real system, this would be based on actual screening results
        screening_score = self._generate_deterministic_score(name, 0.05, 0.8)
        
        # Adjust score based on flags
        if "sanctioned" in screening_flags:
            screening_score = max(0.9, screening_score)
        
        if "politically_exposed" in screening_flags:
            screening_score = max(0.7, screening_score)
        
        # Determine screening status
        if screening_score < self.risk_thresholds["low"]:
            status = "clear"
        elif screening_score < self.risk_thresholds["high"]:
            status = "review_required"
        else:
            status = "blocked"
        
        return {
            "status": status,
            "score": screening_score,
            "flags": screening_flags,
            "screened_at": datetime.now().isoformat()
        }
    
    def _calculate_kyc_risk_score(
        self,
        identity_verification: Dict[str, Any],
        document_validation: Dict[str, Any],
        background_check: Dict[str, Any],
        aml_screening: Dict[str, Any]
    ) -> float:
        """
        Calculate the overall KYC risk score.
        
        Args:
            identity_verification: Identity verification results
            document_validation: Document validation results
            background_check: Background check results
            aml_screening: AML screening results
            
        Returns:
            float: Overall KYC risk score
        """
        # Extract individual scores
        identity_score = identity_verification.get("score", 0.5)
        document_score = document_validation.get("score", 0.5)
        background_score = background_check.get("score", 0.5)
        aml_score = aml_screening.get("score", 0.5)
        
        # Calculate weighted average
        # Weights can be adjusted based on importance
        weights = {
            "identity": 0.3,
            "document": 0.2,
            "background": 0.25,
            "aml": 0.25
        }
        
        weighted_score = (
            identity_score * weights["identity"] +
            document_score * weights["document"] +
            background_score * weights["background"] +
            aml_score * weights["aml"]
        )
        
        # Round to 4 decimal places
        return round(weighted_score, 4)
    
    def _determine_kyc_status(self, risk_score: float) -> Tuple[str, str]:
        """
        Determine the overall KYC status based on the risk score.
        
        Args:
            risk_score: Overall KYC risk score
            
        Returns:
            Tuple[str, str]: Status and reason
        """
        if risk_score < self.risk_thresholds["low"]:
            return "passed", "All KYC checks passed successfully"
        elif risk_score < self.risk_thresholds["medium"]:
            return "review", "Some KYC checks require manual review"
        else:
            return "failed", "KYC checks failed due to high risk score"
    
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
    
    def _generate_deterministic_score(self, seed_string: str, min_score: float, max_score: float) -> float:
        """
        Generate a deterministic but random-looking score based on a seed string.
        
        Args:
            seed_string: Seed string for deterministic generation
            min_score: Minimum score value
            max_score: Maximum score value
            
        Returns:
            float: Generated score
        """
        # Create a hash of the seed string
        hash_obj = hashlib.md5(seed_string.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert first 8 characters of hash to integer
        hash_int = int(hash_hex[:8], 16)
        
        # Normalize to [0, 1] range
        normalized = hash_int / 0xFFFFFFFF
        
        # Scale to [min_score, max_score] range
        score = min_score + normalized * (max_score - min_score)
        
        # Round to 4 decimal places
        return round(score, 4)
    
    def _setup_mock_databases(self):
        """Set up mock databases for demonstration purposes."""
        # Blacklisted names
        self.blacklist = [
            "john doe",
            "jane doe",
            "test user",
            "known fraudster",
            "fake name"
        ]
        
        # Criminal database
        self.criminal_database = [
            "criminal person",
            "wanted individual",
            "fugitive name"
        ]
        
        # Fraud database
        self.fraud_database = [
            "known fraudster",
            "scammer name",
            "identity thief"
        ]
        
        # Sanctions database
        self.sanctions_database = [
            "sanctioned person",
            "embargoed individual",
            "restricted entity"
        ]
        
        # Politically exposed persons database
        self.pep_database = [
            "political figure",
            "government official",
            "diplomat name"
        ]
    
    def _is_in_blacklist(self, name: str) -> bool:
        """Check if a name is in the blacklist."""
        return name.lower() in self.blacklist
    
    def _is_in_criminal_database(self, name: str) -> bool:
        """Check if a name is in the criminal database."""
        return name.lower() in self.criminal_database
    
    def _is_in_fraud_database(self, name: str) -> bool:
        """Check if a name is in the fraud database."""
        return name.lower() in self.fraud_database
    
    def _is_in_sanctions_database(self, name: str) -> bool:
        """Check if a name is in the sanctions database."""
        return name.lower() in self.sanctions_database
    
    def _is_in_pep_database(self, name: str) -> bool:
        """Check if a name is in the PEP database."""
        return name.lower() in self.pep_database

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the agent
    agent = KYCAgent()
    
    # Example state with application data
    state = {
        "application_data": {
            "applicant": {
                "name": "John Smith",
                "contact_info": {
                    "email": "john.smith@example.com",
                    "phone": "+1234567890"
                }
            },
            "documents": {
                "id_proof": "id_proof.pdf",
                "income_proof": "income_proof.pdf",
                "address_proof": "address_proof.pdf"
            }
        }
    }
    
    # Process the state
    updated_state = agent.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

