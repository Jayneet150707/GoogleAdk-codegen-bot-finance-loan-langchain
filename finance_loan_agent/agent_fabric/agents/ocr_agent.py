"""
OCR (Optical Character Recognition) Agent Module

This module implements the OCR Agent for the finance loan agent fabric.
The OCR Agent is responsible for extracting information from loan documents
using optical character recognition techniques.
"""

import logging
import json
import os
import base64
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import hashlib
import random

class OCRAgent:
    """
    OCR Agent for document processing and information extraction.
    
    This agent is responsible for:
    1. Processing document images and PDFs
    2. Extracting text and structured data from documents
    3. Identifying document types
    4. Validating document completeness
    5. Extracting key information for downstream processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCR Agent.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Document type definitions
        self.document_types = {
            "id_proof": {
                "required_fields": ["full_name", "id_number", "date_of_birth", "expiry_date"],
                "patterns": {
                    "id_number": r'(?i)(id|identification)\s*(?:number|#|no)?\s*[:\-]?\s*([A-Z0-9]{6,})',
                    "date_of_birth": r'(?i)(dob|date\s+of\s+birth|birth\s+date)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
                    "expiry_date": r'(?i)(expiry|expiration)\s*(?:date)?\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
                }
            },
            "income_proof": {
                "required_fields": ["employer_name", "income_amount", "period", "employee_name"],
                "patterns": {
                    "employer_name": r'(?i)employer\s*[:\-]?\s*([A-Za-z0-9\s\.,&]+)',
                    "income_amount": r'(?i)(salary|income|earnings|wage)\s*[:\-]?\s*[$£€]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    "period": r'(?i)(annually|monthly|weekly|bi-weekly|yearly|per\s+annum)'
                }
            },
            "address_proof": {
                "required_fields": ["full_name", "address", "issue_date"],
                "patterns": {
                    "address": r'(?i)(address|residence)\s*[:\-]?\s*([A-Za-z0-9\s\.,#\-]+)',
                    "issue_date": r'(?i)(issue|issued)\s*(?:date|on)?\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
                }
            },
            "bank_statement": {
                "required_fields": ["account_holder", "account_number", "bank_name", "statement_period", "balance"],
                "patterns": {
                    "account_number": r'(?i)(?:account|a/c)\s*(?:number|#|no)?\s*[:\-]?\s*([A-Z0-9]{6,})',
                    "bank_name": r'(?i)(bank|financial\s+institution)\s*[:\-]?\s*([A-Za-z\s\.,&]+)',
                    "balance": r'(?i)(balance|closing\s+balance)\s*[:\-]?\s*[$£€]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    "statement_period": r'(?i)(statement|period)\s*(?:date|period)?\s*[:\-]?\s*([A-Za-z0-9\s\.,\-\/]+)'
                }
            }
        }
        
        self.logger.info("OCR Agent initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and extract information from documents.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state with extracted document information
        """
        self.logger.info("Processing documents in OCR Agent")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract document information
        documents = application_data.get("documents", {})
        
        # Process each document
        processed_documents = {}
        extraction_results = {}
        document_validation = {}
        
        for doc_type, doc_path in documents.items():
            # Skip if document path is empty
            if not doc_path:
                continue
            
            # Process the document
            self.logger.info(f"Processing {doc_type} document: {doc_path}")
            
            # In a real system, we would load and process the actual document
            # For demonstration, we'll use mock processing
            doc_content = self._mock_document_content(doc_type, doc_path)
            
            # Extract information from the document
            extracted_info = self._extract_document_info(doc_type, doc_content)
            extraction_results[doc_type] = extracted_info
            
            # Validate the document
            validation_result = self._validate_document(doc_type, extracted_info)
            document_validation[doc_type] = validation_result
            
            # Store processed document
            processed_documents[doc_type] = {
                "path": doc_path,
                "extracted_info": extracted_info,
                "validation": validation_result
            }
        
        # Update state with processed documents
        state["ocr_results"] = {
            "processed_documents": processed_documents,
            "extraction_results": extraction_results,
            "document_validation": document_validation
        }
        
        # Check if any documents failed validation
        failed_documents = [
            doc_type for doc_type, validation in document_validation.items()
            if validation.get("status") == "invalid"
        ]
        
        if failed_documents:
            state["requires_human_review"] = True
            state["human_review_reason"] = f"Document validation failed for: {', '.join(failed_documents)}"
            self.logger.warning(f"Document validation failed for: {', '.join(failed_documents)}")
        
        # Update application data with extracted information
        self._update_application_data(state, extraction_results)
        
        # Add OCR metadata
        state["ocr_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "ocr_agent_version": "1.0.0",
            "documents_processed": list(processed_documents.keys())
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "OCR",
            "timestamp": datetime.now().isoformat(),
            "action": "Processed documents",
            "details": {
                "documents_processed": list(processed_documents.keys()),
                "failed_documents": failed_documents
            }
        })
        
        self.logger.info("OCR Agent processing complete")
        return state
    
    def _mock_document_content(self, doc_type: str, doc_path: str) -> str:
        """
        Generate mock document content for demonstration purposes.
        
        Args:
            doc_type: Type of document
            doc_path: Path to the document
            
        Returns:
            str: Mock document content
        """
        # Generate deterministic but random-looking content based on document type and path
        seed = f"{doc_type}_{doc_path}"
        random.seed(hashlib.md5(seed.encode()).hexdigest())
        
        if doc_type == "id_proof":
            full_name = "John A. Smith"
            id_number = f"ID-{random.randint(100000, 999999)}"
            dob = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(1960, 2000)}"
            expiry = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2023, 2030)}"
            
            return f"""
            IDENTIFICATION DOCUMENT
            
            Full Name: {full_name}
            ID Number: {id_number}
            Date of Birth: {dob}
            Expiry Date: {expiry}
            
            [DOCUMENT IMAGE PLACEHOLDER]
            """
        
        elif doc_type == "income_proof":
            employer = "Acme Corporation"
            employee = "John A. Smith"
            income = f"${random.randint(3000, 10000)}.00"
            period = random.choice(["Monthly", "Bi-weekly", "Annually"])
            
            return f"""
            INCOME VERIFICATION
            
            Employee Name: {employee}
            Employer: {employer}
            Income: {income} {period}
            Issue Date: {random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2022, 2023)}
            
            [DOCUMENT IMAGE PLACEHOLDER]
            """
        
        elif doc_type == "address_proof":
            full_name = "John A. Smith"
            address = f"{random.randint(100, 999)} Main St, Apt {random.randint(1, 100)}, Anytown, ST {random.randint(10000, 99999)}"
            issue_date = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2022, 2023)}"
            
            return f"""
            PROOF OF RESIDENCE
            
            Full Name: {full_name}
            Address: {address}
            Issue Date: {issue_date}
            
            [DOCUMENT IMAGE PLACEHOLDER]
            """
        
        elif doc_type == "bank_statement":
            account_holder = "John A. Smith"
            account_number = f"ACCT-{random.randint(100000, 999999)}"
            bank_name = "First National Bank"
            balance = f"${random.randint(1000, 50000)}.{random.randint(10, 99)}"
            period = f"{random.randint(1, 12)}/01/{random.randint(2022, 2023)} - {random.randint(1, 12)}/30/{random.randint(2022, 2023)}"
            
            return f"""
            BANK STATEMENT
            
            Account Holder: {account_holder}
            Account Number: {account_number}
            Bank Name: {bank_name}
            Statement Period: {period}
            Closing Balance: {balance}
            
            [TRANSACTION DETAILS PLACEHOLDER]
            """
        
        else:
            return f"Unknown document type: {doc_type}"
    
    def _extract_document_info(self, doc_type: str, doc_content: str) -> Dict[str, Any]:
        """
        Extract information from document content using OCR and pattern matching.
        
        Args:
            doc_type: Type of document
            doc_content: Document content
            
        Returns:
            Dict[str, Any]: Extracted information
        """
        # Get document type definition
        doc_def = self.document_types.get(doc_type, {})
        patterns = doc_def.get("patterns", {})
        
        # Extract information using patterns
        extracted_info = {}
        
        # Extract full name (common across document types)
        name_match = re.search(r'(?i)(?:full\s+name|name)[:\s]\s*([A-Za-z\s\.]+)', doc_content)
        if name_match:
            extracted_info["full_name"] = name_match.group(1).strip()
        
        # Extract other fields based on document type
        for field, pattern in patterns.items():
            match = re.search(pattern, doc_content)
            if match and len(match.groups()) > 0:
                extracted_info[field] = match.group(len(match.groups())).strip()
        
        # Add document type
        extracted_info["document_type"] = doc_type
        
        # Add extraction confidence (mock value)
        extracted_info["extraction_confidence"] = round(random.uniform(0.75, 0.98), 2)
        
        return extracted_info
    
    def _validate_document(self, doc_type: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the extracted document information for completeness and correctness.
        
        Args:
            doc_type: Type of document
            extracted_info: Extracted information
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Get document type definition
        doc_def = self.document_types.get(doc_type, {})
        required_fields = doc_def.get("required_fields", [])
        
        # Check for missing required fields
        missing_fields = [field for field in required_fields if field not in extracted_info]
        
        # Check extraction confidence
        confidence = extracted_info.get("extraction_confidence", 0)
        low_confidence_fields = []
        
        # In a real system, we would check confidence for each field
        # For demonstration, we'll use the overall confidence
        if confidence < 0.8:
            low_confidence_fields = ["overall_document"]
        
        # Determine validation status
        if missing_fields:
            status = "invalid"
            reason = f"Missing required fields: {', '.join(missing_fields)}"
        elif low_confidence_fields:
            status = "needs_review"
            reason = f"Low confidence extraction for: {', '.join(low_confidence_fields)}"
        else:
            status = "valid"
            reason = "All required fields extracted with high confidence"
        
        return {
            "status": status,
            "reason": reason,
            "missing_fields": missing_fields,
            "low_confidence_fields": low_confidence_fields,
            "validated_at": datetime.now().isoformat()
        }
    
    def _update_application_data(self, state: Dict[str, Any], extraction_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Update application data with extracted information from documents.
        
        Args:
            state: Current workflow state
            extraction_results: Extracted information from documents
        """
        # Get application data
        application_data = state.get("application_data", {})
        
        # Update applicant information from ID proof
        if "id_proof" in extraction_results:
            id_info = extraction_results["id_proof"]
            
            # Update applicant name if available
            if "full_name" in id_info and "applicant" in application_data:
                application_data["applicant"]["name"] = id_info["full_name"]
            
            # Add ID information
            if "applicant" in application_data:
                if "id_information" not in application_data["applicant"]:
                    application_data["applicant"]["id_information"] = {}
                
                for field in ["id_number", "date_of_birth", "expiry_date"]:
                    if field in id_info:
                        application_data["applicant"]["id_information"][field] = id_info[field]
        
        # Update address information from address proof
        if "address_proof" in extraction_results:
            address_info = extraction_results["address_proof"]
            
            # Update applicant address if available
            if "address" in address_info and "applicant" in application_data:
                if "address" not in application_data["applicant"]:
                    application_data["applicant"]["address"] = {}
                
                application_data["applicant"]["address"]["full_address"] = address_info["address"]
        
        # Update income information from income proof
        if "income_proof" in extraction_results:
            income_info = extraction_results["income_proof"]
            
            # Update applicant income if available
            if "income_amount" in income_info and "applicant" in application_data:
                if "financial_information" not in application_data["applicant"]:
                    application_data["applicant"]["financial_information"] = {}
                
                # Clean up income amount (remove currency symbols and commas)
                income_amount = income_info["income_amount"]
                income_amount = re.sub(r'[$£€,]', '', income_amount)
                
                try:
                    income_amount_float = float(income_amount)
                    
                    # Adjust income to annual if needed
                    period = income_info.get("period", "").lower()
                    if "monthly" in period:
                        income_amount_float *= 12
                    elif "bi-weekly" in period or "bi weekly" in period:
                        income_amount_float *= 26
                    elif "weekly" in period:
                        income_amount_float *= 52
                    
                    application_data["applicant"]["financial_information"]["annual_income"] = income_amount_float
                    application_data["applicant"]["financial_information"]["income_source"] = income_info.get("employer_name", "Employment")
                except ValueError:
                    self.logger.warning(f"Could not convert income amount to float: {income_amount}")
        
        # Update bank information from bank statement
        if "bank_statement" in extraction_results:
            bank_info = extraction_results["bank_statement"]
            
            # Update applicant bank information if available
            if "applicant" in application_data:
                if "financial_information" not in application_data["applicant"]:
                    application_data["applicant"]["financial_information"] = {}
                
                if "bank_accounts" not in application_data["applicant"]["financial_information"]:
                    application_data["applicant"]["financial_information"]["bank_accounts"] = []
                
                # Create bank account entry
                bank_account = {}
                for field in ["bank_name", "account_number", "account_holder"]:
                    if field in bank_info:
                        bank_account[field] = bank_info[field]
                
                # Add balance if available
                if "balance" in bank_info:
                    balance = bank_info["balance"]
                    balance = re.sub(r'[$£€,]', '', balance)
                    
                    try:
                        bank_account["balance"] = float(balance)
                    except ValueError:
                        self.logger.warning(f"Could not convert balance to float: {balance}")
                
                # Add to bank accounts list
                application_data["applicant"]["financial_information"]["bank_accounts"].append(bank_account)
        
        # Update the state with the updated application data
        state["application_data"] = application_data

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the agent
    agent = OCRAgent()
    
    # Example state with application data
    state = {
        "application_data": {
            "applicant": {
                "name": "John Doe",
                "contact_info": {
                    "email": "john.doe@example.com",
                    "phone": "+1234567890"
                }
            },
            "documents": {
                "id_proof": "id_proof.pdf",
                "income_proof": "income_proof.pdf",
                "address_proof": "address_proof.pdf",
                "bank_statement": "bank_statement.pdf"
            }
        }
    }
    
    # Process the state
    updated_state = agent.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

