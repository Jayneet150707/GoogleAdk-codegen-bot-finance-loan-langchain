"""
Vector DB Retrieval Module

This module implements the Vector DB Retrieval component for the finance loan agent fabric.
It provides semantic search capabilities for retrieving relevant information from a
vector database of documents, policies, and historical data.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import hashlib
import random
import numpy as np

class VectorDBRetrieval:
    """
    Vector DB Retrieval for semantic search and information retrieval.
    
    This component is responsible for:
    1. Retrieving relevant documents from a vector database
    2. Providing context for agent decision-making
    3. Finding similar historical cases
    4. Retrieving policy documents and guidelines
    5. Supporting knowledge-intensive tasks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Vector DB Retrieval component.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Vector database connection settings
        self.vector_db_config = self.config.get("vector_db", {
            "host": "localhost",
            "port": 27017,
            "database": "finance_loan_agent",
            "collection": "vector_embeddings"
        })
        
        # Embedding model settings
        self.embedding_config = self.config.get("embedding", {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384
        })
        
        # Set up mock vector database
        self._setup_mock_vector_database()
        
        self.logger.info("Vector DB Retrieval component initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and retrieve relevant information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state with retrieved information
        """
        self.logger.info("Retrieving information from vector database")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract query context from state
        query_context = self._extract_query_context(state)
        
        # Generate query embeddings
        query_embedding = self._generate_embedding(query_context)
        
        # Retrieve relevant documents
        relevant_documents = self._retrieve_documents(query_embedding, top_k=5)
        state["retrieval_results"] = {"relevant_documents": relevant_documents}
        
        # Retrieve similar cases
        similar_cases = self._retrieve_similar_cases(application_data, top_k=3)
        state["retrieval_results"]["similar_cases"] = similar_cases
        
        # Retrieve policy documents
        policy_documents = self._retrieve_policy_documents(query_context, top_k=3)
        state["retrieval_results"]["policy_documents"] = policy_documents
        
        # Add retrieval metadata
        state["retrieval_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "retrieval_version": "1.0.0",
            "query_context": query_context
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "Retrieval",
            "timestamp": datetime.now().isoformat(),
            "action": "Retrieved information from vector database",
            "details": {
                "num_documents": len(relevant_documents),
                "num_similar_cases": len(similar_cases),
                "num_policy_documents": len(policy_documents)
            }
        })
        
        self.logger.info("Vector DB Retrieval processing complete")
        return state
    
    def _extract_query_context(self, state: Dict[str, Any]) -> str:
        """
        Extract query context from the current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            str: Query context
        """
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract applicant information
        applicant = application_data.get("applicant", {})
        applicant_name = applicant.get("name", "")
        
        # Extract loan details
        loan_details = application_data.get("loan_details", {})
        loan_amount = loan_details.get("amount", 0)
        loan_purpose = loan_details.get("purpose", "")
        
        # Extract current agent
        current_agent = None
        if "history" in state and state["history"]:
            last_entry = state["history"][-1]
            current_agent = last_entry.get("agent", "")
        
        # Build query context
        query_parts = []
        
        if applicant_name:
            query_parts.append(f"Applicant: {applicant_name}")
        
        if loan_amount:
            query_parts.append(f"Loan amount: {loan_amount}")
        
        if loan_purpose:
            query_parts.append(f"Loan purpose: {loan_purpose}")
        
        if current_agent:
            query_parts.append(f"Current agent: {current_agent}")
        
        # Add specific context based on current agent
        if current_agent == "Intake":
            query_parts.append("Loan application requirements and validation")
        elif current_agent == "KYC":
            query_parts.append("Identity verification and KYC procedures")
        elif current_agent == "OCR":
            query_parts.append("Document processing and information extraction")
        elif current_agent == "Enrichment":
            query_parts.append("Data enrichment sources and procedures")
        elif current_agent == "Credit":
            query_parts.append("Credit assessment and loan eligibility criteria")
        elif current_agent == "Fraud":
            query_parts.append("Fraud detection patterns and procedures")
        elif current_agent == "Collections":
            query_parts.append("Loan servicing and collection procedures")
        elif current_agent == "PolicyGuardrails":
            query_parts.append("Regulatory compliance and policy rules")
        
        # Combine query parts
        query_context = " ".join(query_parts)
        
        return query_context
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.
        
        Args:
            text: Input text
            
        Returns:
            List[float]: Embedding vector
        """
        # In a real system, this would use a proper embedding model
        # For demonstration, we'll generate a deterministic pseudo-random vector
        
        # Create a hash of the text
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Use the hash to seed the random number generator
        random.seed(hash_hex)
        
        # Generate a random vector with the specified dimension
        dimension = self.embedding_config.get("dimension", 384)
        vector = [random.uniform(-1, 1) for _ in range(dimension)]
        
        # Normalize the vector
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def _retrieve_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Retrieved documents
        """
        # In a real system, this would query a vector database
        # For demonstration, we'll use the mock database
        
        # Calculate similarity scores for all documents
        scored_documents = []
        
        for doc in self.mock_documents:
            # Get document embedding
            doc_embedding = doc.get("embedding", [])
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Add to scored documents
            scored_documents.append((similarity, doc))
        
        # Sort by similarity score (descending)
        scored_documents.sort(reverse=True, key=lambda x: x[0])
        
        # Take top-k results
        top_documents = scored_documents[:top_k]
        
        # Format results
        results = []
        
        for similarity, doc in top_documents:
            # Create a copy without the embedding
            doc_copy = doc.copy()
            doc_copy.pop("embedding", None)
            
            # Add similarity score
            doc_copy["similarity_score"] = round(similarity, 4)
            
            results.append(doc_copy)
        
        return results
    
    def _retrieve_similar_cases(self, application_data: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical cases.
        
        Args:
            application_data: Current application data
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Similar cases
        """
        # Extract key features for similarity matching
        loan_details = application_data.get("loan_details", {})
        loan_amount = loan_details.get("amount", 0)
        loan_purpose = loan_details.get("purpose", "")
        
        applicant = application_data.get("applicant", {})
        credit_info = applicant.get("credit", {})
        credit_score = credit_info.get("credit_score", 0)
        
        # Create a query string for the case
        query_string = f"Loan amount: {loan_amount} Purpose: {loan_purpose} Credit score: {credit_score}"
        
        # Generate embedding
        query_embedding = self._generate_embedding(query_string)
        
        # Filter for historical cases only
        historical_cases = [doc for doc in self.mock_documents if doc.get("type") == "historical_case"]
        
        # Calculate similarity scores
        scored_cases = []
        
        for case in historical_cases:
            # Get case embedding
            case_embedding = case.get("embedding", [])
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, case_embedding)
            
            # Add to scored cases
            scored_cases.append((similarity, case))
        
        # Sort by similarity score (descending)
        scored_cases.sort(reverse=True, key=lambda x: x[0])
        
        # Take top-k results
        top_cases = scored_cases[:top_k]
        
        # Format results
        results = []
        
        for similarity, case in top_cases:
            # Create a copy without the embedding
            case_copy = case.copy()
            case_copy.pop("embedding", None)
            
            # Add similarity score
            case_copy["similarity_score"] = round(similarity, 4)
            
            results.append(case_copy)
        
        return results
    
    def _retrieve_policy_documents(self, query_context: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policy documents.
        
        Args:
            query_context: Query context
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Policy documents
        """
        # Generate embedding
        query_embedding = self._generate_embedding(query_context)
        
        # Filter for policy documents only
        policy_documents = [doc for doc in self.mock_documents if doc.get("type") == "policy"]
        
        # Calculate similarity scores
        scored_policies = []
        
        for policy in policy_documents:
            # Get policy embedding
            policy_embedding = policy.get("embedding", [])
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, policy_embedding)
            
            # Add to scored policies
            scored_policies.append((similarity, policy))
        
        # Sort by similarity score (descending)
        scored_policies.sort(reverse=True, key=lambda x: x[0])
        
        # Take top-k results
        top_policies = scored_policies[:top_k]
        
        # Format results
        results = []
        
        for similarity, policy in top_policies:
            # Create a copy without the embedding
            policy_copy = policy.copy()
            policy_copy.pop("embedding", None)
            
            # Add similarity score
            policy_copy["similarity_score"] = round(similarity, 4)
            
            results.append(policy_copy)
        
        return results
    
    def _setup_mock_vector_database(self):
        """Set up mock vector database for demonstration purposes."""
        # Create mock documents
        self.mock_documents = []
        
        # Add policy documents
        policy_docs = [
            {
                "id": "policy-001",
                "type": "policy",
                "title": "Loan Eligibility Criteria",
                "content": "This document outlines the eligibility criteria for loan applications. Applicants must have a minimum credit score of 620, debt-to-income ratio below 43%, and loan-to-value ratio below 80%. Additional requirements may apply based on loan type and amount.",
                "category": "credit_policy",
                "last_updated": "2023-01-15T00:00:00Z"
            },
            {
                "id": "policy-002",
                "type": "policy",
                "title": "KYC Verification Procedures",
                "content": "This document outlines the Know Your Customer (KYC) verification procedures. All applicants must provide valid identification documents, proof of address, and undergo identity verification. Additional checks may include background screening and anti-money laundering (AML) verification.",
                "category": "compliance",
                "last_updated": "2023-02-10T00:00:00Z"
            },
            {
                "id": "policy-003",
                "type": "policy",
                "title": "Fraud Detection Guidelines",
                "content": "This document outlines the guidelines for detecting and preventing fraud in loan applications. Key indicators include inconsistent information, suspicious documentation, unusual transaction patterns, and identity mismatches. All flagged applications must undergo enhanced due diligence.",
                "category": "risk_management",
                "last_updated": "2023-03-05T00:00:00Z"
            },
            {
                "id": "policy-004",
                "type": "policy",
                "title": "Regulatory Compliance Requirements",
                "content": "This document outlines the regulatory compliance requirements for loan processing. All loan applications must comply with relevant regulations including fair lending laws, disclosure requirements, privacy regulations, and anti-discrimination policies.",
                "category": "compliance",
                "last_updated": "2023-04-20T00:00:00Z"
            },
            {
                "id": "policy-005",
                "type": "policy",
                "title": "Loan Pricing Guidelines",
                "content": "This document outlines the guidelines for loan pricing and interest rate determination. Interest rates are based on credit score, loan-to-value ratio, debt-to-income ratio, loan term, and current market conditions. Risk-based pricing adjustments may apply.",
                "category": "credit_policy",
                "last_updated": "2023-05-12T00:00:00Z"
            }
        ]
        
        # Add historical cases
        historical_cases = [
            {
                "id": "case-001",
                "type": "historical_case",
                "applicant_profile": {
                    "credit_score": 720,
                    "income": 85000,
                    "debt_to_income_ratio": 0.32
                },
                "loan_details": {
                    "amount": 250000,
                    "term": 360,
                    "purpose": "Home purchase",
                    "loan_to_value_ratio": 0.75
                },
                "decision": "approved",
                "decision_factors": ["Strong credit profile", "Stable income", "Acceptable LTV ratio"],
                "outcome": "Loan performing as expected",
                "date": "2022-06-15T00:00:00Z"
            },
            {
                "id": "case-002",
                "type": "historical_case",
                "applicant_profile": {
                    "credit_score": 680,
                    "income": 65000,
                    "debt_to_income_ratio": 0.38
                },
                "loan_details": {
                    "amount": 200000,
                    "term": 360,
                    "purpose": "Home purchase",
                    "loan_to_value_ratio": 0.8
                },
                "decision": "approved",
                "decision_factors": ["Acceptable credit profile", "Stable income", "Maximum LTV ratio"],
                "outcome": "Loan performing with occasional late payments",
                "date": "2022-07-22T00:00:00Z"
            },
            {
                "id": "case-003",
                "type": "historical_case",
                "applicant_profile": {
                    "credit_score": 620,
                    "income": 55000,
                    "debt_to_income_ratio": 0.42
                },
                "loan_details": {
                    "amount": 180000,
                    "term": 360,
                    "purpose": "Home purchase",
                    "loan_to_value_ratio": 0.85
                },
                "decision": "rejected",
                "decision_factors": ["Borderline credit profile", "High DTI ratio", "Exceeds maximum LTV ratio"],
                "outcome": "N/A",
                "date": "2022-08-10T00:00:00Z"
            },
            {
                "id": "case-004",
                "type": "historical_case",
                "applicant_profile": {
                    "credit_score": 750,
                    "income": 120000,
                    "debt_to_income_ratio": 0.28
                },
                "loan_details": {
                    "amount": 400000,
                    "term": 360,
                    "purpose": "Home purchase",
                    "loan_to_value_ratio": 0.7
                },
                "decision": "approved",
                "decision_factors": ["Excellent credit profile", "High income", "Good LTV ratio"],
                "outcome": "Loan performing as expected",
                "date": "2022-09-05T00:00:00Z"
            },
            {
                "id": "case-005",
                "type": "historical_case",
                "applicant_profile": {
                    "credit_score": 600,
                    "income": 60000,
                    "debt_to_income_ratio": 0.45
                },
                "loan_details": {
                    "amount": 150000,
                    "term": 360,
                    "purpose": "Home purchase",
                    "loan_to_value_ratio": 0.9
                },
                "decision": "rejected",
                "decision_factors": ["Below minimum credit score", "Exceeds maximum DTI ratio", "Exceeds maximum LTV ratio"],
                "outcome": "N/A",
                "date": "2022-10-18T00:00:00Z"
            }
        ]
        
        # Add reference documents
        reference_docs = [
            {
                "id": "ref-001",
                "type": "reference",
                "title": "Mortgage Loan Processing Guide",
                "content": "This guide provides comprehensive information on mortgage loan processing, including application requirements, documentation, underwriting criteria, and approval procedures. It covers conventional, FHA, VA, and USDA loans.",
                "category": "procedures",
                "last_updated": "2023-01-30T00:00:00Z"
            },
            {
                "id": "ref-002",
                "type": "reference",
                "title": "Credit Scoring Models Explained",
                "content": "This document explains various credit scoring models used in loan underwriting, including FICO Score, VantageScore, and custom models. It covers score ranges, factors affecting scores, and how scores impact loan decisions.",
                "category": "credit_assessment",
                "last_updated": "2023-02-15T00:00:00Z"
            },
            {
                "id": "ref-003",
                "type": "reference",
                "title": "Document Verification Best Practices",
                "content": "This document outlines best practices for verifying loan application documents, including ID verification, income verification, employment verification, and property documentation. It includes procedures for detecting fraudulent documents.",
                "category": "verification",
                "last_updated": "2023-03-10T00:00:00Z"
            },
            {
                "id": "ref-004",
                "type": "reference",
                "title": "Loan Servicing Procedures",
                "content": "This document outlines procedures for loan servicing, including payment processing, account management, customer communication, delinquency management, and collections. It covers the entire loan lifecycle from closing to payoff.",
                "category": "servicing",
                "last_updated": "2023-04-05T00:00:00Z"
            },
            {
                "id": "ref-005",
                "type": "reference",
                "title": "Regulatory Compliance Handbook",
                "content": "This handbook provides comprehensive information on regulatory compliance requirements for mortgage lending, including TILA, RESPA, ECOA, FCRA, HMDA, and other relevant regulations. It includes compliance checklists and procedures.",
                "category": "compliance",
                "last_updated": "2023-05-20T00:00:00Z"
            }
        ]
        
        # Combine all documents
        all_docs = policy_docs + historical_cases + reference_docs
        
        # Add embeddings to documents
        for doc in all_docs:
            # Generate embedding from document content
            content = doc.get("content", "") + doc.get("title", "")
            doc["embedding"] = self._generate_embedding(content)
            
            # Add to mock database
            self.mock_documents.append(doc)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the retrieval component
    retrieval = VectorDBRetrieval()
    
    # Example state with application data
    state = {
        "application_data": {
            "applicant": {
                "name": "John Smith",
                "contact_info": {
                    "email": "john.smith@example.com",
                    "phone": "+1234567890"
                },
                "credit": {
                    "credit_score": 720
                }
            },
            "loan_details": {
                "amount": 250000,
                "term": 360,
                "purpose": "Home purchase"
            }
        },
        "history": [
            {
                "agent": "Intake",
                "timestamp": "2023-06-01T12:00:00Z",
                "action": "Processed application"
            },
            {
                "agent": "KYC",
                "timestamp": "2023-06-01T12:05:00Z",
                "action": "Verified identity"
            },
            {
                "agent": "Credit",
                "timestamp": "2023-06-01T12:10:00Z",
                "action": "Assessing creditworthiness"
            }
        ]
    }
    
    # Process the state
    updated_state = retrieval.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

