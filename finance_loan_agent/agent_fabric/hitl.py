"""
Human-in-the-Loop (HiTL) Router Module

This module implements the HiTL Router for the finance loan agent fabric.
The HiTL Router is responsible for managing human interventions in the
automated loan processing workflow.
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
import time

class HiTLRouter:
    """
    Human-in-the-Loop Router for managing human interventions.
    
    This component is responsible for:
    1. Routing cases to human reviewers
    2. Managing human review queues
    3. Collecting and processing human decisions
    4. Providing context for human review
    5. Tracking review status and metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the HiTL Router.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Human review queue
        self.review_queue = []
        
        # Review decisions (for demonstration)
        self.review_decisions = {}
        
        # Review handlers
        self.review_handlers = {}
        
        # Default timeout for human review (in seconds)
        self.default_timeout = self.config.get("default_timeout", 3600)  # 1 hour
        
        # Mock human reviewers
        self._setup_mock_reviewers()
        
        self.logger.info("HiTL Router initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and route for human review if needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state with human review results
        """
        self.logger.info("Processing human review request in HiTL Router")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Check if human review is required
        requires_human_review = state.get("requires_human_review", False)
        human_review_reason = state.get("human_review_reason", "")
        
        # If human review is not required, return state unchanged
        if not requires_human_review:
            self.logger.info("Human review not required, skipping")
            return state
        
        # Generate review ID
        review_id = str(uuid.uuid4())
        
        # Prepare review context
        review_context = self._prepare_review_context(state)
        
        # Create review request
        review_request = {
            "review_id": review_id,
            "application_id": application_data.get("application_id", "unknown"),
            "reason": human_review_reason,
            "context": review_context,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": None,
            "assigned_to": None,
            "decision": None,
            "notes": None
        }
        
        # Route to appropriate reviewer
        assigned_reviewer = self._route_to_reviewer(review_request)
        review_request["assigned_to"] = assigned_reviewer
        
        # Add to review queue
        self.review_queue.append(review_request)
        
        # In a real system, we would wait for human review
        # For demonstration, we'll simulate a human review
        review_result = self._simulate_human_review(review_request)
        
        # Update state with review result
        state["hitl_results"] = review_result
        
        # Set human decision for workflow routing
        state["human_decision"] = review_result.get("decision", "more_info")
        
        # Add HiTL metadata
        state["hitl_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "hitl_router_version": "1.0.0",
            "review_id": review_id,
            "reviewer": assigned_reviewer
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "HiTL",
            "timestamp": datetime.now().isoformat(),
            "action": "Processed human review",
            "details": {
                "review_id": review_id,
                "decision": review_result.get("decision"),
                "reviewer": assigned_reviewer
            }
        })
        
        self.logger.info(f"HiTL Router processing complete with decision: {review_result.get('decision')}")
        return state
    
    def register_review_handler(self, review_type: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register a handler for a specific type of review.
        
        Args:
            review_type: Type of review
            handler: Handler function
        """
        self.review_handlers[review_type] = handler
        self.logger.info(f"Registered review handler for {review_type}")
    
    def get_review_queue(self) -> List[Dict[str, Any]]:
        """
        Get the current review queue.
        
        Returns:
            List[Dict[str, Any]]: Review queue
        """
        return self.review_queue
    
    def get_review_status(self, review_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific review.
        
        Args:
            review_id: Review ID
            
        Returns:
            Optional[Dict[str, Any]]: Review status
        """
        for review in self.review_queue:
            if review["review_id"] == review_id:
                return review
        
        return None
    
    def submit_review_decision(self, review_id: str, decision: str, notes: Optional[str] = None) -> bool:
        """
        Submit a review decision.
        
        Args:
            review_id: Review ID
            decision: Decision (approved, rejected, more_info)
            notes: Optional notes
            
        Returns:
            bool: True if successful, False otherwise
        """
        for review in self.review_queue:
            if review["review_id"] == review_id:
                review["decision"] = decision
                review["notes"] = notes
                review["status"] = "completed"
                review["updated_at"] = datetime.now().isoformat()
                
                # Store decision
                self.review_decisions[review_id] = {
                    "decision": decision,
                    "notes": notes,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.logger.info(f"Review decision submitted for {review_id}: {decision}")
                return True
        
        self.logger.warning(f"Review {review_id} not found")
        return False
    
    def _prepare_review_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for human review.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Review context
        """
        # Extract relevant information for review
        context = {}
        
        # Include application data
        if "application_data" in state:
            context["application_data"] = state["application_data"]
        
        # Include agent results
        for key in state:
            if key.endswith("_results"):
                context[key] = state[key]
        
        # Include review reason
        if "human_review_reason" in state:
            context["review_reason"] = state["human_review_reason"]
        
        # Include workflow history
        if "history" in state:
            context["history"] = state["history"]
        
        return context
    
    def _route_to_reviewer(self, review_request: Dict[str, Any]) -> str:
        """
        Route review request to appropriate reviewer.
        
        Args:
            review_request: Review request
            
        Returns:
            str: Assigned reviewer ID
        """
        # Extract reason for review
        reason = review_request.get("reason", "").lower()
        
        # Route based on reason
        if "kyc" in reason or "identity" in reason or "verification" in reason:
            return self._assign_to_compliance_reviewer()
        elif "credit" in reason or "eligibility" in reason or "risk" in reason:
            return self._assign_to_credit_reviewer()
        elif "document" in reason or "ocr" in reason:
            return self._assign_to_document_reviewer()
        elif "fraud" in reason or "suspicious" in reason:
            return self._assign_to_fraud_reviewer()
        elif "policy" in reason or "compliance" in reason or "regulatory" in reason:
            return self._assign_to_compliance_reviewer()
        else:
            return self._assign_to_general_reviewer()
    
    def _assign_to_compliance_reviewer(self) -> str:
        """
        Assign to compliance reviewer.
        
        Returns:
            str: Reviewer ID
        """
        # In a real system, this would use a load balancing algorithm
        # For demonstration, we'll randomly select from compliance reviewers
        import random
        return random.choice(self.mock_reviewers["compliance"])
    
    def _assign_to_credit_reviewer(self) -> str:
        """
        Assign to credit reviewer.
        
        Returns:
            str: Reviewer ID
        """
        # In a real system, this would use a load balancing algorithm
        # For demonstration, we'll randomly select from credit reviewers
        import random
        return random.choice(self.mock_reviewers["credit"])
    
    def _assign_to_document_reviewer(self) -> str:
        """
        Assign to document reviewer.
        
        Returns:
            str: Reviewer ID
        """
        # In a real system, this would use a load balancing algorithm
        # For demonstration, we'll randomly select from document reviewers
        import random
        return random.choice(self.mock_reviewers["document"])
    
    def _assign_to_fraud_reviewer(self) -> str:
        """
        Assign to fraud reviewer.
        
        Returns:
            str: Reviewer ID
        """
        # In a real system, this would use a load balancing algorithm
        # For demonstration, we'll randomly select from fraud reviewers
        import random
        return random.choice(self.mock_reviewers["fraud"])
    
    def _assign_to_general_reviewer(self) -> str:
        """
        Assign to general reviewer.
        
        Returns:
            str: Reviewer ID
        """
        # In a real system, this would use a load balancing algorithm
        # For demonstration, we'll randomly select from general reviewers
        import random
        return random.choice(self.mock_reviewers["general"])
    
    def _simulate_human_review(self, review_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate human review for demonstration purposes.
        
        Args:
            review_request: Review request
            
        Returns:
            Dict[str, Any]: Review result
        """
        # In a real system, this would wait for human input
        # For demonstration, we'll simulate a review decision
        
        # Extract context
        context = review_request.get("context", {})
        application_data = context.get("application_data", {})
        
        # Extract applicant information
        applicant = application_data.get("applicant", {})
        credit_info = applicant.get("credit", {})
        credit_score = credit_info.get("credit_score", 0)
        
        # Extract loan details
        loan_details = application_data.get("loan_details", {})
        loan_amount = loan_details.get("amount", 0)
        
        # Extract review reason
        reason = review_request.get("reason", "").lower()
        
        # Simulate review decision based on context
        decision = ""
        notes = ""
        
        # Simulate thinking time
        time.sleep(0.1)
        
        # Make decision based on reason and context
        if "kyc" in reason or "identity" in reason:
            # For KYC issues, approve if credit score is high, otherwise reject
            if credit_score >= 700:
                decision = "approved"
                notes = "Identity verified through alternative means. Credit history supports approval."
            else:
                decision = "rejected"
                notes = "Unable to verify identity with provided documentation."
        
        elif "credit" in reason or "eligibility" in reason:
            # For credit issues, approve if credit score is borderline, otherwise more info
            if 620 <= credit_score < 660:
                decision = "approved"
                notes = "Borderline credit score but other factors support approval."
            else:
                decision = "more_info"
                notes = "Need additional income verification and explanation for recent credit inquiries."
        
        elif "document" in reason:
            # For document issues, usually need more info
            decision = "more_info"
            notes = "Need clearer copies of income documentation and address verification."
        
        elif "fraud" in reason or "suspicious" in reason:
            # For fraud concerns, usually reject
            decision = "rejected"
            notes = "Multiple inconsistencies detected in application that could not be resolved."
        
        elif "policy" in reason or "compliance" in reason:
            # For policy issues, depends on loan amount
            if loan_amount > 500000:
                decision = "more_info"
                notes = "Need additional documentation for high-value loan to meet compliance requirements."
            else:
                decision = "approved"
                notes = "Minor policy exceptions approved based on compensating factors."
        
        else:
            # Default to more info
            decision = "more_info"
            notes = "Please provide additional documentation to support the application."
        
        # Update review request
        review_request["decision"] = decision
        review_request["notes"] = notes
        review_request["status"] = "completed"
        review_request["updated_at"] = datetime.now().isoformat()
        
        # Return review result
        return {
            "review_id": review_request["review_id"],
            "decision": decision,
            "notes": notes,
            "reviewer": review_request["assigned_to"],
            "reviewed_at": datetime.now().isoformat()
        }
    
    def _setup_mock_reviewers(self):
        """Set up mock reviewers for demonstration purposes."""
        self.mock_reviewers = {
            "compliance": ["compliance-reviewer-1", "compliance-reviewer-2"],
            "credit": ["credit-reviewer-1", "credit-reviewer-2", "credit-reviewer-3"],
            "document": ["document-reviewer-1", "document-reviewer-2"],
            "fraud": ["fraud-reviewer-1", "fraud-reviewer-2"],
            "general": ["general-reviewer-1", "general-reviewer-2", "general-reviewer-3"]
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the HiTL router
    hitl_router = HiTLRouter()
    
    # Example state with application data
    state = {
        "application_data": {
            "application_id": "LOAN-2023-001",
            "applicant": {
                "name": "John Smith",
                "contact_info": {
                    "email": "john.smith@example.com",
                    "phone": "+1234567890"
                },
                "credit": {
                    "credit_score": 650
                }
            },
            "loan_details": {
                "amount": 250000,
                "term": 360,
                "purpose": "Home purchase"
            }
        },
        "requires_human_review": True,
        "human_review_reason": "Borderline credit score requires manual review"
    }
    
    # Process the state
    updated_state = hitl_router.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

