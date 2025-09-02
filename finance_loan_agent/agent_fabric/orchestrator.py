"""
Orchestrator Module for Agent Fabric Architecture

This module implements the main orchestrator for the finance loan agent fabric
using LangGraph/ADK to coordinate the workflow between specialized agents.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum
import json

try:
    from langchain.schema import Document
    from langchain.schema.runnable import Runnable, RunnableConfig
    from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
    from langchain.schema.output_parser import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
except ImportError:
    logging.error("Required LangChain and LangGraph packages not found. Please install them.")
    logging.error("pip install langchain langchain-core langgraph")
    sys.exit(1)

# Import agent modules
try:
    from .agents.intake_agent import IntakeAgent
    from .agents.kyc_agent import KYCAgent
    from .agents.ocr_agent import OCRAgent
    from .agents.enrichment_agent import EnrichmentAgent
    from .agents.credit_agent import CreditAgent
    from .agents.fraud_agent import FraudAgent
    from .agents.collections_agent import CollectionsAgent
    from .retrieval import VectorDBRetrieval
    from .policy import PolicyGuardrails
    from .hitl import HiTLRouter
except ImportError:
    logging.warning("Agent modules not found. Using placeholder implementations.")
    # We'll define placeholder classes below

# Define workflow states
class WorkflowState(Enum):
    INTAKE = "intake"
    KYC = "kyc"
    OCR = "ocr"
    ENRICHMENT = "enrichment"
    CREDIT = "credit"
    FRAUD = "fraud"
    COLLECTIONS = "collections"
    RETRIEVAL = "retrieval"
    POLICY_CHECK = "policy_check"
    HITL = "hitl"
    COMPLETE = "complete"
    ERROR = "error"

# Placeholder agent classes if imports fail
class BaseAgent:
    """Base agent class with common functionality."""
    
    def __init__(self, name: str):
        self.name = name
        logging.info(f"Initializing {name} agent")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and return updated state."""
        logging.info(f"{self.name} agent processing state")
        return state

# Placeholder agent implementations
class IntakeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Intake")

class KYCAgent(BaseAgent):
    def __init__(self):
        super().__init__("KYC")

class OCRAgent(BaseAgent):
    def __init__(self):
        super().__init__("OCR")

class EnrichmentAgent(BaseAgent):
    def __init__(self):
        super().__init__("Enrichment")

class CreditAgent(BaseAgent):
    def __init__(self):
        super().__init__("Credit")

class FraudAgent(BaseAgent):
    def __init__(self):
        super().__init__("Fraud")

class CollectionsAgent(BaseAgent):
    def __init__(self):
        super().__init__("Collections")

class VectorDBRetrieval(BaseAgent):
    def __init__(self):
        super().__init__("Retrieval")

class PolicyGuardrails(BaseAgent):
    def __init__(self):
        super().__init__("Policy")

class HiTLRouter(BaseAgent):
    def __init__(self):
        super().__init__("HiTL")

class AgentFabricOrchestrator:
    """
    Orchestrator for the Agent Fabric Architecture.
    
    This class coordinates the workflow between specialized agents using LangGraph.
    """
    
    def __init__(self):
        """Initialize the orchestrator with all required agents."""
        # Initialize agents
        self.intake_agent = IntakeAgent()
        self.kyc_agent = KYCAgent()
        self.ocr_agent = OCRAgent()
        self.enrichment_agent = EnrichmentAgent()
        self.credit_agent = CreditAgent()
        self.fraud_agent = FraudAgent()
        self.collections_agent = CollectionsAgent()
        
        # Initialize supporting components
        self.retrieval = VectorDBRetrieval()
        self.policy_guardrails = PolicyGuardrails()
        self.hitl_router = HiTLRouter()
        
        # Build the workflow graph
        self.workflow = self._build_workflow_graph()
        
        logging.info("Agent Fabric Orchestrator initialized successfully")
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the workflow graph using LangGraph.
        
        Returns:
            StateGraph: The configured workflow graph
        """
        # Define the workflow graph
        workflow = StateGraph(name="LoanProcessingWorkflow")
        
        # Add nodes for each agent
        workflow.add_node(WorkflowState.INTAKE.value, self._run_intake_agent)
        workflow.add_node(WorkflowState.KYC.value, self._run_kyc_agent)
        workflow.add_node(WorkflowState.OCR.value, self._run_ocr_agent)
        workflow.add_node(WorkflowState.ENRICHMENT.value, self._run_enrichment_agent)
        workflow.add_node(WorkflowState.CREDIT.value, self._run_credit_agent)
        workflow.add_node(WorkflowState.FRAUD.value, self._run_fraud_agent)
        workflow.add_node(WorkflowState.COLLECTIONS.value, self._run_collections_agent)
        workflow.add_node(WorkflowState.RETRIEVAL.value, self._run_retrieval)
        workflow.add_node(WorkflowState.POLICY_CHECK.value, self._run_policy_check)
        workflow.add_node(WorkflowState.HITL.value, self._run_hitl_router)
        
        # Define the edges (transitions between states)
        # Start with intake
        workflow.add_edge(WorkflowState.INTAKE.value, WorkflowState.OCR.value)
        
        # OCR to KYC
        workflow.add_edge(WorkflowState.OCR.value, WorkflowState.KYC.value)
        
        # KYC to Enrichment
        workflow.add_edge(WorkflowState.KYC.value, WorkflowState.ENRICHMENT.value)
        
        # Enrichment to Credit
        workflow.add_edge(WorkflowState.ENRICHMENT.value, WorkflowState.CREDIT.value)
        
        # Credit to Fraud
        workflow.add_edge(WorkflowState.CREDIT.value, WorkflowState.FRAUD.value)
        
        # Fraud to Policy Check
        workflow.add_edge(WorkflowState.FRAUD.value, WorkflowState.POLICY_CHECK.value)
        
        # Policy Check conditional routing
        workflow.add_conditional_edges(
            WorkflowState.POLICY_CHECK.value,
            self._policy_check_router,
            {
                "approved": WorkflowState.COLLECTIONS.value,
                "rejected": END,
                "human_review": WorkflowState.HITL.value
            }
        )
        
        # HITL Router conditional routing
        workflow.add_conditional_edges(
            WorkflowState.HITL.value,
            self._hitl_decision_router,
            {
                "approved": WorkflowState.COLLECTIONS.value,
                "rejected": END,
                "more_info": WorkflowState.ENRICHMENT.value
            }
        )
        
        # Collections to Complete
        workflow.add_edge(WorkflowState.COLLECTIONS.value, END)
        
        # Set the entry point
        workflow.set_entry_point(WorkflowState.INTAKE.value)
        
        return workflow.compile()
    
    def _run_intake_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the intake agent on the current state."""
        return self.intake_agent.process(state)
    
    def _run_kyc_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the KYC agent on the current state."""
        return self.kyc_agent.process(state)
    
    def _run_ocr_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the OCR agent on the current state."""
        return self.ocr_agent.process(state)
    
    def _run_enrichment_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enrichment agent on the current state."""
        # First run retrieval to get additional information
        enriched_state = self._run_retrieval(state)
        # Then run the enrichment agent
        return self.enrichment_agent.process(enriched_state)
    
    def _run_credit_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the credit agent on the current state."""
        return self.credit_agent.process(state)
    
    def _run_fraud_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the fraud agent on the current state."""
        return self.fraud_agent.process(state)
    
    def _run_collections_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the collections agent on the current state."""
        return self.collections_agent.process(state)
    
    def _run_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the vector DB retrieval on the current state."""
        return self.retrieval.process(state)
    
    def _run_policy_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the policy guardrails on the current state."""
        return self.policy_guardrails.process(state)
    
    def _run_hitl_router(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the HiTL router on the current state."""
        return self.hitl_router.process(state)
    
    def _policy_check_router(self, state: Dict[str, Any]) -> str:
        """
        Route based on policy check results.
        
        Returns:
            str: Next state to transition to
        """
        if "policy_decision" not in state:
            logging.warning("No policy decision found in state, defaulting to human review")
            return "human_review"
        
        decision = state["policy_decision"]
        if decision == "approved":
            return "approved"
        elif decision == "rejected":
            return "rejected"
        else:
            return "human_review"
    
    def _hitl_decision_router(self, state: Dict[str, Any]) -> str:
        """
        Route based on human decision.
        
        Returns:
            str: Next state to transition to
        """
        if "human_decision" not in state:
            logging.warning("No human decision found in state, defaulting to more info")
            return "more_info"
        
        decision = state["human_decision"]
        if decision == "approved":
            return "approved"
        elif decision == "rejected":
            return "rejected"
        else:
            return "more_info"
    
    def process_loan_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a loan application through the entire workflow.
        
        Args:
            application_data: The initial loan application data
            
        Returns:
            Dict[str, Any]: The final state after processing
        """
        # Initialize the state with the application data
        initial_state = {
            "application_data": application_data,
            "metadata": {
                "workflow_id": application_data.get("application_id", "unknown"),
                "start_time": application_data.get("submission_time", "unknown"),
                "status": "started"
            },
            "history": []
        }
        
        try:
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Update the status
            final_state["metadata"]["status"] = "completed"
            
            return final_state
        except Exception as e:
            logging.error(f"Error processing loan application: {str(e)}")
            # Return error state
            return {
                "application_data": application_data,
                "metadata": {
                    "workflow_id": application_data.get("application_id", "unknown"),
                    "start_time": application_data.get("submission_time", "unknown"),
                    "status": "error",
                    "error": str(e)
                },
                "history": []
            }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the orchestrator
    orchestrator = AgentFabricOrchestrator()
    
    # Example loan application
    application = {
        "application_id": "LOAN-2023-001",
        "submission_time": "2023-01-01T12:00:00Z",
        "applicant": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1234567890"
        },
        "loan_details": {
            "amount": 50000,
            "term": 60,
            "purpose": "Home renovation"
        },
        "documents": {
            "id_proof": "id_proof.pdf",
            "income_proof": "income_proof.pdf",
            "address_proof": "address_proof.pdf"
        }
    }
    
    # Process the application
    result = orchestrator.process_loan_application(application)
    
    # Print the result
    print(json.dumps(result, indent=2))

