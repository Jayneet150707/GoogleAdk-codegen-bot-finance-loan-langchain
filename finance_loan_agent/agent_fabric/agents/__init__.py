"""
Agent Modules Package

This package contains the specialized agent implementations for the finance loan processing system.
"""

# Import all agents for convenience
from .intake_agent import IntakeAgent
from .kyc_agent import KYCAgent
from .ocr_agent import OCRAgent
from .enrichment_agent import EnrichmentAgent
from .credit_agent import CreditAgent
from .fraud_agent import FraudAgent
from .collections_agent import CollectionsAgent

__all__ = [
    'IntakeAgent',
    'KYCAgent',
    'OCRAgent',
    'EnrichmentAgent',
    'CreditAgent',
    'FraudAgent',
    'CollectionsAgent'
]

