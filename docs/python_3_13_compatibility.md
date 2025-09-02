# Python 3.13.2 Compatibility Guide

This document explains the changes made to ensure compatibility with Python 3.13.2 and provides guidance on using the codebase with this version of Python.

## Overview

Python 3.13 introduces several changes that affect our codebase, particularly around TensorFlow compatibility. The main changes include:

1. TensorFlow is not officially compatible with Python 3.13.2 yet
2. Improved error handling and fallback mechanisms
3. Code structure and import path improvements
4. Type hints using Python 3.13 features

## TensorFlow Compatibility Layer

The most significant change is the addition of a TensorFlow compatibility layer that allows the code to run with reduced ML functionality when TensorFlow is not available. This is implemented in `finance_loan_agent/models/ml_compatibility.py`.

### Key Features

- Automatic detection of TensorFlow availability
- Fallback to scikit-learn models when TensorFlow is not available
- Graceful degradation of functionality
- Consistent API regardless of the backend used

### Usage

The compatibility layer is used automatically by the `CreditScoringModel` and `RiskAssessmentModel` classes. You don't need to change your code to use it.

```python
from finance_loan_agent.models.credit_scoring import CreditScoringModel

# This will use TensorFlow if available, or scikit-learn if not
model = CreditScoringModel()
model.train(X, y)
```

## Scikit-learn Fallback Models

Two new model implementations have been added to provide fallback functionality when TensorFlow is not available:

1. `finance_loan_agent/models/sklearn_credit_scoring.py`: A scikit-learn implementation of the credit scoring model
2. `finance_loan_agent/models/sklearn_risk_assessment.py`: A scikit-learn implementation of the risk assessment model

These models provide similar functionality to the TensorFlow models but may have reduced accuracy or performance.

## Improved Error Handling

Error handling has been improved throughout the codebase to provide better error messages and more graceful failure modes. This includes:

- More specific exception types
- Better error messages
- Fallback mechanisms for common failure modes
- Warnings when using reduced functionality

## Import Path Improvements

Import paths have been standardized to use both relative and absolute imports as appropriate. This makes the code more robust when run as a script or imported as a module.

```python
try:
    from .tools.finance_tools import analyze_loan_application
except ImportError:
    # Handle relative import error when running as script
    from tools.finance_tools import analyze_loan_application
```

## Type Hints

Type hints have been added throughout the codebase to improve code readability and maintainability. These use Python 3.13's improved typing features.

```python
def calculate_loan_terms(loan_amount: float, loan_term: int, interest_rate: float) -> Dict[str, float]:
    """Calculate monthly payment and total interest for a loan."""
    # ...
```

## Running with Python 3.13.2

When running with Python 3.13.2, you'll see a warning message indicating that some features may use fallback implementations:

```
Python version: 3.13.2
⚠️ Running on Python 3.13+. Some features may use fallback implementations.
```

This is normal and indicates that the code is using the compatibility layer.

## Known Limitations

When running with Python 3.13.2, the following limitations apply:

1. TensorFlow-based models are not available
2. Embeddings are created using a simple fallback method that is not suitable for production use
3. Some advanced features may not be available or may have reduced functionality

## Future Improvements

As TensorFlow adds official support for Python 3.13, we will update the codebase to take advantage of it. In the meantime, the compatibility layer provides a way to use the code with Python 3.13.2 with reduced functionality.

