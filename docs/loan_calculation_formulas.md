# Loan Calculation Formulas and Mathematical Algorithms

This document provides a detailed explanation of the mathematical formulas and algorithms used in the loan calculations within the Finance Loan Agent.

## Interest Rate Calculation

### Simple Interest

Simple interest is calculated using the formula:

$$I = P \times r \times t$$

Where:
- $I$ = Interest amount
- $P$ = Principal (loan amount)
- $r$ = Interest rate (as a decimal)
- $t$ = Time period in years

### Compound Interest

Compound interest is calculated using the formula:

$$A = P \times (1 + r)^t$$

Where:
- $A$ = Final amount (principal + interest)
- $P$ = Principal (loan amount)
- $r$ = Interest rate per period (as a decimal)
- $t$ = Number of time periods

## Loan Payment Calculation

### Monthly Payment Formula

The monthly payment for a loan is calculated using the formula:

$$M = P \times \frac{r(1+r)^n}{(1+r)^n-1}$$

Where:
- $M$ = Monthly payment
- $P$ = Principal (loan amount)
- $r$ = Monthly interest rate (annual rate divided by 12, as a decimal)
- $n$ = Total number of payments (loan term in months)

This formula is derived from the present value of an annuity formula and ensures that the loan is fully amortized over the specified term.

### Implementation in Code

```python
def calculate_loan_terms(loan_amount, loan_term, interest_rate):
    """
    Calculate monthly payment and total interest for a loan.
    
    Args:
        loan_amount: Principal loan amount
        loan_term: Loan term in months
        interest_rate: Annual interest rate as a percentage
        
    Returns:
        Dict: Monthly payment, total payment, and total interest
    """
    # Convert annual interest rate to monthly
    monthly_rate = interest_rate / 12 / 100
    
    # Calculate monthly payment using the loan formula
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term) / ((1 + monthly_rate) ** loan_term - 1)
    
    # Calculate total payment and interest
    total_payment = monthly_payment * loan_term
    total_interest = total_payment - loan_amount
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2)
    }
```

## Amortization Schedule Calculation

An amortization schedule shows the breakdown of each payment into principal and interest components over the life of the loan.

### Algorithm for Generating Amortization Schedule

1. Calculate the monthly payment using the formula above
2. For each payment period:
   - Calculate the interest portion: `interest = remaining_balance * monthly_rate`
   - Calculate the principal portion: `principal = monthly_payment - interest`
   - Update the remaining balance: `remaining_balance = remaining_balance - principal`
   - Add the payment details to the schedule

### Implementation in Code

```python
def generate_amortization_schedule(loan_amount, loan_term, interest_rate):
    """
    Generate an amortization schedule for a loan.
    
    Args:
        loan_amount: Principal loan amount
        loan_term: Loan term in months
        interest_rate: Annual interest rate as a percentage
        
    Returns:
        List: Amortization schedule with payment details
    """
    # Convert annual interest rate to monthly
    monthly_rate = interest_rate / 12 / 100
    
    # Calculate monthly payment
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term) / ((1 + monthly_rate) ** loan_term - 1)
    
    # Initialize variables
    remaining_balance = loan_amount
    schedule = []
    
    # Generate schedule for each payment period
    for period in range(1, loan_term + 1):
        # Calculate interest and principal for this period
        interest_payment = remaining_balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        
        # Update remaining balance
        remaining_balance -= principal_payment
        
        # Add payment details to schedule
        schedule.append({
            "period": period,
            "payment": monthly_payment,
            "principal": principal_payment,
            "interest": interest_payment,
            "remaining_balance": max(0, remaining_balance)  # Ensure balance doesn't go below 0 due to rounding
        })
    
    return schedule
```

## Interest Rate Recommendation Algorithm

The interest rate recommendation is based on several factors including credit score, loan term, and loan amount. The algorithm uses a base rate and applies adjustments based on these factors.

### Base Rate

The base rate is the starting point for the interest rate calculation. This is typically set based on current market conditions and the financial institution's policies.

### Credit Score Adjustment

Credit score adjustments are applied based on the borrower's creditworthiness:

| Credit Score Range | Adjustment |
|-------------------|------------|
| 800+              | -1.5%      |
| 750-799           | -1.0%      |
| 700-749           | -0.5%      |
| 650-699           | 0.0%       |
| 600-649           | +1.0%      |
| Below 600         | +2.0%      |

### Loan Term Adjustment

Loan term adjustments are applied based on the length of the loan:

| Loan Term Range | Adjustment |
|----------------|------------|
| ≤ 36 months    | -0.25%     |
| 37-60 months   | 0.0%       |
| > 60 months    | +0.5%      |

### Loan Amount Adjustment

Loan amount adjustments are applied based on the size of the loan:

| Loan Amount Range | Adjustment |
|------------------|------------|
| ≥ $100,000       | -0.25%     |
| $50,000-$99,999  | 0.0%       |
| < $50,000        | +0.25%     |

### Implementation in Code

```python
def recommend_interest_rate(credit_score, loan_term, loan_amount):
    """
    Recommend an interest rate based on credit score and loan details.
    
    Args:
        credit_score: Applicant's credit score
        loan_term: Loan term in months
        loan_amount: Principal loan amount
        
    Returns:
        float: Recommended interest rate as a percentage
    """
    # Base rate
    base_rate = 5.0
    
    # Adjust for credit score
    if credit_score >= 800:
        credit_adjustment = -1.5
    elif credit_score >= 750:
        credit_adjustment = -1.0
    elif credit_score >= 700:
        credit_adjustment = -0.5
    elif credit_score >= 650:
        credit_adjustment = 0.0
    elif credit_score >= 600:
        credit_adjustment = 1.0
    else:
        credit_adjustment = 2.0
    
    # Adjust for loan term
    if loan_term <= 36:
        term_adjustment = -0.25
    elif loan_term <= 60:
        term_adjustment = 0.0
    else:
        term_adjustment = 0.5
    
    # Adjust for loan amount
    if loan_amount >= 100000:
        amount_adjustment = -0.25
    elif loan_amount >= 50000:
        amount_adjustment = 0.0
    else:
        amount_adjustment = 0.25
    
    # Calculate recommended rate
    recommended_rate = base_rate + credit_adjustment + term_adjustment + amount_adjustment
    
    # Ensure rate is within reasonable bounds
    recommended_rate = max(2.0, min(recommended_rate, 15.0))
    
    return round(recommended_rate, 2)
```

## Risk Assessment Algorithms

### Credit Risk Scoring

The credit risk scoring model uses machine learning to predict the probability of default. The model is trained on historical loan data and uses features such as:

- Income
- Credit score
- Debt-to-income ratio
- Loan amount
- Loan term

The model outputs a risk score between 0 and 1, where higher values indicate higher risk.

### Risk Level Classification

The risk score is classified into risk levels using the following thresholds:

| Risk Score Range | Risk Level  | Recommendation |
|-----------------|-------------|---------------|
| < 0.3           | Low Risk    | Approve       |
| 0.3 - 0.7       | Medium Risk | Review Manually |
| > 0.7           | High Risk   | Deny          |

### Implementation in Code

```python
def analyze_loan_application(applicant_data):
    """
    Analyze a loan application and return a risk assessment.
    
    Args:
        applicant_data: Dictionary containing applicant information
        
    Returns:
        Dict: Risk assessment results including score, level, recommendation, and explanation
    """
    # Convert applicant data to DataFrame
    df = pd.DataFrame([applicant_data])
    
    # Make prediction using the model
    risk_score = model.predict(df)[0][0]
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = "Low Risk"
        recommendation = "Approve"
    elif risk_score < 0.7:
        risk_level = "Medium Risk"
        recommendation = "Review Manually"
    else:
        risk_level = "High Risk"
        recommendation = "Deny"
    
    return {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "explanation": generate_explanation(df, risk_score)
    }
```

## Advanced Mathematical Concepts

### Present Value of Annuity

The loan payment formula is derived from the present value of an annuity formula:

$$PV = PMT \times \frac{1 - (1 + r)^{-n}}{r}$$

Where:
- $PV$ = Present value (loan amount)
- $PMT$ = Payment amount
- $r$ = Interest rate per period
- $n$ = Number of periods

Solving for PMT gives us the loan payment formula:

$$PMT = PV \times \frac{r}{1 - (1 + r)^{-n}} = PV \times \frac{r(1+r)^n}{(1+r)^n-1}$$

### Effective Annual Rate (EAR)

The effective annual rate accounts for compounding and is calculated using the formula:

$$EAR = \left(1 + \frac{r}{m}\right)^m - 1$$

Where:
- $EAR$ = Effective annual rate
- $r$ = Nominal annual interest rate
- $m$ = Number of compounding periods per year

### Annual Percentage Rate (APR)

The APR includes the effect of fees and is calculated using the formula:

$$APR = \frac{Fees + (Principal \times Rate \times Time)}{Principal \times Time} \times 100\%$$

Where:
- $APR$ = Annual percentage rate
- $Fees$ = Total fees charged
- $Principal$ = Loan amount
- $Rate$ = Interest rate
- $Time$ = Loan term in years

## Conclusion

These mathematical formulas and algorithms form the foundation of the loan calculations in the Finance Loan Agent. They ensure accurate and consistent results for loan payments, interest rates, and risk assessments.

