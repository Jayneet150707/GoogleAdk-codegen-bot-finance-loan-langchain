# Loan Calculation Formulas and Mathematical Algorithms

This document provides a detailed explanation of the mathematical formulas and algorithms used in the loan calculations within the Finance Loan Agent.

## Interest Rate Systems

### Flat Interest

Flat interest is calculated on the full principal amount throughout the loan term, regardless of the declining balance. This is the default interest system used in our application for Indian Rupee loans.

The formula for flat interest is:

$$I = P \times r \times t$$

Where:
- $I$ = Total interest amount
- $P$ = Principal (loan amount)
- $r$ = Interest rate (as a decimal)
- $t$ = Time period in years

The monthly payment with flat interest is calculated as:

$$M = \frac{P + I}{n}$$

Where:
- $M$ = Monthly payment
- $P$ = Principal (loan amount)
- $I$ = Total interest amount
- $n$ = Total number of payments (loan term in months)

### Reducing Balance (EMI)

Reducing balance interest (also known as EMI - Equated Monthly Installment) is calculated on the remaining principal amount, which decreases with each payment.

The formula for the monthly payment with reducing balance interest is:

$$M = P \times \frac{r(1+r)^n}{(1+r)^n-1}$$

Where:
- $M$ = Monthly payment
- $P$ = Principal (loan amount)
- $r$ = Monthly interest rate (annual rate divided by 12, as a decimal)
- $n$ = Total number of payments (loan term in months)

## CIBIL Score Relaxation

For Indian borrowers with high CIBIL scores (>750), we provide special interest rate relaxations:

| CIBIL Score Range | Interest Rate Relaxation |
|-------------------|--------------------------|
| 800+              | -3.0%                    |
| 775-799           | -2.0%                    |
| 751-774           | -1.0%                    |
| ≤750              | Standard adjustments     |

### Implementation in Code

```python
# Special relaxation for CIBIL scores > 750
if credit_score > 750:
    # Apply relaxation for high credit scores
    if credit_score >= 800:
        credit_adjustment = -3.0
    elif credit_score >= 775:
        credit_adjustment = -2.0
    else:  # 751-774
        credit_adjustment = -1.0
        
    relaxation_applied = True
else:
    # Standard adjustments for other credit scores
    if credit_score >= 700:
        credit_adjustment = -0.5
    elif credit_score >= 650:
        credit_adjustment = 0.0
    elif credit_score >= 600:
        credit_adjustment = 1.0
    else:
        credit_adjustment = 2.0
        
    relaxation_applied = False
```

## Loan Payment Calculation

### Flat Interest Monthly Payment Formula

For flat interest loans, the monthly payment is calculated using the formula:

$$M = \frac{P + (P \times r \times t)}{n}$$

Where:
- $M$ = Monthly payment
- $P$ = Principal (loan amount)
- $r$ = Annual interest rate (as a decimal)
- $t$ = Loan term in years
- $n$ = Total number of payments (loan term in months)

### Implementation in Code

```python
def calculate_flat_interest_loan_terms(loan_amount, loan_term, interest_rate, currency="INR"):
    """
    Calculate monthly payment and total interest for a loan using flat interest rate.
    """
    # Calculate total interest for the entire loan term
    total_interest = loan_amount * (interest_rate / 100) * (loan_term / 12)
    
    # Calculate total payment
    total_payment = loan_amount + total_interest
    
    # Calculate monthly payment (equal installments)
    monthly_payment = total_payment / loan_term
    
    # Format currency symbol
    currency_symbol = "₹" if currency == "INR" else "$"
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "interest_type": "Flat Interest",
        "currency": currency,
        "currency_symbol": currency_symbol
    }
```

## Amortization Schedule Calculation

### Flat Interest Amortization Schedule

For flat interest loans, the amortization schedule is calculated as follows:

1. Monthly interest = (Principal × Annual interest rate) ÷ (12 × 100)
2. Monthly principal = Principal ÷ Loan term in months
3. Monthly payment = Monthly principal + Monthly interest
4. Remaining balance = Principal - (Monthly principal × Payment number)

### Implementation in Code

```python
# Use flat interest calculation if specified
if interest_type.lower() == "flat":
    # Calculate monthly interest
    monthly_interest = (loan_amount * interest_rate / 100) / 12
    
    # Calculate monthly principal
    monthly_principal = loan_amount / loan_term
    
    # Calculate monthly payment
    monthly_payment = monthly_principal + monthly_interest
    
    # Generate schedule for each payment period
    remaining_balance = loan_amount
    for period in range(1, loan_term + 1):
        # Update remaining balance
        remaining_balance -= monthly_principal
        
        # Add payment details to schedule
        schedule.append({
            "period": period,
            "payment": round(monthly_payment, 2),
            "principal": round(monthly_principal, 2),
            "interest": round(monthly_interest, 2),
            "remaining_balance": round(max(0, remaining_balance), 2),
            "currency": currency,
            "currency_symbol": currency_symbol
        })
```

## Interest Rate Recommendation Algorithm

The interest rate recommendation is based on several factors including CIBIL score, loan term, and loan amount. The algorithm uses a base rate and applies adjustments based on these factors.

### Base Rate

The base rate is the starting point for the interest rate calculation:

- For flat interest: 12.0%
- For reducing balance: 9.0%

### CIBIL Score Adjustment

CIBIL score adjustments are applied based on the borrower's creditworthiness:

| CIBIL Score Range | Adjustment |
|-------------------|------------|
| 800+              | -3.0%      |
| 775-799           | -2.0%      |
| 751-774           | -1.0%      |
| 700-750           | -0.5%      |
| 650-699           | 0.0%       |
| 600-649           | +1.0%      |
| Below 600         | +2.0%      |

### Loan Term Adjustment

Loan term adjustments are applied based on the length of the loan:

| Loan Term Range | Adjustment |
|----------------|------------|
| ≤ 12 months    | -0.5%      |
| 13-36 months   | -0.25%     |
| 37-60 months   | 0.0%       |
| > 60 months    | +0.5%      |

### Loan Amount Adjustment

Loan amount adjustments are applied based on the size of the loan:

| Loan Amount Range | Adjustment |
|------------------|------------|
| ≥ ₹10,00,000     | -0.5%      |
| ₹5,00,000-₹9,99,999 | -0.25%  |
| ₹1,00,000-₹4,99,999 | 0.0%    |
| < ₹1,00,000      | +0.25%     |

### Interest Rate Bounds

The recommended interest rate is constrained within reasonable bounds:

- For flat interest: 7.0% to 18.0%
- For reducing balance: 6.0% to 16.0%

## Risk Assessment Algorithms

### Credit Risk Scoring

The credit risk scoring model uses machine learning to predict the probability of default. The model is trained on historical loan data and uses features such as:

- Income
- CIBIL score
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

## Advanced Mathematical Concepts

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

For flat interest loans, the APR is calculated as:

$$APR = \frac{Fees + (Principal \times Rate \times Time)}{Principal \times Time} \times 100\%$$

For reducing balance loans, the APR is calculated based on the total interest paid:

$$APR = \frac{Fees + Total Interest}{Principal \times Time} \times 100\%$$

## Conclusion

These mathematical formulas and algorithms form the foundation of the loan calculations in the Finance Loan Agent. They ensure accurate and consistent results for loan payments, interest rates, and risk assessments, with special considerations for Indian borrowers using the flat interest system and CIBIL score relaxations.

