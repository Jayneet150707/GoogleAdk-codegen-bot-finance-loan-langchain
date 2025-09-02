"""
Enrichment Agent Module

This module implements the Enrichment Agent for the finance loan agent fabric.
The Enrichment Agent is responsible for enhancing application data with additional
information from external sources and internal databases.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import hashlib
import random

class EnrichmentAgent:
    """
    Enrichment Agent for enhancing application data with additional information.
    
    This agent is responsible for:
    1. Retrieving additional information from external sources
    2. Standardizing and normalizing data
    3. Calculating derived metrics
    4. Enhancing application data for downstream processing
    5. Resolving data inconsistencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Enrichment Agent.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Mock external data sources
        self._setup_mock_data_sources()
        
        self.logger.info("Enrichment Agent initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and enrich application data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict[str, Any]: Updated workflow state with enriched data
        """
        self.logger.info("Enriching application data in Enrichment Agent")
        
        # Extract application data
        application_data = state.get("application_data", {})
        
        # Extract applicant information
        applicant = application_data.get("applicant", {})
        
        # Enrich with property information
        property_info = self._enrich_property_information(application_data)
        state["enrichment_results"] = {"property_information": property_info}
        
        # Enrich with employment verification
        employment_verification = self._verify_employment(applicant)
        state["enrichment_results"]["employment_verification"] = employment_verification
        
        # Enrich with credit bureau data
        credit_bureau_data = self._get_credit_bureau_data(applicant)
        state["enrichment_results"]["credit_bureau_data"] = credit_bureau_data
        
        # Enrich with market data
        market_data = self._get_market_data(application_data)
        state["enrichment_results"]["market_data"] = market_data
        
        # Calculate derived metrics
        derived_metrics = self._calculate_derived_metrics(
            application_data,
            property_info,
            employment_verification,
            credit_bureau_data,
            market_data
        )
        state["enrichment_results"]["derived_metrics"] = derived_metrics
        
        # Update application data with enriched information
        self._update_application_data(state, state["enrichment_results"])
        
        # Add enrichment metadata
        state["enrichment_metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "enrichment_agent_version": "1.0.0",
            "data_sources": ["property_database", "employment_verification", "credit_bureau", "market_data"]
        }
        
        # Add to history
        if "history" not in state:
            state["history"] = []
        
        state["history"].append({
            "agent": "Enrichment",
            "timestamp": datetime.now().isoformat(),
            "action": "Enriched application data",
            "details": {
                "enrichment_sources": ["property_database", "employment_verification", "credit_bureau", "market_data"],
                "derived_metrics_calculated": list(derived_metrics.keys())
            }
        })
        
        self.logger.info("Enrichment Agent processing complete")
        return state
    
    def _enrich_property_information(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich application with property information.
        
        Args:
            application_data: Application data
            
        Returns:
            Dict[str, Any]: Property information
        """
        # Extract property information from application
        loan_details = application_data.get("loan_details", {})
        property_address = loan_details.get("property_address", "")
        
        # If no property address, try to get applicant address
        if not property_address and "applicant" in application_data:
            applicant = application_data["applicant"]
            if "address" in applicant:
                property_address = applicant["address"].get("full_address", "")
        
        # If still no address, return empty result
        if not property_address:
            return {
                "status": "not_found",
                "reason": "No property address provided"
            }
        
        # In a real system, we would query a property database
        # For demonstration, we'll use mock data
        property_key = hashlib.md5(property_address.encode()).hexdigest()
        
        # Check if property exists in mock database
        if property_key in self.property_database:
            property_data = self.property_database[property_key]
            return {
                "status": "found",
                "property_data": property_data,
                "source": "property_database"
            }
        else:
            # Generate mock property data
            random.seed(property_key)
            
            property_type = random.choice(["Single Family", "Condo", "Townhouse", "Multi-Family"])
            bedrooms = random.randint(1, 5)
            bathrooms = random.randint(1, 4)
            square_feet = random.randint(800, 3500)
            year_built = random.randint(1950, 2020)
            estimated_value = random.randint(150000, 1500000)
            
            property_data = {
                "address": property_address,
                "property_type": property_type,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "square_feet": square_feet,
                "year_built": year_built,
                "estimated_value": estimated_value,
                "last_sale_date": f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2010, 2022)}",
                "last_sale_price": int(estimated_value * random.uniform(0.7, 0.95))
            }
            
            return {
                "status": "generated",
                "property_data": property_data,
                "source": "estimated_data"
            }
    
    def _verify_employment(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify employment information.
        
        Args:
            applicant: Applicant information
            
        Returns:
            Dict[str, Any]: Employment verification results
        """
        # Extract employment information
        employer_name = ""
        
        # Try to get from financial information
        if "financial_information" in applicant:
            financial_info = applicant["financial_information"]
            if "income_source" in financial_info:
                employer_name = financial_info["income_source"]
        
        # If no employer name, return empty result
        if not employer_name:
            return {
                "status": "not_verified",
                "reason": "No employer information provided"
            }
        
        # In a real system, we would verify with the employer or a verification service
        # For demonstration, we'll use mock data
        employer_key = hashlib.md5(employer_name.encode()).hexdigest()
        
        # Check if employer exists in mock database
        if employer_key in self.employer_database:
            employer_data = self.employer_database[employer_key]
            
            # Check if applicant name matches an employee
            applicant_name = applicant.get("name", "").lower()
            employees = [emp.lower() for emp in employer_data.get("employees", [])]
            
            if any(applicant_name in emp for emp in employees):
                return {
                    "status": "verified",
                    "employer_data": employer_data,
                    "source": "employer_database"
                }
            else:
                return {
                    "status": "not_verified",
                    "reason": "Applicant not found in employer records",
                    "employer_data": employer_data
                }
        else:
            # Generate mock employer data
            random.seed(employer_key)
            
            industry = random.choice(["Technology", "Finance", "Healthcare", "Education", "Retail", "Manufacturing"])
            years_in_business = random.randint(1, 50)
            
            employer_data = {
                "name": employer_name,
                "industry": industry,
                "years_in_business": years_in_business,
                "verification_status": random.choice(["active", "active", "active", "inactive"])
            }
            
            verification_status = "verified" if employer_data["verification_status"] == "active" else "not_verified"
            verification_reason = "Employer verified" if verification_status == "verified" else "Employer not active"
            
            return {
                "status": verification_status,
                "reason": verification_reason,
                "employer_data": employer_data,
                "source": "generated_data"
            }
    
    def _get_credit_bureau_data(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve credit bureau data for the applicant.
        
        Args:
            applicant: Applicant information
            
        Returns:
            Dict[str, Any]: Credit bureau data
        """
        # Extract applicant information
        name = applicant.get("name", "")
        
        # If no name, return empty result
        if not name:
            return {
                "status": "not_found",
                "reason": "No applicant name provided"
            }
        
        # In a real system, we would query a credit bureau
        # For demonstration, we'll use mock data
        applicant_key = hashlib.md5(name.encode()).hexdigest()
        
        # Check if applicant exists in mock database
        if applicant_key in self.credit_bureau_database:
            credit_data = self.credit_bureau_database[applicant_key]
            return {
                "status": "found",
                "credit_data": credit_data,
                "source": "credit_bureau_database"
            }
        else:
            # Generate mock credit data
            random.seed(applicant_key)
            
            credit_score = random.randint(300, 850)
            
            credit_data = {
                "credit_score": credit_score,
                "credit_score_provider": random.choice(["Experian", "TransUnion", "Equifax"]),
                "credit_score_date": datetime.now().isoformat(),
                "accounts": {
                    "total": random.randint(1, 15),
                    "open": random.randint(1, 10),
                    "closed": random.randint(0, 5)
                },
                "inquiries_last_6_months": random.randint(0, 5),
                "derogatory_marks": random.randint(0, 3),
                "credit_utilization": round(random.uniform(0, 0.9), 2),
                "payment_history": {
                    "on_time_payments_percentage": round(random.uniform(0.7, 1.0), 2),
                    "late_payments_30_days": random.randint(0, 3),
                    "late_payments_60_days": random.randint(0, 2),
                    "late_payments_90_days": random.randint(0, 1)
                },
                "public_records": random.randint(0, 1),
                "collections": random.randint(0, 2)
            }
            
            return {
                "status": "generated",
                "credit_data": credit_data,
                "source": "generated_data"
            }
    
    def _get_market_data(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant market data for the application.
        
        Args:
            application_data: Application data
            
        Returns:
            Dict[str, Any]: Market data
        """
        # Extract loan details
        loan_details = application_data.get("loan_details", {})
        loan_purpose = loan_details.get("purpose", "").lower()
        loan_amount = loan_details.get("amount", 0)
        loan_term = loan_details.get("term", 0)
        
        # Determine loan type
        loan_type = "mortgage" if "home" in loan_purpose or "house" in loan_purpose else "personal"
        
        # In a real system, we would query market data sources
        # For demonstration, we'll use mock data
        current_date = datetime.now()
        
        # Generate market data based on loan type
        if loan_type == "mortgage":
            return {
                "interest_rates": {
                    "30_year_fixed": round(random.uniform(3.0, 7.0), 2),
                    "15_year_fixed": round(random.uniform(2.5, 6.5), 2),
                    "5_1_arm": round(random.uniform(2.8, 6.8), 2)
                },
                "housing_market": {
                    "median_home_price": random.randint(200000, 500000),
                    "price_change_yoy": round(random.uniform(-0.05, 0.15), 2),
                    "inventory_level": random.choice(["low", "medium", "high"]),
                    "days_on_market_avg": random.randint(10, 90)
                },
                "as_of_date": current_date.isoformat(),
                "source": "market_data_service"
            }
        else:
            return {
                "interest_rates": {
                    "personal_loan_avg": round(random.uniform(5.0, 15.0), 2),
                    "credit_card_avg": round(random.uniform(12.0, 24.0), 2),
                    "auto_loan_avg": round(random.uniform(3.0, 9.0), 2)
                },
                "consumer_market": {
                    "consumer_confidence_index": random.randint(70, 120),
                    "unemployment_rate": round(random.uniform(3.0, 8.0), 1),
                    "inflation_rate": round(random.uniform(1.0, 8.0), 1)
                },
                "as_of_date": current_date.isoformat(),
                "source": "market_data_service"
            }
    
    def _calculate_derived_metrics(
        self,
        application_data: Dict[str, Any],
        property_info: Dict[str, Any],
        employment_verification: Dict[str, Any],
        credit_bureau_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate derived metrics from enriched data.
        
        Args:
            application_data: Application data
            property_info: Property information
            employment_verification: Employment verification results
            credit_bureau_data: Credit bureau data
            market_data: Market data
            
        Returns:
            Dict[str, Any]: Derived metrics
        """
        derived_metrics = {}
        
        # Extract loan details
        loan_details = application_data.get("loan_details", {})
        loan_amount = loan_details.get("amount", 0)
        loan_term = loan_details.get("term", 0)
        
        # Extract applicant financial information
        applicant = application_data.get("applicant", {})
        financial_info = applicant.get("financial_information", {})
        annual_income = financial_info.get("annual_income", 0)
        
        # Calculate loan-to-value ratio if property information is available
        if property_info.get("status") in ["found", "generated"] and "property_data" in property_info:
            property_data = property_info["property_data"]
            property_value = property_data.get("estimated_value", 0)
            
            if property_value > 0 and loan_amount > 0:
                ltv_ratio = loan_amount / property_value
                derived_metrics["loan_to_value_ratio"] = round(ltv_ratio, 2)
        
        # Calculate debt-to-income ratio if credit bureau data is available
        if credit_bureau_data.get("status") in ["found", "generated"] and "credit_data" in credit_bureau_data:
            credit_data = credit_bureau_data["credit_data"]
            
            # In a real system, we would calculate this from actual debt obligations
            # For demonstration, we'll use a simplified approach
            monthly_debt = 0
            
            # Estimate monthly debt from credit utilization and accounts
            if "credit_utilization" in credit_data and "accounts" in credit_data:
                credit_utilization = credit_data["credit_utilization"]
                open_accounts = credit_data["accounts"].get("open", 0)
                
                # Rough estimate of monthly debt payments
                monthly_debt = credit_utilization * open_accounts * 500
            
            # Calculate monthly income
            monthly_income = annual_income / 12 if annual_income > 0 else 0
            
            if monthly_income > 0:
                dti_ratio = monthly_debt / monthly_income
                derived_metrics["debt_to_income_ratio"] = round(dti_ratio, 2)
        
        # Calculate affordability metrics
        if annual_income > 0:
            # Maximum affordable loan amount (typically 4-5x annual income)
            max_affordable_loan = annual_income * 4.5
            derived_metrics["max_affordable_loan"] = max_affordable_loan
            
            # Affordability ratio (loan amount / max affordable loan)
            if max_affordable_loan > 0:
                affordability_ratio = loan_amount / max_affordable_loan
                derived_metrics["affordability_ratio"] = round(affordability_ratio, 2)
        
        # Calculate estimated interest rate based on credit score and market data
        if credit_bureau_data.get("status") in ["found", "generated"] and "credit_data" in credit_bureau_data:
            credit_data = credit_bureau_data["credit_data"]
            credit_score = credit_data.get("credit_score", 0)
            
            # Get base interest rate from market data
            base_rate = 0
            if "interest_rates" in market_data:
                interest_rates = market_data["interest_rates"]
                if "30_year_fixed" in interest_rates:
                    base_rate = interest_rates["30_year_fixed"]
                elif "personal_loan_avg" in interest_rates:
                    base_rate = interest_rates["personal_loan_avg"]
            
            # Adjust rate based on credit score
            if credit_score >= 740:
                rate_adjustment = -0.5
            elif credit_score >= 700:
                rate_adjustment = -0.25
            elif credit_score >= 660:
                rate_adjustment = 0
            elif credit_score >= 620:
                rate_adjustment = 0.5
            else:
                rate_adjustment = 1.5
            
            estimated_rate = base_rate + rate_adjustment
            derived_metrics["estimated_interest_rate"] = round(estimated_rate, 2)
            
            # Calculate estimated monthly payment
            if loan_amount > 0 and loan_term > 0:
                monthly_rate = estimated_rate / 100 / 12
                num_payments = loan_term
                
                if monthly_rate > 0:
                    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
                    derived_metrics["estimated_monthly_payment"] = round(monthly_payment, 2)
        
        return derived_metrics
    
    def _update_application_data(self, state: Dict[str, Any], enrichment_results: Dict[str, Any]) -> None:
        """
        Update application data with enriched information.
        
        Args:
            state: Current workflow state
            enrichment_results: Enrichment results
        """
        # Get application data
        application_data = state.get("application_data", {})
        
        # Update with property information
        if "property_information" in enrichment_results:
            property_info = enrichment_results["property_information"]
            
            if property_info.get("status") in ["found", "generated"] and "property_data" in property_info:
                property_data = property_info["property_data"]
                
                # Add property information to loan details
                if "loan_details" not in application_data:
                    application_data["loan_details"] = {}
                
                if "property" not in application_data["loan_details"]:
                    application_data["loan_details"]["property"] = {}
                
                # Copy relevant property data
                for field in ["address", "property_type", "bedrooms", "bathrooms", "square_feet", "year_built", "estimated_value"]:
                    if field in property_data:
                        application_data["loan_details"]["property"][field] = property_data[field]
        
        # Update with employment verification
        if "employment_verification" in enrichment_results:
            employment_verification = enrichment_results["employment_verification"]
            
            if "applicant" not in application_data:
                application_data["applicant"] = {}
            
            if "employment" not in application_data["applicant"]:
                application_data["applicant"]["employment"] = {}
            
            # Add verification status
            application_data["applicant"]["employment"]["verification_status"] = employment_verification.get("status", "unknown")
            
            # Add employer data if available
            if "employer_data" in employment_verification:
                employer_data = employment_verification["employer_data"]
                
                for field in ["name", "industry", "years_in_business"]:
                    if field in employer_data:
                        application_data["applicant"]["employment"][field] = employer_data[field]
        
        # Update with credit bureau data
        if "credit_bureau_data" in enrichment_results:
            credit_bureau_data = enrichment_results["credit_bureau_data"]
            
            if credit_bureau_data.get("status") in ["found", "generated"] and "credit_data" in credit_bureau_data:
                credit_data = credit_bureau_data["credit_data"]
                
                if "applicant" not in application_data:
                    application_data["applicant"] = {}
                
                if "credit" not in application_data["applicant"]:
                    application_data["applicant"]["credit"] = {}
                
                # Copy relevant credit data
                for field in ["credit_score", "credit_score_provider", "credit_score_date", "accounts", "inquiries_last_6_months", "derogatory_marks", "credit_utilization"]:
                    if field in credit_data:
                        application_data["applicant"]["credit"][field] = credit_data[field]
        
        # Update with derived metrics
        if "derived_metrics" in enrichment_results:
            derived_metrics = enrichment_results["derived_metrics"]
            
            if "loan_details" not in application_data:
                application_data["loan_details"] = {}
            
            if "derived_metrics" not in application_data["loan_details"]:
                application_data["loan_details"]["derived_metrics"] = {}
            
            # Copy all derived metrics
            for key, value in derived_metrics.items():
                application_data["loan_details"]["derived_metrics"][key] = value
        
        # Update the state with the updated application data
        state["application_data"] = application_data
    
    def _setup_mock_data_sources(self):
        """Set up mock data sources for demonstration purposes."""
        # Property database
        self.property_database = {
            # Example properties
            "123mainst": {
                "address": "123 Main St, Anytown, ST 12345",
                "property_type": "Single Family",
                "bedrooms": 3,
                "bathrooms": 2,
                "square_feet": 1800,
                "year_built": 1985,
                "estimated_value": 350000,
                "last_sale_date": "06/15/2018",
                "last_sale_price": 320000
            },
            "456oakave": {
                "address": "456 Oak Ave, Othertown, ST 67890",
                "property_type": "Condo",
                "bedrooms": 2,
                "bathrooms": 2,
                "square_feet": 1200,
                "year_built": 2005,
                "estimated_value": 275000,
                "last_sale_date": "03/22/2020",
                "last_sale_price": 260000
            }
        }
        
        # Employer database
        self.employer_database = {
            # Example employers
            "acmecorp": {
                "name": "Acme Corporation",
                "industry": "Technology",
                "years_in_business": 15,
                "verification_status": "active",
                "employees": ["John Smith", "Jane Doe", "Bob Johnson"]
            },
            "globexinc": {
                "name": "Globex Inc",
                "industry": "Finance",
                "years_in_business": 25,
                "verification_status": "active",
                "employees": ["Alice Brown", "Charlie Davis", "Eve Wilson"]
            }
        }
        
        # Credit bureau database
        self.credit_bureau_database = {
            # Example credit records
            "johnsmith": {
                "credit_score": 720,
                "credit_score_provider": "Experian",
                "credit_score_date": "2023-01-15T00:00:00Z",
                "accounts": {
                    "total": 8,
                    "open": 5,
                    "closed": 3
                },
                "inquiries_last_6_months": 2,
                "derogatory_marks": 0,
                "credit_utilization": 0.25,
                "payment_history": {
                    "on_time_payments_percentage": 0.98,
                    "late_payments_30_days": 1,
                    "late_payments_60_days": 0,
                    "late_payments_90_days": 0
                },
                "public_records": 0,
                "collections": 0
            },
            "janedoe": {
                "credit_score": 680,
                "credit_score_provider": "TransUnion",
                "credit_score_date": "2023-02-10T00:00:00Z",
                "accounts": {
                    "total": 6,
                    "open": 4,
                    "closed": 2
                },
                "inquiries_last_6_months": 3,
                "derogatory_marks": 1,
                "credit_utilization": 0.35,
                "payment_history": {
                    "on_time_payments_percentage": 0.92,
                    "late_payments_30_days": 2,
                    "late_payments_60_days": 1,
                    "late_payments_90_days": 0
                },
                "public_records": 0,
                "collections": 1
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create the agent
    agent = EnrichmentAgent()
    
    # Example state with application data
    state = {
        "application_data": {
            "applicant": {
                "name": "John Smith",
                "contact_info": {
                    "email": "john.smith@example.com",
                    "phone": "+1234567890"
                },
                "financial_information": {
                    "annual_income": 85000,
                    "income_source": "Acme Corporation"
                }
            },
            "loan_details": {
                "amount": 250000,
                "term": 360,
                "purpose": "Home purchase",
                "property_address": "123 Main St, Anytown, ST 12345"
            }
        }
    }
    
    # Process the state
    updated_state = agent.process(state)
    
    # Print the result
    print(json.dumps(updated_state, indent=2))

