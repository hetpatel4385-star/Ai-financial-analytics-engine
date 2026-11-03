import os
import pandas as pd
import anthropic

# Initialize Anthropic Claude Client
client = anthropic.Anthropic()

def load_financial_data(csv_path: str) -> pd.DataFrame:
    """Load transaction data into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'])
    return df

def calculate_summary_metrics(df: pd.DataFrame) -> dict:
    """Calculate key financial metrics across spending categories."""
    total_spend = df[df['Amount'] > 0]['Amount'].sum()
    category_summary = df.groupby('Category')['Amount'].sum().to_dict()
    top_category = max(category_summary, key=category_summary.get)
    
    return {
        "total_spending": round(total_spend, 2),
        "category_breakdown": category_summary,
        "highest_spending_category": top_category
    }

def generate_claude_financial_insights(metrics: dict) -> str:
    """Send structured summary metrics to Claude API for automated risk & budgeting analysis."""
    prompt = f"""
    You are an automated financial risk analyst. Analyze the following monthly spending summary and provide:
    1. A brief executive breakdown of spending patterns.
    2. Primary budget risk areas or anomalies.
    3. Three actionable recommendations for cost optimization.

    Financial Data Summary:
    {metrics}
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

if __name__ == "__main__":
    sample_data_path = "transactions.csv"
    
    if os.path.exists(sample_data_path):
        data = load_financial_data(sample_data_path)
        summary = calculate_summary_metrics(data)
        
        print("=== Quantitative Spending Metrics ===")
        print(summary)
        
        print("\n=== Claude API Automated Insights ===")
        insights = generate_claude_financial_insights(summary)
        print(insights)
    else:
        print(f"Please provide a valid CSV file at {sample_data_path}")
