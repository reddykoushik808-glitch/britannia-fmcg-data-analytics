import os
import pandas as pd
import numpy as np

def main():
    print("=== STAGE 09: FEATURE ENGINEERING & RFM SEGMENTATION ===")
    
    input_file = "data/processed/final_clean_britannia_dataset.csv"
    output_file = "data/processed/engineered_features.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    
    # 1. Temporal & Calendar Features
    df['Order_Date_DT'] = pd.to_datetime(df['Order_Date'])
    df['Year'] = df['Order_Date_DT'].dt.year
    df['Month'] = df['Order_Date_DT'].dt.month
    df['Quarter'] = df['Order_Date_DT'].dt.quarter
    df['Week'] = df['Order_Date_DT'].dt.isocalendar().week
    df['Day_of_Week'] = df['Order_Date_DT'].dt.dayofweek
    df['Weekend_Flag'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Ratios & Metric Features
    df['Sales_Per_Unit'] = (df['Net_Sales'] / df['Quantity']).round(2)
    df['Profit_Per_Unit'] = (df['Gross_Profit'] / df['Quantity']).round(2)
    df['Profit_Margin'] = df['Profit_Margin_Pct'] if 'Profit_Margin_Pct' in df.columns else ((df['Gross_Profit'] / df['Net_Sales']) * 100).round(2)
    
    # 3. RFM Features (Recency, Frequency, Monetary)
    ref_date = pd.Timestamp("2026-08-30")
    
    # Compute RFM per Customer_ID
    customer_rfm = df.groupby('Customer_ID').agg(
        Last_Order_Date=('Order_Date_DT', 'max'),
        Frequency=('Order_ID', 'nunique'),
        Monetary=('Net_Sales', 'sum'),
        Avg_Order_Value=('Net_Sales', 'mean')
    ).reset_index()
    
    customer_rfm['Recency'] = (ref_date - customer_rfm['Last_Order_Date']).dt.days
    
    # Quantile-based RFM scoring (1-4)
    customer_rfm['R_Score'] = pd.qcut(customer_rfm['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
    customer_rfm['F_Score'] = pd.qcut(customer_rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
    customer_rfm['M_Score'] = pd.qcut(customer_rfm['Monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
    
    def segment_customer(row):
        score = int(row['R_Score']) + int(row['F_Score']) + int(row['M_Score'])
        if score >= 10:
            return 'High Value'
        elif score >= 7:
            return 'Mid Value'
        elif score >= 5:
            return 'Low Value'
        else:
            return 'At Risk'
            
    customer_rfm['RFM_Segment'] = customer_rfm.apply(segment_customer, axis=1)
    
    # Merge RFM features back to main order dataset
    df = df.merge(
        customer_rfm[['Customer_ID', 'Recency', 'Frequency', 'Monetary', 'Avg_Order_Value', 'RFM_Segment']],
        on='Customer_ID',
        how='left'
    )
    
    # Rename Avg_Order_Value to Average_Order_Value and Monetary to Revenue_Per_Customer
    df['Average_Order_Value'] = df['Avg_Order_Value'].round(2)
    df['Revenue_Per_Customer'] = df['Monetary'].round(2)
    df['Customer_Lifetime_Value'] = (df['Average_Order_Value'] * df['Frequency'] * 1.5).round(2)
    
    # Clean temporary helper columns
    df = df.drop(columns=['Order_Date_DT', 'Avg_Order_Value'])
    
    print(f"Engineered Features Added ({len(df.columns)} Total Columns):")
    print("  New Features: Year, Month, Quarter, Week, Day_of_Week, Weekend_Flag, Sales_Per_Unit, Profit_Per_Unit, Recency, Frequency, Revenue_Per_Customer, RFM_Segment, Customer_Lifetime_Value")
    
    # Save engineered dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved engineered feature dataset to: {output_file}")
    print("Stage 09 completed successfully.\n")

if __name__ == "__main__":
    main()
