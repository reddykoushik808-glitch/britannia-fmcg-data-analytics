import os
import pandas as pd
import numpy as np

def main():
    print("=== STAGE 07: NUMERIC DATA CLEANING & QUALITY AUDIT ===")
    
    input_file = "data/interim/06_after_geography_validation.csv"
    output_file = "data/interim/07_after_numeric_cleaning.csv"
    report_file = "reports/numeric_quality_report.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    report_data = []
    
    # 1. Quantity Cleaning
    q_neg = (df['Quantity'] < 0).sum()
    q_outlier = (df['Quantity'] > 200).sum() # e.g. 9999
    
    # Fix negative quantity -> absolute value
    df['Quantity'] = df['Quantity'].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
    # Fix extreme outlier (e.g. 9999) -> product group median
    q_median = df[df['Quantity'] <= 200]['Quantity'].median()
    df['Quantity'] = df['Quantity'].apply(lambda x: q_median if pd.notna(x) and x > 200 else x)
    
    report_data.append({"Metric": "Quantity", "Negative_Count": q_neg, "Outlier_Count": q_outlier, "Action": "Abs value for negative, median for >200"})

    # 2. Unit_Price Cleaning
    up_neg = (df['Unit_Price'] <= 0).sum()
    # Replace negative or zero price with Product_Name median
    if 'Product_Name' in df.columns:
        valid_prices = df[df['Unit_Price'] > 0]
        prod_price_medians = valid_prices.groupby('Product_Name')['Unit_Price'].median()
        df['Unit_Price'] = df.apply(
            lambda r: prod_price_medians.get(r['Product_Name'], 120.0) if r['Unit_Price'] <= 0 or pd.isna(r['Unit_Price']) else r['Unit_Price'],
            axis=1
        )
    report_data.append({"Metric": "Unit_Price", "Negative_Count": up_neg, "Outlier_Count": 0, "Action": "Imputed positive price using Product_Name median"})

    # 3. Discount_Pct Cleaning
    disc_neg = (df['Discount_Pct'] < 0).sum()
    disc_outlier = (df['Discount_Pct'] > 50).sum() # e.g. 999%
    
    df['Discount_Pct'] = df['Discount_Pct'].apply(lambda x: 0.0 if pd.notna(x) and x < 0 else x)
    df['Discount_Pct'] = df['Discount_Pct'].apply(lambda x: 10.0 if pd.notna(x) and x > 50 else x) # cap at 10% median
    
    report_data.append({"Metric": "Discount_Pct", "Negative_Count": disc_neg, "Outlier_Count": disc_outlier, "Action": "Set negative to 0%, capped >50% to median 10%"})

    # 4. Delivery_Days Cleaning
    dd_neg = (df['Delivery_Days'] < 0).sum()
    dd_outlier = (df['Delivery_Days'] > 30).sum()
    
    df['Delivery_Days'] = df['Delivery_Days'].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
    df['Delivery_Days'] = df['Delivery_Days'].apply(lambda x: 4.0 if pd.notna(x) and x > 30 else x) # median 4 days
    
    report_data.append({"Metric": "Delivery_Days", "Negative_Count": dd_neg, "Outlier_Count": dd_outlier, "Action": "Abs value for negative, capped >30 to 4 days"})

    # 5. Customer_Age Cleaning
    age_neg = (df['Customer_Age'] < 0).sum()
    age_outlier = ((df['Customer_Age'] > 100) | (df['Customer_Age'] < 10)).sum()
    
    median_age = df[(df['Customer_Age'] >= 10) & (df['Customer_Age'] <= 100)]['Customer_Age'].median()
    df['Customer_Age'] = df['Customer_Age'].apply(lambda x: median_age if pd.isna(x) or x < 10 or x > 100 else x)
    
    report_data.append({"Metric": "Customer_Age", "Negative_Count": age_neg, "Outlier_Count": age_outlier, "Action": "Imputed invalid age (<10 or >100) with median age"})

    # 6. Customer_Rating Cleaning
    rat_inv = ((df['Customer_Rating'] < 1) | (df['Customer_Rating'] > 5)).sum()
    df['Customer_Rating'] = df['Customer_Rating'].clip(lower=1.0, upper=5.0)
    
    report_data.append({"Metric": "Customer_Rating", "Negative_Count": 0, "Outlier_Count": rat_inv, "Action": "Clipped ratings to valid range [1.0, 5.0]"})

    # 7. Reorder_Level Cleaning
    if 'Reorder_Level' in df.columns:
        ro_neg = (df['Reorder_Level'] < 0).sum()
        df['Reorder_Level'] = df['Reorder_Level'].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
        report_data.append({"Metric": "Reorder_Level", "Negative_Count": ro_neg, "Outlier_Count": 0, "Action": "Converted negative reorder levels to positive"})

    # Save numeric quality report
    report_df = pd.DataFrame(report_data)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report_df.to_csv(report_file, index=False)
    print(f"Saved numeric quality report to: {report_file}")

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved numeric-cleaned data to: {output_file}")
    print("Stage 07 completed successfully.\n")

if __name__ == "__main__":
    main()
