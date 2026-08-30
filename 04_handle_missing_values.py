import os
import pandas as pd
import numpy as np

def main():
    print("=== STAGE 04: MISSING VALUE ANALYSIS & TREATMENT ===")
    
    input_file = "data/interim/03_after_date_cleaning.csv"
    output_file = "data/interim/04_after_missing_value_treatment.csv"
    report_file = "reports/missing_value_report.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    # 1. Generate Missing Value Report
    missing_data = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / total_rows) * 100
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "Numerical"
        elif "date" in col.lower():
            col_type = "Date"
        elif "id" in col.lower():
            col_type = "Identifier"
        else:
            col_type = "Categorical"
            
        missing_data.append({
            "Column_Name": col,
            "Column_Type": col_type,
            "Missing_Count": null_count,
            "Missing_Percentage": round(null_pct, 4)
        })
        
    report_df = pd.DataFrame(missing_data)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report_df.to_csv(report_file, index=False)
    print(f"Saved missing value report to: {report_file}")
    
    # 2. Treat Missing Values
    
    # Specific categorical fields that indicate absence of event
    if 'Return_Reason' in df.columns:
        df['Return_Reason'] = df['Return_Reason'].fillna('No Return')
        
    if 'Promotion_Type' in df.columns:
        df['Promotion_Type'] = df['Promotion_Type'].fillna('No Promotion')
        
    # General categorical columns: impute using mode within Category or 'Unknown'
    cat_cols = ['State', 'City', 'Payment_Method', 'Product_Name']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    # Numerical columns: Impute missing values with group median (by Product_Name or Category if possible) or overall median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in num_cols:
        if df[col].isna().sum() > 0:
            if 'Product_Name' in df.columns and df['Product_Name'].nunique() > 1:
                # Fill missing with median per product name
                group_medians = df.groupby('Product_Name')[col].transform('median')
                df[col] = df[col].fillna(group_medians)
                
            if 'Category' in df.columns and df[col].isna().sum() > 0:
                group_medians_cat = df.groupby('Category')[col].transform('median')
                df[col] = df[col].fillna(group_medians_cat)
                
            # Overall median fallback
            overall_median = df[col].median()
            df[col] = df[col].fillna(overall_median)
            
    remaining_missing = df.isna().sum().sum()
    print(f"Remaining Missing Values Across Dataset: {remaining_missing}")
    
    # 3. Save treated output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved missing-value treated data to: {output_file}")
    print("Stage 04 completed successfully.\n")

if __name__ == "__main__":
    main()
