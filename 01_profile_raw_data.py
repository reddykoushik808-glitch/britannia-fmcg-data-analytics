import os
import shutil
import pandas as pd
import numpy as np

def main():
    print("=== STAGE 01: PROFILING RAW DATA ===")
    
    # 1. Ensure directory structure exists
    dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src",
        "models",
        "reports"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created/Verified directory: {d}")
        
    raw_source = "britanniaraw.csv"
    raw_target = "data/raw/01_raw_data.csv"
    
    if not os.path.exists(raw_source):
        raise FileNotFoundError(f"Raw source file '{raw_source}' not found.")
        
    shutil.copyfile(raw_source, raw_target)
    print(f"Copied raw dataset to: {raw_target} (Original untouched)")
    
    # 2. Load dataset
    df = pd.read_csv(raw_target)
    
    num_rows, num_cols = df.shape
    print(f"\nDataset Dimensions: {num_rows} rows, {num_cols} columns")
    print(f"Column Names ({num_cols}): {df.columns.tolist()}")
    
    # 3. Missing values check
    missing_series = df.isna().sum()
    missing_cols = missing_series[missing_series > 0]
    print("\n--- Missing Values by Column ---")
    if len(missing_cols) == 0:
        print("No missing values found.")
    else:
        for col, count in missing_cols.items():
            pct = (count / num_rows) * 100
            print(f"  {col:25s}: {count:5d} ({pct:5.2f}%)")
            
    # 4. Duplicate check
    exact_dups = df.duplicated().sum()
    print(f"\nExact Duplicate Rows: {exact_dups} ({(exact_dups/num_rows)*100:.2f}%)")
    
    # 5. Categorical inspect
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\nCategorical Columns ({len(cat_cols)}):")
    for col in ['State', 'City', 'Region', 'Category', 'Product_Name', 'Payment_Method', 'Order_Status']:
        if col in df.columns:
            uniques = df[col].dropna().unique()
            print(f"  {col} ({len(uniques)} unique values): {uniques[:7]}")
            
    # 6. Date inspect
    if 'Order_Date' in df.columns:
        parsed_dates = pd.to_datetime(df['Order_Date'], errors='coerce')
        invalid_dates = parsed_dates.isna().sum()
        min_date = parsed_dates.min()
        max_date = parsed_dates.max()
        print(f"\nOrder_Date Statistics:")
        print(f"  Invalid / Malformed Dates: {invalid_dates}")
        print(f"  Date Range: {min_date} to {max_date}")
        
    # 7. Summary statistics
    num_df = df.select_dtypes(include=[np.number])
    print(f"\nNumerical Statistics Summary ({len(num_df.columns)} columns):")
    print(num_df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
    
    print("\nStage 01 completed successfully.\n")

if __name__ == "__main__":
    main()
