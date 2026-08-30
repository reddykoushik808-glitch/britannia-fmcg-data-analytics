import os
import pandas as pd

def main():
    print("=== STAGE 03: DATE CLEANING ===")
    
    input_file = "data/interim/02_after_duplicate_check.csv"
    output_file = "data/interim/03_after_date_cleaning.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    
    # 1. Parse Order_Date
    raw_dates = df['Order_Date'].copy()
    coerced_dates = pd.to_datetime(raw_dates, errors='coerce')
    
    missing_count = raw_dates.isna().sum()
    invalid_format_count = coerced_dates.isna().sum() - missing_count
    total_unparsed = coerced_dates.isna().sum()
    
    print(f"Total Rows: {initial_rows}")
    print(f"Missing Date Strings: {missing_count}")
    print(f"Invalid / Malformed Date Strings: {invalid_format_count}")
    print(f"Total Unparsed Dates (NaT): {total_unparsed}")
    
    # Check date range boundaries (Expected: 2023-08-30 to 2026-08-30)
    min_expected = pd.Timestamp("2023-08-30")
    max_expected = pd.Timestamp("2026-08-30")
    
    valid_dates_mask = coerced_dates.notna()
    out_of_range_before = (coerced_dates[valid_dates_mask] < min_expected).sum()
    out_of_range_after = (coerced_dates[valid_dates_mask] > max_expected).sum()
    
    print(f"Valid Dates Range: {coerced_dates.min()} to {coerced_dates.max()}")
    print(f"Dates before expected range ({min_expected.strftime('%Y-%m-%d')}): {out_of_range_before}")
    print(f"Dates after expected range ({max_expected.strftime('%Y-%m-%d')}): {out_of_range_after}")
    
    # 2. Impute Invalid / Missing Dates
    # If Order_Date is missing or invalid, impute using forward fill / backward fill sorted by Order_ID / Index
    df['Order_Date_Parsed'] = coerced_dates
    
    # Sort temporarily by Order_ID if available to preserve chronological context
    if 'Order_ID' in df.columns:
        df = df.sort_values(by='Order_ID').reset_index(drop=True)
        
    # Forward fill then backward fill NaT values
    df['Order_Date_Parsed'] = df['Order_Date_Parsed'].ffill().bfill()
    
    # Clip any dates strictly to expected boundaries if needed
    df['Order_Date_Parsed'] = df['Order_Date_Parsed'].clip(lower=min_expected, upper=max_expected)
    
    # Replace original Order_Date with clean string format YYYY-MM-DD
    df['Order_Date'] = df['Order_Date_Parsed'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['Order_Date_Parsed'])
    
    print(f"After Treatment: Invalid/Missing Date Count = {df['Order_Date'].isna().sum()}")
    print(f"Final Cleaned Date Range: {df['Order_Date'].min()} to {df['Order_Date'].max()}")
    
    # 3. Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved date-cleaned data to: {output_file}")
    print("Stage 03 completed successfully.\n")

if __name__ == "__main__":
    main()
