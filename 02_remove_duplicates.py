import os
import pandas as pd

def main():
    print("=== STAGE 02: REMOVE DUPLICATES ===")
    
    input_file = "data/raw/01_raw_data.csv"
    output_file = "data/interim/02_after_duplicate_check.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    
    # 1. Exact Duplicate Analysis
    dup_mask = df.duplicated()
    num_exact_dups = dup_mask.sum()
    pct_exact_dups = (num_exact_dups / initial_rows) * 100
    
    print(f"Initial Row Count: {initial_rows}")
    print(f"Exact Duplicate Count: {num_exact_dups}")
    print(f"Exact Duplicate Percentage: {pct_exact_dups:.2f}%")
    
    # Check partial duplicates on key business identifiers (Order_ID)
    if 'Order_ID' in df.columns:
        order_id_dups = df.duplicated(subset=['Order_ID']).sum()
        print(f"Duplicate Order_IDs (Partial/Exact): {order_id_dups}")
        
    # 2. Remove Exact Duplicates
    df_clean = df.drop_duplicates()
    final_rows = len(df_clean)
    rows_removed = initial_rows - final_rows
    
    print(f"Rows Removed: {rows_removed}")
    print(f"Final Row Count after Duplicate Removal: {final_rows}")
    
    # 3. Save to intermediate file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    print(f"Saved deduplicated data to: {output_file}")
    print("Stage 02 completed successfully.\n")

if __name__ == "__main__":
    main()
