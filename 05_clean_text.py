import os
import pandas as pd

def main():
    print("=== STAGE 05: TEXT CLEANING & NORMALIZATION ===")
    
    input_file = "data/interim/04_after_missing_value_treatment.csv"
    output_file = "data/interim/05_after_text_cleaning.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    
    # Identify string columns
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"String Columns to Clean ({len(str_cols)}): {str_cols}")
    
    # 1. Strip whitespace across all string columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        
    # 2. Specific text normalization maps
    category_map = {
        'biscuits': 'Biscuits', 'biscuits ': 'Biscuits', 'BISCUITS': 'Biscuits', 'Biscuts': 'Biscuits',
        'Snaks': 'Snacks', 'snacks': 'Snacks', 'SNACKS': 'Snacks',
        'cakes': 'Cakes', 'cakes ': 'Cakes', 'CAKES': 'Cakes',
        'rusk': 'Rusk', 'rusk ': 'Rusk', 'RUSK': 'Rusk',
        'Dairy ': 'Dairy', 'dairy': 'Dairy', 'DAIRY': 'Dairy',
        'bread': 'Bread', 'BREAD': 'Bread'
    }
    
    channel_map = {
        'modern trade': 'Modern Trade', 'modern trade ': 'Modern Trade', 'MODERN TRADE': 'Modern Trade',
        'e-commerce': 'E-commerce', 'e-commerce ': 'E-commerce', 'E-COMMERCE': 'E-commerce',
        'general trade': 'General Trade', 'general trade ': 'General Trade', 'GENERAL TRADE': 'General Trade',
        'wholesale': 'Wholesale', 'wholesale ': 'Wholesale', 'WHOLESALE': 'Wholesale'
    }
    
    payment_map = {
        'Credit Crd': 'Credit Card', 'credit card': 'Credit Card', 'CREDIT CARD': 'Credit Card',
        'upi': 'UPI', 'UPI ': 'UPI', 'Upi': 'UPI',
        'net banking': 'Net Banking', 'debit card': 'Debit Card', 'cash': 'Cash'
    }
    
    order_status_map = {
        'Delivred': 'Delivered', 'delivered': 'Delivered',
        'Cancled': 'Cancelled', 'cancelled': 'Cancelled',
        'Return': 'Returned', 'returned': 'Returned',
        'pending': 'Pending'
    }
    
    # Apply maps if columns exist
    if 'Category' in df.columns:
        df['Category'] = df['Category'].replace(category_map).str.title()
        
    if 'Channel' in df.columns:
        df['Channel'] = df['Channel'].replace(channel_map)
        # Apply standard Title Case for consistency
        df['Channel'] = df['Channel'].apply(lambda x: 'E-commerce' if 'e-commerce' in x.lower() else x.title())
        
    if 'Payment_Method' in df.columns:
        df['Payment_Method'] = df['Payment_Method'].replace(payment_map)
        df['Payment_Method'] = df['Payment_Method'].apply(lambda x: 'UPI' if x.upper() == 'UPI' else x.title())
        
    if 'Order_Status' in df.columns:
        df['Order_Status'] = df['Order_Status'].replace(order_status_map).str.title()
        
    if 'Brand' in df.columns:
        df['Brand'] = df['Brand'].astype(str).str.strip().str.title()
        
    if 'Product_Name' in df.columns:
        df['Product_Name'] = df['Product_Name'].astype(str).str.strip().str.title()
        
    if 'State' in df.columns:
        df['State'] = df['State'].astype(str).str.strip().str.title()
        
    if 'City' in df.columns:
        df['City'] = df['City'].astype(str).str.strip().str.title()
        
    if 'Region' in df.columns:
        df['Region'] = df['Region'].astype(str).str.strip().str.title()
        
    print("\nSample Category Unique Values After Cleaning:")
    if 'Category' in df.columns:
        print(df['Category'].unique())
        
    print("\nSample Payment Method Unique Values After Cleaning:")
    if 'Payment_Method' in df.columns:
        print(df['Payment_Method'].unique())
        
    print("\nSample Order Status Unique Values After Cleaning:")
    if 'Order_Status' in df.columns:
        print(df['Order_Status'].unique())
        
    # 3. Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved text-cleaned data to: {output_file}")
    print("Stage 05 completed successfully.\n")

if __name__ == "__main__":
    main()
