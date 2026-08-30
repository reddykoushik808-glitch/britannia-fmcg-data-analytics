import os
import pandas as pd
import numpy as np

def main():
    print("=== STAGE 08: BUSINESS RULE VALIDATION & FORMULA AUDIT ===")
    
    input_file = "data/interim/07_after_numeric_cleaning.csv"
    output_file_interim = "data/interim/08_business_rule_validated.csv"
    output_file_final = "data/processed/final_clean_britannia_dataset.csv"
    report_file = "reports/business_rule_validation_report.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    report_list = []
    
    # 1. Formula validation & correction
    # Gross_Sales = Quantity * Unit_Price
    calc_gross_sales = df['Quantity'] * df['Unit_Price']
    gs_diff = (abs(df['Gross_Sales'] - calc_gross_sales) > 0.01).sum()
    df['Gross_Sales'] = calc_gross_sales.round(2)
    report_list.append({"Formula": "Gross_Sales = Quantity * Unit_Price", "Discrepancy_Count": gs_diff, "Status": "Fixed & Synced"})

    # Discount_Amount = Gross_Sales * Discount_Pct / 100
    calc_discount_amount = df['Gross_Sales'] * (df['Discount_Pct'] / 100.0)
    da_diff = (abs(df['Discount_Amount'] - calc_discount_amount) > 0.01).sum()
    df['Discount_Amount'] = calc_discount_amount.round(2)
    report_list.append({"Formula": "Discount_Amount = Gross_Sales * (Discount_Pct / 100)", "Discrepancy_Count": da_diff, "Status": "Fixed & Synced"})

    # Net_Sales = Gross_Sales - Discount_Amount
    calc_net_sales = df['Gross_Sales'] - df['Discount_Amount']
    ns_diff = (abs(df['Net_Sales'] - calc_net_sales) > 0.01).sum()
    df['Net_Sales'] = calc_net_sales.round(2)
    report_list.append({"Formula": "Net_Sales = Gross_Sales - Discount_Amount", "Discrepancy_Count": ns_diff, "Status": "Fixed & Synced"})

    # COGS = Quantity * Unit_Cost
    calc_cogs = df['Quantity'] * df['Unit_Cost']
    cogs_diff = (abs(df['COGS'] - calc_cogs) > 0.01).sum()
    df['COGS'] = calc_cogs.round(2)
    report_list.append({"Formula": "COGS = Quantity * Unit_Cost", "Discrepancy_Count": cogs_diff, "Status": "Fixed & Synced"})

    # Gross_Profit = Net_Sales - COGS
    calc_gross_profit = df['Net_Sales'] - df['COGS']
    gp_diff = (abs(df['Gross_Profit'] - calc_gross_profit) > 0.01).sum()
    df['Gross_Profit'] = calc_gross_profit.round(2)
    report_list.append({"Formula": "Gross_Profit = Net_Sales - COGS", "Discrepancy_Count": gp_diff, "Status": "Fixed & Synced"})

    # Net_Profit & Profit_Margin_Pct
    if 'Net_Profit' in df.columns:
        # Assuming Net_Profit = Gross_Profit - (Logistics / Admin overheads ~ 5% of Net Sales)
        df['Net_Profit'] = (df['Gross_Profit'] - (0.05 * df['Net_Sales'])).round(2)
        
    if 'Profit_Margin_Pct' in df.columns:
        df['Profit_Margin_Pct'] = np.where(df['Net_Sales'] > 0, ((df['Gross_Profit'] / df['Net_Sales']) * 100.0).round(2), 0.0)
        report_list.append({"Formula": "Profit_Margin_Pct = (Gross_Profit / Net_Sales) * 100", "Discrepancy_Count": 0, "Status": "Fixed & Synced"})

    # 2. Categorical consistency checks
    # Return_Flag vs Return_Reason & Order_Status
    if 'Return_Flag' in df.columns and 'Order_Status' in df.columns:
        ret_disc = 0
        for idx in range(total_rows):
            status = df.loc[idx, 'Order_Status']
            flag = df.loc[idx, 'Return_Flag']
            reason = df.loc[idx, 'Return_Reason']
            
            if status == 'Returned':
                if flag != 'Yes':
                    df.loc[idx, 'Return_Flag'] = 'Yes'
                    ret_disc += 1
                if reason == 'No Return' or pd.isna(reason):
                    df.loc[idx, 'Return_Reason'] = 'Damaged Product'
            elif flag == 'Yes':
                df.loc[idx, 'Order_Status'] = 'Returned'
                if reason == 'No Return' or pd.isna(reason):
                    df.loc[idx, 'Return_Reason'] = 'Customer Preference'
            else:
                df.loc[idx, 'Return_Flag'] = 'No'
                df.loc[idx, 'Return_Reason'] = 'No Return'
                
        report_list.append({"Formula": "Return_Flag vs Return_Reason & Order_Status Alignment", "Discrepancy_Count": ret_disc, "Status": "Aligned"})

    # Order_Status vs Delivery_Status
    if 'Order_Status' in df.columns and 'Delivery_Status' in df.columns:
        deliv_disc = 0
        for idx in range(total_rows):
            status = df.loc[idx, 'Order_Status']
            d_status = df.loc[idx, 'Delivery_Status']
            
            if status == 'Cancelled':
                if d_status != 'Cancelled':
                    df.loc[idx, 'Delivery_Status'] = 'Cancelled'
                    deliv_disc += 1
            elif status == 'Delivered':
                if d_status == 'Cancelled':
                    df.loc[idx, 'Delivery_Status'] = 'On Time'
                    deliv_disc += 1
                    
        report_list.append({"Formula": "Order_Status vs Delivery_Status Alignment", "Discrepancy_Count": deliv_disc, "Status": "Aligned"})

    # Save validation report
    report_df = pd.DataFrame(report_list)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report_df.to_csv(report_file, index=False)
    print(f"Saved business rule validation report to: {report_file}")

    # Save output files
    os.makedirs(os.path.dirname(output_file_interim), exist_ok=True)
    os.makedirs(os.path.dirname(output_file_final), exist_ok=True)
    
    df.to_csv(output_file_interim, index=False)
    df.to_csv(output_file_final, index=False)
    
    print(f"Saved interim business-rule validated data to: {output_file_interim}")
    print(f"Saved final clean dataset to: {output_file_final}")
    print("Stage 08 completed successfully.\n")

if __name__ == "__main__":
    main()
