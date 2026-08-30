import os
import joblib
import pandas as pd
import numpy as np

def predict_future_sales(scenarios_df, pipeline_path="models/preprocessing_pipeline.pkl", model_path="models/regression_model.pkl"):
    """
    Accepts a DataFrame of future commercial scenarios, applies the fitted 
    preprocessing pipeline, and generates Net_Sales predictions.
    """
    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or Pipeline files not found. Run training scripts first.")
        
    preprocessor = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    
    # Preprocess business inputs
    X_processed = preprocessor.transform(scenarios_df)
    
    # Predict sales
    predictions = model.predict(X_processed)
    
    output_df = scenarios_df.copy()
    output_df['Predicted_Net_Sales'] = np.round(predictions, 2)
    return output_df

def main():
    print("=== STAGE 13: FUTURE SALES PREDICTION SCENARIOS ===")
    
    output_file = "reports/future_sales_predictions.csv"
    
    # Define 5 hypothetical future business scenarios
    scenarios = [
        {
            "Scenario_ID": "Scenario_01_HighVolume_Biscuits_ModernTrade",
            "Category": "Biscuits",
            "Product_Name": "Good Day",
            "Quantity": 100.0,
            "Unit_Price": 45.0,
            "Discount_Pct": 10.0,
            "Unit_Cost": 30.0,
            "Channel": "Modern Trade",
            "Payment_Method": "UPI",
            "Customer_Segment": "Family",
            "Region": "South",
            "State": "Karnataka",
            "Order_Status": "Delivered",
            "Loyalty_Member": "Yes",
            "RFM_Segment": "High Value",
            "Customer_Age": 38.0,
            "Delivery_Days": 3.0,
            "Customer_Rating": 4.5,
            "Inventory_End_Qty": 500.0,
            "Competitor_Price_Index": 1.02,
            "SKU_Size_g": 150.0,
            "Recency": 12.0,
            "Frequency": 8.0,
            "Year": 2026,
            "Month": 9
        },
        {
            "Scenario_ID": "Scenario_02_Premium_Snacks_Ecommerce",
            "Category": "Snacks",
            "Product_Name": "Cheez Bit",
            "Quantity": 50.0,
            "Unit_Price": 120.0,
            "Discount_Pct": 5.0,
            "Unit_Cost": 75.0,
            "Channel": "E-commerce",
            "Payment_Method": "Credit Card",
            "Customer_Segment": "Premium",
            "Region": "West",
            "State": "Maharashtra",
            "Order_Status": "Delivered",
            "Loyalty_Member": "Yes",
            "RFM_Segment": "High Value",
            "Customer_Age": 29.0,
            "Delivery_Days": 2.0,
            "Customer_Rating": 4.8,
            "Inventory_End_Qty": 250.0,
            "Competitor_Price_Index": 1.15,
            "SKU_Size_g": 200.0,
            "Recency": 5.0,
            "Frequency": 12.0,
            "Year": 2026,
            "Month": 10
        },
        {
            "Scenario_ID": "Scenario_03_Bulk_Rusk_Wholesale",
            "Category": "Rusk",
            "Product_Name": "Toastea",
            "Quantity": 150.0,
            "Unit_Price": 35.0,
            "Discount_Pct": 15.0,
            "Unit_Cost": 22.0,
            "Channel": "Wholesale",
            "Payment_Method": "Net Banking",
            "Customer_Segment": "Mass",
            "Region": "North",
            "State": "Uttar Pradesh",
            "Order_Status": "Delivered",
            "Loyalty_Member": "No",
            "RFM_Segment": "Mid Value",
            "Customer_Age": 45.0,
            "Delivery_Days": 5.0,
            "Customer_Rating": 4.0,
            "Inventory_End_Qty": 1000.0,
            "Competitor_Price_Index": 0.95,
            "SKU_Size_g": 250.0,
            "Recency": 25.0,
            "Frequency": 4.0,
            "Year": 2026,
            "Month": 11
        },
        {
            "Scenario_ID": "Scenario_04_Health_Biscuits_GeneralTrade",
            "Category": "Biscuits",
            "Product_Name": "NutriChoice",
            "Quantity": 40.0,
            "Unit_Price": 80.0,
            "Discount_Pct": 8.0,
            "Unit_Cost": 50.0,
            "Channel": "General Trade",
            "Payment_Method": "Cash",
            "Customer_Segment": "Health",
            "Region": "East",
            "State": "West Bengal",
            "Order_Status": "Delivered",
            "Loyalty_Member": "Yes",
            "RFM_Segment": "Mid Value",
            "Customer_Age": 42.0,
            "Delivery_Days": 4.0,
            "Customer_Rating": 4.3,
            "Inventory_End_Qty": 300.0,
            "Competitor_Price_Index": 1.05,
            "SKU_Size_g": 100.0,
            "Recency": 18.0,
            "Frequency": 6.0,
            "Year": 2026,
            "Month": 9
        },
        {
            "Scenario_ID": "Scenario_05_Festive_Cakes_Ecommerce",
            "Category": "Cakes",
            "Product_Name": "Fruit Cake",
            "Quantity": 80.0,
            "Unit_Price": 150.0,
            "Discount_Pct": 12.0,
            "Unit_Cost": 90.0,
            "Channel": "E-commerce",
            "Payment_Method": "UPI",
            "Customer_Segment": "Family",
            "Region": "Central",
            "State": "Madhya Pradesh",
            "Order_Status": "Delivered",
            "Loyalty_Member": "Yes",
            "RFM_Segment": "High Value",
            "Customer_Age": 34.0,
            "Delivery_Days": 2.0,
            "Customer_Rating": 4.7,
            "Inventory_End_Qty": 400.0,
            "Competitor_Price_Index": 1.10,
            "SKU_Size_g": 300.0,
            "Recency": 7.0,
            "Frequency": 10.0,
            "Year": 2026,
            "Month": 12
        }
    ]
    
    scenarios_df = pd.DataFrame(scenarios)
    results_df = predict_future_sales(scenarios_df)
    
    print("\n--- Future Sales Predictions for 5 Business Scenarios ---")
    for idx, row in results_df.iterrows():
        print(f"[{row['Scenario_ID']}]")
        print(f"  Qty: {row['Quantity']} | Unit Price: Rs. {row['Unit_Price']} | Disc: {row['Discount_Pct']}% | Channel: {row['Channel']}")
        print(f"  -> Predicted Net Sales: Rs. {row['Predicted_Net_Sales']:,.2f}\n")
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Saved scenario predictions to: {output_file}")
    print("Stage 13 completed successfully.\n")

if __name__ == "__main__":
    main()
