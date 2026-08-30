import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def main():
    print("=== STAGE 10: FEATURE SCALING & PREPROCESSING PIPELINE ===")
    
    input_file = "data/processed/engineered_features.csv"
    output_scaled = "data/processed/scaled_features.csv"
    output_pipeline = "models/preprocessing_pipeline.pkl"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    
    # 1. Define Target and Features
    target_col = "Net_Sales"
    if target_col not in df.columns:
        target_col = "Gross_Sales"
        
    # Numerical predictors
    num_predictors = [
        'Quantity', 'Unit_Price', 'Discount_Pct', 'Unit_Cost',
        'Customer_Age', 'Delivery_Days', 'Customer_Rating',
        'Inventory_End_Qty', 'Competitor_Price_Index', 'SKU_Size_g',
        'Recency', 'Frequency', 'Year', 'Month'
    ]
    num_predictors = [c for c in num_predictors if c in df.columns]
    
    # Categorical predictors
    cat_predictors = [
        'Category', 'Product_Name', 'Channel', 'Payment_Method',
        'Customer_Segment', 'Region', 'State', 'Order_Status',
        'Loyalty_Member', 'RFM_Segment'
    ]
    cat_predictors = [c for c in cat_predictors if c in df.columns]
    
    print(f"Target Variable: {target_col}")
    print(f"Numerical Predictors ({len(num_predictors)}): {num_predictors}")
    print(f"Categorical Predictors ({len(cat_predictors)}): {cat_predictors}")
    
    X = df[num_predictors + cat_predictors]
    y = df[target_col]
    
    # 2. Build ColumnTransformer Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_predictors),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_predictors)
        ]
    )
    
    # Fit and transform features
    X_scaled_array = preprocessor.fit_transform(X)
    
    # Retrieve feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_predictors).tolist()
    all_feature_names = num_predictors + encoded_cat_names
    
    scaled_df = pd.DataFrame(X_scaled_array, columns=all_feature_names)
    scaled_df[target_col] = y.values
    
    print(f"Total Scaled Feature Dimension: {scaled_df.shape[1] - 1} predictor columns")
    
    # 3. Save pipeline and scaled features
    os.makedirs(os.path.dirname(output_pipeline), exist_ok=True)
    os.makedirs(os.path.dirname(output_scaled), exist_ok=True)
    
    joblib.dump(preprocessor, output_pipeline)
    print(f"Saved fitted preprocessing pipeline to: {output_pipeline}")
    
    scaled_df.to_csv(output_scaled, index=False)
    print(f"Saved scaled feature dataset to: {output_scaled}")
    print("Stage 10 completed successfully.\n")

if __name__ == "__main__":
    main()
