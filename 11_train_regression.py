import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def main():
    print("=== STAGE 11: TIME-BASED REGRESSION MODEL TRAINING & EVALUATION ===")
    
    data_file = "data/processed/engineered_features.csv"
    pipeline_file = "models/preprocessing_pipeline.pkl"
    model_output_file = "models/regression_model.pkl"
    report_file = "reports/model_comparison.csv"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Input file '{data_file}' not found.")
        
    df = pd.read_csv(data_file)
    
    # 1. Prepare Target and Non-leaking Predictors
    target_col = "Net_Sales"
    
    num_predictors = [
        'Quantity', 'Unit_Price', 'Discount_Pct', 'Unit_Cost',
        'Customer_Age', 'Delivery_Days', 'Customer_Rating',
        'Inventory_End_Qty', 'Competitor_Price_Index', 'SKU_Size_g',
        'Recency', 'Frequency', 'Year', 'Month'
    ]
    num_predictors = [c for c in num_predictors if c in df.columns]
    
    cat_predictors = [
        'Category', 'Product_Name', 'Channel', 'Payment_Method',
        'Customer_Segment', 'Region', 'State', 'Order_Status',
        'Loyalty_Member', 'RFM_Segment'
    ]
    cat_predictors = [c for c in cat_predictors if c in df.columns]
    
    X = df[num_predictors + cat_predictors]
    y = df[target_col]
    
    # 2. Time-Based Train / Test Split
    # Training: 2023-08-30 to 2025-12-31; Testing: 2026-01-01 to 2026-08-30
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    train_mask = df['Order_Date'] <= pd.Timestamp('2025-12-31')
    test_mask = df['Order_Date'] >= pd.Timestamp('2026-01-01')
    
    X_train_raw, X_test_raw = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training Period: {df[train_mask]['Order_Date'].min().strftime('%Y-%m-%d')} to {df[train_mask]['Order_Date'].max().strftime('%Y-%m-%d')} ({len(X_train_raw)} samples)")
    print(f"Testing Period:  {df[test_mask]['Order_Date'].min().strftime('%Y-%m-%d')} to {df[test_mask]['Order_Date'].max().strftime('%Y-%m-%d')} ({len(X_test_raw)} samples)")
    
    # Load or fit preprocessing pipeline on Training set ONLY
    preprocessor = joblib.load(pipeline_file)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Re-save pipeline fitted strictly on training data
    joblib.dump(preprocessor, pipeline_file)
    
    # 3. Define Candidate Models
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    comparison_results = []
    trained_model_objs = {}
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        comparison_results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 6),
            "MAPE_Pct": round(mape, 4)
        })
        trained_model_objs[name] = model
        print(f"  {name:30s} | R2: {r2:8.5f} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | MAPE: {mape:6.2f}%")
        
    # Save model comparison table
    comparison_df = pd.DataFrame(comparison_results)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    comparison_df.to_csv(report_file, index=False)
    print(f"\nSaved model comparison table to: {report_file}")
    
    # Select Linear Regression as baseline / best interpretability model (and save primary linear regression model)
    best_model_name = "Multiple Linear Regression"
    best_model = trained_model_objs[best_model_name]
    
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
    joblib.dump(best_model, model_output_file)
    print(f"Saved primary trained model ({best_model_name}) to: {model_output_file}")
    print("Stage 11 completed successfully.\n")

if __name__ == "__main__":
    main()
