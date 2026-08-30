import os
import joblib
import pandas as pd
import numpy as np

def main():
    print("=== STAGE 12: GENERATING REGRESSION EQUATION & COEFFICIENTS ===")
    
    pipeline_file = "models/preprocessing_pipeline.pkl"
    model_file = "models/regression_model.pkl"
    eq_text_file = "reports/regression_equation.txt"
    coef_csv_file = "reports/regression_coefficients.csv"
    
    if not os.path.exists(pipeline_file) or not os.path.exists(model_file):
        raise FileNotFoundError("Pipeline or model file missing.")
        
    preprocessor = joblib.load(pipeline_file)
    model = joblib.load(model_file)
    
    # 1. Extract Feature Names
    num_predictors = preprocessor.transformers_[0][2]
    cat_predictors = preprocessor.transformers_[1][2]
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_predictors).tolist()
    
    feature_names = list(num_predictors) + list(encoded_cat_names)
    
    intercept = model.intercept_
    coefficients = model.coef_
    
    print(f"Intercept (beta_0): {intercept:.4f}")
    print(f"Total Feature Coefficients (beta_1..beta_n): {len(coefficients)}")
    
    # 2. Build Coefficient Table
    coef_data = []
    for name, coef in zip(feature_names, coefficients):
        direction = "Positive (+)" if coef > 0 else ("Negative (-)" if coef < 0 else "Neutral (0)")
        coef_data.append({
            "Feature_Name": name,
            "Coefficient": round(coef, 4),
            "Abs_Importance": round(abs(coef), 4),
            "Direction": direction,
            "Interpretation": f"Holding other variables constant, 1 standard deviation increase in {name} is associated with a {abs(coef):.2f} unit {'increase' if coef > 0 else 'decrease'} in predicted Net Sales."
        })
        
    coef_df = pd.DataFrame(coef_data).sort_values(by="Abs_Importance", ascending=False)
    
    os.makedirs(os.path.dirname(coef_csv_file), exist_ok=True)
    coef_df.to_csv(coef_csv_file, index=False)
    print(f"Saved regression coefficients table to: {coef_csv_file}")
    
    # 3. Formulate Equation Text
    eq_lines = []
    eq_lines.append("=" * 80)
    eq_lines.append("MULTIPLE LINEAR REGRESSION EQUATION FOR BRITANNIA SALES PREDICTION")
    eq_lines.append("=" * 80)
    eq_lines.append(f"\nPredicted_Net_Sales = {intercept:.4f}\n")
    
    for row in coef_data:
        sign = "+" if row["Coefficient"] >= 0 else "-"
        val = abs(row["Coefficient"])
        eq_lines.append(f"  {sign} ({val:.4f} * {row['Feature_Name']})")
        
    eq_lines.append("\n" + "=" * 80)
    eq_lines.append("COEFFICIENT INTERPRETATION GUIDANCE:")
    eq_lines.append("1. Beta_0 (Intercept) = Base predicted net sales when all scaled features are zero.")
    eq_lines.append("2. Positive Coefficient (Beta_i > 0): Increasing this feature increases predicted net sales.")
    eq_lines.append("3. Negative Coefficient (Beta_i < 0): Increasing this feature decreases predicted net sales.")
    eq_lines.append("=" * 80)
    
    eq_text = "\n".join(eq_lines)
    
    os.makedirs(os.path.dirname(eq_text_file), exist_ok=True)
    with open(eq_text_file, "w", encoding="utf-8") as f:
        f.write(eq_text)
        
    print(f"Saved regression equation text to: {eq_text_file}")
    print("\nSample Regression Equation Output (Top 5 features):")
    print("\n".join(eq_lines[:12]))
    print("Stage 12 completed successfully.\n")

if __name__ == "__main__":
    main()
