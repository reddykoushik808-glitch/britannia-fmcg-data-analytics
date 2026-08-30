import os
import pandas as pd

def main():
    print("=== STAGE 06: GEOGRAPHY VALIDATION & CORRECTION ===")
    
    input_file = "data/interim/05_after_text_cleaning.csv"
    output_file = "data/interim/06_after_geography_validation.csv"
    report_file = "reports/geography_validation_report.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    df = pd.read_csv(input_file)
    total_rows = len(df)
    
    state_city_mapping = {
        "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
        "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
        "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Varanasi"],
        "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
        "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
        "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala"],
        "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati"],
        "Madhya Pradesh": ["Indore", "Bhopal", "Gwalior", "Jabalpur"],
        "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri"],
        "Delhi": ["New Delhi", "Dwarka", "Rohini", "Saket"]
    }
    
    state_region_mapping = {
        "Telangana": "South", "Karnataka": "South", "Tamil Nadu": "South", "Kerala": "South", "Andhra Pradesh": "South",
        "Maharashtra": "West", "Gujarat": "West",
        "West Bengal": "East", "Odisha": "East",
        "Uttar Pradesh": "North", "Rajasthan": "North", "Punjab": "North", "Haryana": "North", "Delhi": "North",
        "Madhya Pradesh": "Central"
    }
    
    # Invert city to state lookup dictionary (case-insensitive)
    city_to_state = {}
    for st, cities in state_city_mapping.items():
        for c in cities:
            city_to_state[c.lower()] = st
            
    initial_valid = 0
    mismatch_count = 0
    unresolved_count = 0
    
    corrected_states = []
    corrected_cities = []
    corrected_regions = []
    status_list = []
    
    for idx, row in df.iterrows():
        raw_st = str(row['State']).strip() if pd.notna(row['State']) else 'Unknown'
        raw_ct = str(row['City']).strip() if pd.notna(row['City']) else 'Unknown'
        raw_reg = str(row['Region']).strip() if pd.notna(row['Region']) else 'Unknown'
        
        ct_key = raw_ct.lower()
        
        if ct_key in city_to_state:
            valid_st = city_to_state[ct_key]
            valid_reg = state_region_mapping.get(valid_st, "Unknown")
            
            if raw_st == valid_st:
                initial_valid += 1
                status_list.append("Valid")
            else:
                mismatch_count += 1
                status_list.append("Mismatched & Corrected")
                
            corrected_states.append(valid_st)
            corrected_cities.append(raw_ct.title())
            corrected_regions.append(valid_reg)
        else:
            unresolved_count += 1
            status_list.append("Unresolved / Unknown")
            corrected_states.append("Unknown")
            corrected_cities.append("Unknown")
            corrected_regions.append("Unknown")
            
    df['State'] = corrected_states
    df['City'] = corrected_cities
    df['Region'] = corrected_regions
    df['Geography_Quality_Status'] = status_list
    
    print(f"Total Records Evaluated: {total_rows}")
    print(f"Initial Valid State-City Combinations: {initial_valid} ({(initial_valid/total_rows)*100:.2f}%)")
    print(f"Mismatched State-City Combinations Corrected: {mismatch_count} ({(mismatch_count/total_rows)*100:.2f}%)")
    print(f"Unresolved / Unknown Geography Combinations: {unresolved_count} ({(unresolved_count/total_rows)*100:.2f}%)")
    
    # Generate geography validation report
    geo_report_df = pd.DataFrame([{
        "Total_Records": total_rows,
        "Initially_Valid_Count": initial_valid,
        "Mismatched_Corrected_Count": mismatch_count,
        "Unresolved_Unknown_Count": unresolved_count,
        "Final_Quality_Percentage": round(((initial_valid + mismatch_count) / total_rows) * 100, 2)
    }])
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    geo_report_df.to_csv(report_file, index=False)
    print(f"Saved geography validation report to: {report_file}")
    
    # Save clean dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved geography-validated data to: {output_file}")
    print("Stage 06 completed successfully.\n")

if __name__ == "__main__":
    main()
