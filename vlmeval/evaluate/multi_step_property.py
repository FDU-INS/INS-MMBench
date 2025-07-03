import pandas as pd
import re
import ast
import os

def extract_numeric(text):
    """
    Extract numeric value from text.
    If two numbers are found (indicating a range), return the average of the first two.
    Otherwise, return the first numeric value found.
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if numbers:
        if len(numbers) >= 2:
            try:
                num1 = float(numbers[0])
                num2 = float(numbers[1])
                return str((num1 + num2) / 2)
            except Exception:
                return numbers[0]
        else:
            return numbers[0]
    return None

def case_insensitive_equal(pred, true):
    if pd.isna(pred) or pd.isna(true):
        return False
    pred, true = str(pred).strip().lower(), str(true).strip().lower()
    
    # Special case: treat "fire" and "wildfire" as equivalent
    if (pred == "fire" and true == "wildfire") or (pred == "wildfire" and true == "fire"):
        return True
    
    return pred == true

def parse_prediction(prediction):
    """Parse prediction text and extract relevant fields."""
    try:
        # Handle cases where prediction is stored as a list of dictionaries
        if prediction.startswith("[") and prediction.endswith("]"):
            try:
                parsed_list = ast.literal_eval(prediction)  # Safely evaluate the list
                if isinstance(parsed_list, list) and len(parsed_list) > 0 and isinstance(parsed_list[0], dict):
                    prediction = parsed_list[0].get("text", "")  # Extract the text field
            except (SyntaxError, ValueError):
                pass  # If parsing fails, treat it as a regular string

        # 1. Extract Disaster Occurred (Yes/No)
        disaster_occ_match = re.search(r"1\.\s+Disaster Occurred:\s+(Yes|No)", prediction)
        disaster_occurred = disaster_occ_match.group(1).strip() if disaster_occ_match else None

        # 2. Extract Disaster Type
        disaster_type_match = re.search(r"2\.\s+Disaster Type:\s+([^\n]+)", prediction)
        disaster_type = disaster_type_match.group(1).strip() if disaster_type_match else None

        # 3. Extract House Count
        house_count_match = re.search(r"3\.\s+House Count:\s+([^\n]+)", prediction)
        house_count_text = house_count_match.group(1).strip() if house_count_match else None
        house_count = extract_numeric(house_count_text) if house_count_text else None

        # 4. Extract Damaged House Count
        damaged_house_match = re.search(r"4\.\s+Damaged House Count:\s+([^\n]+)", prediction)
        damaged_house_text = damaged_house_match.group(1).strip() if damaged_house_match else None
        damaged_house_count = extract_numeric(damaged_house_text) if damaged_house_text else None

        # 5. Extract Insurance Decision
        decision_match = re.search(r"5\.\s+Insurance Decision:\s+(No Compensation|Partial Compensation|Full Compensation)\. Reason:", prediction)
        insurance_decision = decision_match.group(1).strip() if decision_match else None

        return pd.Series([disaster_occurred, disaster_type, house_count, damaged_house_count, insurance_decision])
    except Exception:
        return pd.Series([None, None, None, None, None])

def multi_step_property_eval(input_file):
    data = pd.read_excel(input_file)
    
    # Apply parsing function
    data[['Predicted_Disaster_Occurred', 'Predicted_Disaster_Type', 'Predicted_House_Count', 
          'Predicted_Damaged_House_Count', 'Predicted_Insurance_Decision']] = data['prediction'].apply(parse_prediction)
    
    # Convert numerical values
    for col in ['Predicted_House_Count', 'number', 'Predicted_Damaged_House_Count', 'damage_number']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Compute correctness
    data['Disaster_Occurred_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Disaster_Occurred'], row['disaster']), axis=1)
    data['Disaster_Type_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Disaster_Type'], row['type']), axis=1)
    data['House_Count_Correct'] = data['Predicted_House_Count'] == data['number']
    data['Damaged_House_Count_Correct'] = data['Predicted_Damaged_House_Count'] == data['damage_number']
    data['Insurance_Decision_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Insurance_Decision'], row['claim_decision']), axis=1)
    
    # Compute correctness rates
    correct_rates = {col: data[col].mean() for col in ['Disaster_Occurred_Correct', 'Disaster_Type_Correct', 'House_Count_Correct', 'Damaged_House_Count_Correct', 'Insurance_Decision_Correct']}
    
    # Compute average numeric differences
    avg_house_diff = data.loc[data['number'] != 0, 'Predicted_House_Count'].sub(data['number']).abs().mean()
    avg_damaged_diff = data.loc[data['damage_number'] != 0, 'Predicted_Damaged_House_Count'].sub(data['damage_number']).abs().mean()
    
    # Print correctness rates
    print("Correctness Rates:")
    for feature, rate in correct_rates.items():
        print(f"{feature}: {rate:.2%}")
    
    print("\nAverage House Count Difference:", f"{avg_house_diff:.2f}" if avg_house_diff is not None else "N/A")
    print("Average Damaged House Count Difference:", f"{avg_damaged_diff:.2f}" if avg_damaged_diff is not None else "N/A")
    
    # Save results
    output_file = f"{os.path.splitext(input_file)[0]}_evaluation.xlsx"
    data.to_excel(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")