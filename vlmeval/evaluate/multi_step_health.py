import pandas as pd
import re
import os

def case_insensitive_equal(pred, true):
    if pd.isna(pred) or pd.isna(true):
        return False
    return str(pred).strip().lower() == str(true).strip().lower()

def parse_prediction(prediction):
    try:
        # 1. Extract Scan Region
        scan_region_match = re.search(r"1\.\s+Scan Region:\s+([^\n]+)", prediction)
        scan_region = scan_region_match.group(1).strip() if scan_region_match else None

        # 2. Extract Scan Result Match (Yes/No)
        scan_result_match = re.search(r"2\.\s+Scan Result Match:\s+(Yes|No)", prediction)
        scan_result = scan_result_match.group(1).strip() if scan_result_match else None

        # 3. Extract Health Risk Severity (Low/Moderate/High)
        health_risk_match = re.search(r"3\.\s+Health Risk Severity:\s+(no risk|mild risk|severe risk)", prediction, re.IGNORECASE)
        health_risk = health_risk_match.group(1).strip().lower() if health_risk_match else None

        # 4. Extract Underwriting Evaluation and its Reason
        underwriting_match = re.search(r"4\.\s+Underwriting Evaluation:\s+(Declined|Standard|Higher Premium)\. Reason:\s+([^\n]+)", prediction)
        if underwriting_match:
            underwriting_decision = underwriting_match.group(1).strip()
            underwriting_reason = underwriting_match.group(2).strip()
        else:
            underwriting_decision = None
            underwriting_reason = None

        return pd.Series([scan_region, scan_result, health_risk, underwriting_decision, underwriting_reason])
    except AttributeError:
        return pd.Series([None, None, None, None, None])

def multi_step_health_eval(input_file):
    # Load dataset from the Excel file
    data = pd.read_excel(input_file)

    # Apply the parsing function to the 'prediction' column
    data[['Predicted_Scan_Region', 'Predicted_Scan_Result', 'Predicted_Health_Risk', 
          'Predicted_Underwriting_Decision', 'Predicted_Underwriting_Reason']] = data['prediction'].apply(parse_prediction)

    # Compute correctness for each feature by comparing predictions with ground truth.
    # Ground truth columns: 'scan_region', 'scan_result_match', 'health_risk', 'underwriting_decision'
    data['Scan_Region_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Scan_Region'], row['part']), axis=1)
    data['Scan_Result_Match_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Scan_Result'], row['match']), axis=1)
    data['Health_Risk_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Health_Risk'], row['risk_level']), axis=1)
    data['Underwriting_Decision_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Underwriting_Decision'], row['underwriting_decision']), axis=1)

    # Calculate correctness rates for each feature
    correct_rates = {
        'Scan_Region_Correct': data['Scan_Region_Correct'].mean(),
        'Scan_Result_Match_Correct': data['Scan_Result_Match_Correct'].mean(),
        'Health_Risk_Correct': data['Health_Risk_Correct'].mean(),
        'Underwriting_Decision_Correct': data['Underwriting_Decision_Correct'].mean()
    }

    print("Correctness Rates:")
    for feature, rate in correct_rates.items():
        print(f"{feature}: {rate:.2%}")

    # Construct output file path
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_evaluation{ext}"

    # Save the evaluation results to a new Excel file
    data.to_excel(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")