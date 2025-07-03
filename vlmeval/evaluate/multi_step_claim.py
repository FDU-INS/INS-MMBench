import pandas as pd
import re
import os

def multi_step_claim_eval(input_file):
    data = pd.read_excel(input_file)

    def parse_prediction(prediction):
        try:
            damaged = re.search(r"1\.\s+(Yes|No)", prediction).group(1)
            severity_match = re.search(r"2\.\s+(Minor|Moderate|Severe|N/A)", prediction)
            severity = severity_match.group(1) if severity_match else None
            repair_cost_match = re.search(r"3\.\s+\$(\d+[,.]?\d*|N/A)", prediction)
            repair_cost = float(repair_cost_match.group(1).replace(',', '')) if repair_cost_match and repair_cost_match.group(1) != "N/A" else 0
            claim_eligible_match = re.search(r"4\.\s+(Yes|No|N/A)\.\s+Reason:", prediction)
            claim_eligible = claim_eligible_match.group(1) if claim_eligible_match else None
            final_claim_match = re.search(r"5\.\s+Final Claim Amount:\s+\$(\d+[,.]?\d*|\d+[,.]?\d*)", prediction)
            if final_claim_match:
                final_claim = float(final_claim_match.group(1).replace(',', ''))
            else:
                final_claim_match_direct = re.search(r"5\.\s+\$(\d+[,.]?\d*)", prediction)
                final_claim = float(final_claim_match_direct.group(1).replace(',', '')) if final_claim_match_direct else 0
            return pd.Series([damaged, severity, repair_cost, claim_eligible, final_claim])
        except AttributeError:
            return pd.Series([None, None, 0, None, 0])

    data[['Predicted_Damaged', 'Predicted_Severity', 'Predicted_Repair_Cost', 'Predicted_Claim_Eligible', 'Predicted_Final_Claim']] = data['prediction'].apply(parse_prediction)

    data['Amount'] = data['Amount'].astype(float)
    data['claim'] = data['claim'].astype(float)

    correctness = {
        'Damaged_Correct': data['Predicted_Damaged'] == data['Condition'].apply(lambda x: 'Yes' if x == 1 else 'No'),
        'Severity_Correct': data['Predicted_Severity'].str.lower() == data['Severity'],
        'Claim_Eligible_Correct': data['Predicted_Claim_Eligible'] == data['expired'].apply(lambda x: 'No' if x == 1 else 'Yes')
    }

    data = data.assign(**correctness)

    if data['Predicted_Repair_Cost'].notna().any():
        data['Repair_Cost_Diff'] = abs(data['Predicted_Repair_Cost'] - data['Amount'])
        avg_repair_cost_diff = data['Repair_Cost_Diff'].mean()
    else:
        avg_repair_cost_diff = None

    valid_final_claims = data[(data['Predicted_Final_Claim'] != 0) & (data['claim'] != 0)]
    if not valid_final_claims.empty:
        valid_final_claims['Final_Claim_Diff'] = abs(valid_final_claims['Predicted_Final_Claim'] - valid_final_claims['claim'])
        avg_final_claim_diff = valid_final_claims['Final_Claim_Diff'].mean()
    else:
        avg_final_claim_diff = None

    correct_rates = {key: value.mean() for key, value in correctness.items()}

    print("Correct:")
    for feature, rate in correct_rates.items():
        print(f"{feature}: {rate:.2%}")

    print("\Average Diff:")
    if avg_repair_cost_diff is not None:
        print(f"Average_Repair_Cost_Diff: {avg_repair_cost_diff:.2f}")
    else:
        print("Average_Repair_Cost_Diff: N/A")

    if avg_final_claim_diff is not None:
        print(f"Average_Final_Claim_Diff: {avg_final_claim_diff:.2f}")
    else:
        print("Average_Final_Claim_Diff: N/A")

    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_evaluation{ext}"

    data.to_excel(output_file, index=False)
    print(f"saved to {output_file}")
