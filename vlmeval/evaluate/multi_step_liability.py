import pandas as pd
import re
import os
import concurrent.futures
from openai import OpenAI

# ---------------- OpenAI API Configuration ----------------
if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
    api_base = os.environ['OPENAI_API_BASE']
else:
    api_base = 'https://api.openai.com/v1/chat/completions'

env_key = os.environ.get('OPENAI_API_KEY', '')

client = OpenAI(
    base_url=api_base,  # Replace with your API endpoint
    api_key= env_key  # Your API key
)

def get_res(messages: list):
    completion = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
    return completion.choices[0].message.content

# ---------------- Accident Cause Consistency Evaluation ----------------
def assess_cause_consistency(gt_cause: str, pred_cause: str) -> str:
    prompt = f"""Given the following ground truth accident cause and the predicted accident cause, determine if they are closely related or consistent.

Ground Truth Accident Cause: {gt_cause}
Predicted Accident Cause: {pred_cause}

Please provide your conclusion in the following format without further explanation:
[Accident Cause Consistency: Consistent] or [Accident Cause Consistency: Not Consistent]"""
    
    messages = [
        {"role": "system", "content": "You are an expert in traffic accident analysis and legal liability."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        answer = get_res(messages)
        return answer.strip()
    except Exception as e:
        return "error"

def assess_cause_consistency_concurrent(pairs: list, max_workers: int = 8):
    """
    Given a list of tuples (gt_cause, pred_cause), concurrently obtain consistency evaluations.
    Returns a dictionary mapping each (gt_cause, pred_cause) tuple to its GPT evaluation result.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(assess_cause_consistency, gt, pred): (gt, pred) for gt, pred in pairs}
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
            except Exception:
                result = "error"
            results[pair] = result
    return results

# ---------------- Helper Function for Direct Matching ----------------
def case_insensitive_equal(pred, true):
    if pd.isna(pred) or pd.isna(true):
        return False
    return str(pred).strip().lower() == str(true).strip().lower()

# ---------------- Parsing Function for Auto Insurance Predictions ----------------
def parse_prediction(prediction):
    try:
        # 1. Extract Weather: (sunny/rainy/snowy/foggy)
        weather_match = re.search(r"1\.\s+Weather:\s+([^\n]+)", prediction)
        weather = weather_match.group(1).strip() if weather_match else None

        # 2. Extract Scene: (highway/tunnel/mountain/urban/rural)
        scene_match = re.search(r"2\.\s+Scene:\s+([^\n]+)", prediction)
        scene = scene_match.group(1).strip() if scene_match else None

        # 3. Extract Linear: (arterials/curve/intersection/T-junction/ramp)
        linear_match = re.search(r"3\.\s+Linear:\s+([^\n]+)", prediction)
        linear = linear_match.group(1).strip() if linear_match else None

        # 4. Extract Accident Occurred: (Yes/No)
        accident_match = re.search(r"4\.\s+Accident Occurred:\s+(Yes|No)", prediction)
        accident_occurred = accident_match.group(1).strip() if accident_match else None

        # 5. Extract Accident Cause: (one sentence brief description)
        cause_match = re.search(r"5\.\s+Accident Cause:\s+([^\n]+)", prediction)
        accident_cause = cause_match.group(1).strip() if cause_match else None

        # 6. Extract Main Responsible Party: (Ego-car, Pedestrian, other car)
        responsible_match = re.search(r"6\.\s+Main Responsible Party:\s+([^\n]+)", prediction)
        responsible_party = responsible_match.group(1).strip() if responsible_match else None

        return pd.Series([weather, scene, linear, accident_occurred, accident_cause, responsible_party])
    except AttributeError:
        return pd.Series([None, None, None, None, None, None])

# ---------------- Main Evaluation Function ----------------
def multi_step_liability_eval(input_file):
    # Load dataset from the Excel file
    data = pd.read_excel(input_file)

    # Apply the parsing function to the 'prediction' column
    data[['Predicted_Weather', 'Predicted_Scene', 'Predicted_Linear',
          'Predicted_Accident_Occurred', 'Predicted_Accident_Cause', 'Predicted_Responsible_Party']] = \
          data['prediction'].apply(parse_prediction)

    # ---------------- Evaluate Accident Cause Consistency via GPT ----------------
    # Create a list of tuples (gt_cause, pred_cause) for rows where both are provided
    cause_pairs = []
    for index, row in data.iterrows():
        gt_cause = row.get("causes")
        pred_cause = row.get("Predicted_Accident_Cause")
        if pd.notna(gt_cause) and pd.notna(pred_cause):
            cause_pairs.append((gt_cause, pred_cause))
    
    if cause_pairs:
        cause_consistency_results = assess_cause_consistency_concurrent(cause_pairs)
    else:
        cause_consistency_results = {}

    def get_consistency(row):
        gt = row.get("causes")
        pred = row.get("Predicted_Accident_Cause")
        if pd.notna(gt) and pd.notna(pred):
            return cause_consistency_results.get((gt, pred), "Not Evaluated")
        return "Not Provided"
    
    data["Accident_Cause_Consistency"] = data.apply(get_consistency, axis=1)

    # Compute accident cause consistency correctness: count as consistent if GPT output indicates "Consistent"
    def is_consistent(consistency_str):
        if pd.isna(consistency_str):
            return False
        text = consistency_str.strip().lower()
        # If the text contains "consistent" and does not contain "not", consider it consistent.
        return "consistent" in text and "not" not in text

    data['Accident_Cause_Consistent'] = data["Accident_Cause_Consistency"].apply(is_consistent)
    
    # Calculate consistency rate over rows where both GT and predicted accident causes are provided
    valid_cause = data[(data["causes"].notna()) & (data["Predicted_Accident_Cause"].notna())]
    if not valid_cause.empty:
        accident_cause_consistency_rate = valid_cause['Accident_Cause_Consistent'].mean()
    else:
        accident_cause_consistency_rate = None

    # ---------------- Direct Matching for Remaining Fields ----------------
    # Ground truth columns: weather, scenes, whether an accident occurred, linear, primary_responsibility
    data['Weather_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Weather'], row['weather']), axis=1)
    data['Scene_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Scene'], row['scenes']), axis=1)
    data['Linear_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Linear'], row['linear']), axis=1)
    data['Accident_Occurred_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Accident_Occurred'], row['whether an accident occurred']), axis=1)
    data['Responsible_Party_Correct'] = data.apply(lambda row: case_insensitive_equal(row['Predicted_Responsible_Party'], row['primary_responsibility']), axis=1)

    # Calculate correctness rates for directly matched fields
    correct_rates = {
        'Weather_Correct': data['Weather_Correct'].mean(),
        'Scene_Correct': data['Scene_Correct'].mean(),
        'Linear_Correct': data['Linear_Correct'].mean(),
        'Accident_Occurred_Correct': data['Accident_Occurred_Correct'].mean(),
        'Responsible_Party_Correct': data['Responsible_Party_Correct'].mean()
    }

    print("Correctness Rates for Directly Matched Fields:")
    for feature, rate in correct_rates.items():
        print(f"{feature}: {rate:.2%}")
    
    if accident_cause_consistency_rate is not None:
        print(f"\nAccident Cause Consistency Rate: {accident_cause_consistency_rate:.2%}")
    else:
        print("\nAccident Cause Consistency Rate: Not Evaluated")

    # Construct output file path
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_evaluation{ext}"

    # Save the evaluation results to a new Excel file
    data.to_excel(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")