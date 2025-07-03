import pandas as pd
import re
import os

def species_match(pred, true):
    # Convert both to lowercase and strip spaces
    pred = str(pred).strip().lower()
    true = str(true).strip().lower()
    # Consider "grape" and "grapes" as equivalent.
    if pred == true or (pred in ["grape", "grapes"] and true in ["grape", "grapes"]):
        return True
    return False

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

def multi_step_agri_eval(input_file):
    # Load dataset from the Excel file
    data = pd.read_excel(input_file)

    # Define a function to parse the prediction text into its constituent parts
    def parse_prediction(prediction):
        try:
            # Extract Crop Species from the prediction text
            crop_species_match = re.search(r"1\.\s+Crop species:\s+([^\n]+)", prediction)
            crop_species = crop_species_match.group(1).strip() if crop_species_match else None

            # Extract Fruit Cluster Count from the prediction text
            fruit_count_match = re.search(r"2\.\s+Fruit cluster count:\s+([^\n]+)", prediction)
            fruit_count_text = fruit_count_match.group(1).strip() if fruit_count_match else None
            # Extract numeric value or average if a range is given
            if fruit_count_text:
                numeric_value = extract_numeric(fruit_count_text)
                fruit_count = numeric_value if numeric_value is not None else fruit_count_text
            else:
                fruit_count = None

            # Extract Pest/Disease Infection status (Yes/No)
            pest_infection_match = re.search(r"3\.\s+Pest/Disease infection:\s+(Yes|No)", prediction)
            pest_infection = pest_infection_match.group(1) if pest_infection_match else None

            # Extract Insurance Decision from the prediction text
            decision_match = re.search(r"4\.\s+Insurance decision:\s+(No Compensation|Partial Compensation|Full Compensation)", prediction)
            insurance_decision = decision_match.group(1) if decision_match else None

            return pd.Series([crop_species, fruit_count, pest_infection, insurance_decision])
        except AttributeError:
            return pd.Series([None, None, None, None])

    # Apply the parsing function to the 'prediction' column in the dataset
    data[['Predicted_Crop_Species', 'Predicted_Fruit_Count', 'Predicted_Pest_Infection', 
          'Predicted_Insurance_Decision']] = data['prediction'].apply(parse_prediction)

    # Convert fruit count predictions and ground truth 'count' to numeric if possible
    try:
        data['Predicted_Fruit_Count'] = pd.to_numeric(data['Predicted_Fruit_Count'], errors='coerce')
        data['count'] = pd.to_numeric(data['count'], errors='coerce')
    except Exception as e:
        print("Error converting fruit counts to numeric:", e)

    # Compute correctness for Crop Species using custom matching
    data['Crop_Species_Correct'] = data.apply(
        lambda row: species_match(row['Predicted_Crop_Species'], row['species']), axis=1
    )

    # Compare predicted values with ground truth for Fruit Count, Pest Infection, and Insurance Decision.
    correctness = {
        'Fruit_Count_Correct': data['Predicted_Fruit_Count'] == data['count'],
        'Pest_Infection_Correct': data['Predicted_Pest_Infection'].str.lower() == data['infection'].str.lower(),
        'Insurance_Decision_Correct': data['Predicted_Insurance_Decision'].str.lower() == data['claim'].str.lower()
    }

    # Add the computed correctness columns to the DataFrame
    data = data.assign(**correctness)

    # Compute average differences for numeric predictions (Fruit Count)
    # Only consider rows where 'count' (ground truth fruit count) is non-zero
    valid_fruit_count = data[data['count'] != 0].copy()
    try:
        valid_fruit_count['Fruit_Count_Diff'] = abs(valid_fruit_count['Predicted_Fruit_Count'] - valid_fruit_count['count'])
        avg_fruit_count_diff = valid_fruit_count['Fruit_Count_Diff'].mean()
    except Exception:
        avg_fruit_count_diff = None

    # Calculate correctness rates for each feature
    correct_rates = {
        'Crop_Species_Correct': data['Crop_Species_Correct'].mean(),
        'Fruit_Count_Correct': correctness['Fruit_Count_Correct'].mean(),
        'Pest_Infection_Correct': correctness['Pest_Infection_Correct'].mean(),
        'Insurance_Decision_Correct': correctness['Insurance_Decision_Correct'].mean()
    }

    # Print correctness rates
    print("Correctness Rates:")
    for feature, rate in correct_rates.items():
        print(f"{feature}: {rate:.2%}")

    # Print average fruit count difference (for non-zero true values)
    print("\nAverage Fruit Count Difference (non-zero 'count'):")
    if avg_fruit_count_diff is not None:
        print(f"Average_Fruit_Count_Diff: {avg_fruit_count_diff:.2f}")
    else:
        print("Average_Fruit_Count_Diff: N/A")

    # Construct output file path
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_evaluation{ext}"

    # Save the evaluation results to a new Excel file
    data.to_excel(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")
