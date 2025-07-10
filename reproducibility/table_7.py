import pandas as pd
import os

model_list = [
    "GPT4o",
    "QwenVLMax",
    "Gemini1_5Flash"
]


dataset_list = [
    "INS-MMBench_claude",
    "INS-MMBench_ds"
]

target_columns = [
    "driving behavior detection",
    "medical image recognition",
    "vehicle appearance recognition"
]

results = []

for model in model_list:
    for dataset in dataset_list:
        folder = model
        csv_filename = f"{model}_{dataset}_acc.csv"
        csv_path = os.path.join(folder, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        row = df.loc[df["split"] == "none"].iloc[0]
        
        row_data = {
            "Model": model,
            "Dataset": dataset
        }
        for col in target_columns:
            row_data[col] = row.get(col, None)
        
        results.append(row_data)

summary_df = pd.DataFrame(results)

print(summary_df)

# Note that the output is not directly a table. Instead, the output is the source result for generating the table in latex. 
# To generate the table, please replace the result in the following latex code.

# \begin{table}[t]
# \centering
# \caption{Results of different distractor generation models.}
# \label{tab:different_distractor}
# \vspace{-3mm}
# \resizebox{0.45\textwidth}{!}{%
# \begin{tabular}{cccccccccc}
# \toprule
# \textbf{Task} &
# \multicolumn{3}{c}{\textbf{GPT4o}} &
# \multicolumn{3}{c}{\textbf{QwenVLMax}} &
# \multicolumn{3}{c}{\textbf{Gemini-1.5 -Flash}} \\
# \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
#  & origin & claude & deepseek & origin & claude & deepseek & origin & claude & deepseek \\
# \midrule
# Driving behavior detection & 88.60 & 85.40 & 86.60 & 74.80 & 75.40 & 76.00 & 79.20 & 80.60 & 78.60 \\
# Medical image recognition & 49.00 & 50.20 & 46.60 & 44.00 & 41.40 & 40.60 & 46.40 & 43.20 & 44.60 \\
# Vehicle appearance recognition & 98.20 & 93.00 & 96.00 & 98.20 & 92.60 & 95.20 & 96.80 & 91.20 & 93.40 \\
# \bottomrule
# \end{tabular}
# }
# \end{table}
