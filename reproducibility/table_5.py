import subprocess
import pandas as pd
import re

model_list = [
    "GPT4o",
    "QwenVLMax",
    "Gemini1_5Flash"
]

dataset_list = [
    "multi_step_claim",
    "multi_step_liability",
    "multi_step_health",
    "multi_step_property",
    "multi_step_agri"
]

all_records = []

for model in model_list:
    for dataset in dataset_list:
        print(f"Running model={model} on dataset={dataset}")
        
        cmd = [
            "python",
            "run.py",
            "--data", dataset,
            "--model", model,
            "--verbose",
            "--nproc", "4"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        
        output = result.stdout


# Note that the output is not directly a table. Instead, the output is the source result for generating the table in latex. 
# To generate the table, please replace the result in the following latex code.

# \begin{table}
#     \centering
#     \caption{Evaluation results of the LVLMs across different scenario tasks. The evaluation metrics include two types: (1) Accuracy (\%) – metrics evaluating correctness in percentage format, and (2) Difference (Diff) – numerical metrics representing the average deviation between true and predicted values. Rows with a \colorbox{yellow!20}{yellow} background indicate the final insurance decision metric for each scenario.}
#     \label{tab:insurance_tasks}
#     \resizebox{0.48\textwidth}{!}{ 
#     \begin{tabular}{p{4.5cm} l c c c} 
#         \toprule
#         \textbf{Scenario Tasks} & \textbf{Reason Step} & \textbf{GPT-4o} & \textbf{QwenVLMax} & \textbf{Gemini-1.5-Flash} \\
#         \midrule
#         \multirow{5}{=}{\raggedright Auto Insurance Claim Processing} 
#         & Damage judgment (\%) & 82.75 & 86.75 & 82.00 \\
#         & Damage Severity (\%) & 30.50 & 29.00 & 25.50 \\
#         & Estimated Loss (Diff) & 6185.61 & 9584.37 & 10894.78 \\
#         & Claim Eligibility (\%) & 62.00 & 55.50 & 47.50 \\
#          &\rowcolor{yellow!20} Final Claim Decision (Diff) & 3029.09 & 3686.32 & 4258.76 \\
#         \midrule
#         \multirow{6}{=}{\raggedright Auto Insurance Accident Liability Determination} 
#         & Weather Classification (\%) & 71.11 & 64.07 & 71.11 \\
#         & Scene Classification (\%) & 81.85 & 81.85 & 84.07 \\
#         & Linear Classification (\%) & 61.11 & 54.44 & 56.30 \\
#         & Accident Occurrence (\%) & 55.19 & 31.48 & 14.81 \\
#         & Accident Cause judgment (\%) & 24.80 & 9.63 & 3.33 \\
#          &\rowcolor{yellow!20} Responsible Party Identification (\%) & 9.26 & 5.56 & 2.59 \\
#         \midrule
#         \multirow{4}{=}{\raggedright Health Insurance Risk Assessment} 
#         & Scan Region Classification (\%) & 94.00 & 94.25 & 100.00 \\
#         & Fraud detection (\%) & 50.75 & 56.75 & 57.75 \\
#         & Health Risk Assessment (\%) & 84.07 & 83.19 & 85.80 \\
#          &\rowcolor{yellow!20} Underwriting Decision (\%) & 43.50 & 22.00 & 48.25 \\
#         \midrule
#         \multirow{5}{=}{\raggedright Property Insurance Risk Management} 
#         & Disaster Occurrence judgment (\%) & 61.00 & 72.50 & 61.50 \\
#         & Disaster Type Classification (\%) & 30.50 & 32.00 & 38.00 \\
#         & Properties Count (Diff) & 43.41 & 38.60 & 32.97 \\
#         & Damaged Properties Count (Diff) & 19.15 & 17.77 & 17.80 \\
#         & \rowcolor{yellow!20} Insurance Decision (\%) & 52.00 & 46.00 & 49.50 \\
#         \midrule
#         \multirow{4}{=}{\raggedright Agricultural Insurance Claim Processing} 
#         & Crop Species Classification (\%) & 100.00 & 100.00 & 97.97 \\
#         & Fruit Count (Diff) & 7.76 & 6.83 & 11.57 \\
#         & Pest Infection judgment (\%) & 87.80 & 85.37 & 72.36 \\
#          & \rowcolor{yellow!20} Claim Decision (\%) & 84.15 & 81.30 & 68.70 \\
#         \bottomrule
#     \end{tabular}}
# \end{table}
