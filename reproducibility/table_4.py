import subprocess
import pandas as pd
import os

model_list = [
    "GPT4o",
    "QwenVLMax",
    "Gemini1_5Flash",
    "GLM4V",
    "GPT4V",
    "GPT4o_MINI",
    "QwenVLPlus",
    "Claude3V_Haiku",
    "QwenVL2_5",
    "QweVLChat",
    "LLaVA"
]
data_name = "INS_MMBench_fundamental"


for model in model_list:
    cmd = [
        "python",
        "run.py",
        "--data", data_name,
        "--model", model,
        "--verbose",
        "--nproc", "4"
    ]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

target_columns = [
    "Overall",
    "auto insurance",
    "household/commercial property insurance",
    "health insurance",
    "agricultural insurance"
]

results = []

for model in model_list:
    folder = model
    csv_filename = f"{model}_{data_name}_acc.csv"
    csv_path = os.path.join(folder, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)  
    row = df.loc[df["split"] == "none"].iloc[0]
    
    row_data = [model]
    for col in target_columns:
        value = row.get(col, None)
        row_data.append(value)
    
    results.append(row_data)

columns = ["Model"] + target_columns
summary_df = pd.DataFrame(results, columns=columns)

print(summary_df)


# Note that the output is not directly a table. Instead, the output is the source result for generating the table in latex. 
# To generate the table, please replace the result in the following latex code.

# Latex source code:
# \begin{table}
#     \centering
#     \caption{Evaluation results of the LVLMs across different insurance types. The values in the table represent the average accuracy. The highest and second-highest results are highlighted in \textbf{bold} and \underline{underlined}, respectively.}
#     \label{evaluation results}
#     \adjustbox{max width=0.47\textwidth}{
#     \begin{tabular}{lccccc}
#         \toprule
#         \textbf{Model}  & \textbf{Overall}     & \textbf{\begin{tabular}[c]{@{}c@{}}Auto\\      insurance\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Household/commercial\\      property insurance\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Health\\      insurance\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}Agricultural\\      insurance\end{tabular}} \\
#         \midrule
#         GPT-4o          & {\textbf{69.70}} & {\textbf{86.00}}      & {\textbf{63.77}}                             & {\textbf{76.73}}        & {\ul{36.38}}           \\
#         Qwen-VL-Max       & {\ul{65.33}}                & 80.86                     & {\ul{61.99}}                                            & 70.60                       & 33.18                           \\
#         Gemini 1.5 Flash &  64.21                & {\ul{79.40}}                     & 60.18                                            & 70.31                       & 32.84                           \\
#         GLM4V-Plus-0111 & 63.51 & {\ul{81.79}} & 53.57 & 72.49 & 29.92 \\
#         GPT-4V      & 62.79                & 77.35                     & 60.55                                           & 70.82                       & 29.23       \\        
#         GPT-4o-mini       & 60.66                & 77.77                     & 58.53                                            & 63.61                       & 25.80                           \\
#         Qwen-VL-Plus      & 54.94                & 71.42                     & 48.51                                            & 64.92                       & 20.48       \\
#         Claude3V\_Haiku      & 48.95                & 59.95                     & 49.63                                             & 59.02                        & 17.91        \\
#         \cline{1-6}
#         Qwen-2.5-VL-32B & 64.10 & 79.34 & 54.58 & {\ul{76.27}} & 33.70 \\
#         Qwen-VL-Chat      & 48.85                & 57.64                     & 45.90                                            & 65.14                       & 21.34       \\
#         LLaVA &  46.99  &  45.47  &  56.82  &  65.25  &   26.26\\
#         \midrule 
#         Human baseline      & 60.45               & 62.22                    & 60.00                                            & 75.00                       & \textbf{42.50}       \\ 
#         \bottomrule
#     \end{tabular}}
# \end{table}
