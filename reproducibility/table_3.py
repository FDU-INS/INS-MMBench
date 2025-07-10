import subprocess
import pandas as pd
import os

model_list = [
    "GPT4o",
    "QwenVLMax",
    "Gemini 1.5 Flash",
    "GLM4V",
    "GPT4V",
    "GPT4o_MINI",
    "QwenVLPlus",
    "Claude3V_Haiku",
    "QwenVL2_5",
    "QweVLChat",
    "LLaVA"
]

data_name = "INS-MMBench"

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
    subprocess.run(cmd, check=True)

target_columns = [
    "vehicle information extraction",
    "vehicle appearance recognition",
    "driving behavior detection",
    "vehicle damage detection",
    "household/commercial property anomaly detection",
    "household/commercial property damage detection",
    "household/commercial property risk assessment",
    "health risk monitoring",
    "medical image recognition",
    "crop growth status identification",
    "crop type identification",
    "farmland damage detection"
]

results = []

for model in model_list:
    folder = model
    csv_filename = f"{model}_{data_name}_acc.csv"
    csv_path = os.path.join(folder, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path, sep="\t")
    
    row = df.loc[df["split"] == "none"].iloc[0]
    
    row_data = [model]
    for col in target_columns:
        value = row.get(col, None)
        row_data.append(value)
    
    results.append(row_data)

columns = ["Model"] + target_columns
summary_df = pd.DataFrame(results, columns=columns)

print(summary_df)

"""
Note that the output is not directly a table. Instead, the output is the source result for generating the table in latex. 
To generate the table, please replace the result in the following latex code.

\begin{table*}
    \centering
    \caption{Evaluation results of the LVLMs across different meta-tasks. The values in the table represent the average accuracy. Specifically, \textbf{VIE} denotes vehicle information extraction, \textbf{VAR} denotes vehicle appearance recognition, \textbf{DBD} denotes driving behavior detection, \textbf{VDD} denotes vehicle damage detection, \textbf{HPAD} denotes household/commercial property anomaly detection, \textbf{HPDD} denotes household/commercial property damage detection, \textbf{HPRA} denotes household/commercial property risk assessment, \textbf{HRM} denotes health risk monitoring, \textbf{MIR} denotes medical image recognition, \textbf{CGSI} denotes crop growth stage identification, \textbf{CTI} denotes crop type identification, \textbf{FDD} denotes farmland damage detection. The highest and second-highest results are highlighted in \textbf{bold} and \underline{underlined}, respectively.}
    \label{evaluation results detail}
    \adjustbox{max width=0.75\textwidth}{
    \begin{tabular}{lcccccccccccc}
    \toprule
    \textbf{Model}  & \multicolumn{1}{l}{\textbf{VIE}} & \multicolumn{1}{l}{\textbf{VAR}} & \multicolumn{1}{l}{\textbf{DBD}} & \multicolumn{1}{l}{\textbf{VDD}} & \multicolumn{1}{l}{\textbf{HPAD}} & \multicolumn{1}{l}{\textbf{HPDD}} & \multicolumn{1}{l}{\textbf{HPRA}} & \multicolumn{1}{l}{\textbf{HRM}} & \multicolumn{1}{l}{\textbf{MIR}} & \multicolumn{1}{l}{\textbf{CGSI}} & \multicolumn{1}{l}{\textbf{CTI}} & \multicolumn{1}{l}{\textbf{FDD}} \\
    \midrule
    GPT-4o           & { \textbf{81.12}}             & { \textbf{98.50}}             & {\ul {88.60}}             & {\ul {83.94}}                            &  \textbf{91.16}              & {\textbf{47.04}}              & 65.50              & {\textbf{95.72}}             & {\ul{66.50}}             & 30.80              & {\textbf{41.31}}             & {\ul {34.60}}             \\
    Qwen-VL-Max        & 75.28                            & {\ul {98.20}}                            & 74.80                            & 81.88             & 80.72                             & 45.79                             & {\textbf{71.80}}                             & 88.24                            & 64.00                            & 29.60                             & {\ul {40.37}}                   & 26.00                            \\
    Gemini 1.5 Flash  & 67.28                            & 96.80             & 79.20                            & \textbf{84.40}                            & 74.30              & 46.36                             & { \ul {70.40}}                             & 81.82                            & 66.00                            & {\ul {36.60}}                             & 38.10                            & 21.20                            \\
    GLM4V-Plus-0111 & 77.68 & 92.60 & 84.40 & 80.58 & 58.84 & 42.94 & 67.00 & 90.11 & 65.90 & 28.80 & 33.56 & 25.60 \\
    
    GPT-4V & 72.16                            & 93.60                            & 66.20                            & 80.35                            & 88.35                             & 41.80                             & 65.80                             & 94.12                            & 62.10                            & 23.60                             & 39.17                            & 20.00                   \\
    
    GPT-4o-mini        & 70.24                           & 95.20                            & 85.80                            & 75.23             & {\ul {89.56}}                             & 39.75                             & 60.60                             & {\ul {94.39}}                            & 52.10                            & 23.80                            & 34.36                   & 15.00                            \\
    Qwen-VL-Plus       & 63.84                            & 96.20                            & 69.60                            & 69.88                            & 57.03                             & 39.18                             & 56.40                             & 86.10                            & 57.00                            & 15.40                             & 25.40                            & 18.20                            \\
    
    Claude3V\_Haiku & 45.76                             & 86.8                             & 52.40                            & 66.13                             & 75.10                             & 27.90                             & 62.40                             & 84.49                            & 49.50                            & 19.80                             & 23.53                            & 7.60                   \\
    \midrule
    Qwen-2.5-VL-32B & 76.48 & 94.80 & 75.00 & 77.83 & 73.69 & 38.04 & 64.60 & 91.98 & \textbf{70.40} & 27.60 & 38.90 & 32.00 \\
    Qwen-VL-Chat & 44.32                             & 94.60                             & 59.60                             & 55.50                             & 59.04                              & 30.41                              & 60.00                              & 80.75                             & 59.30                             & 15.80                              & 30.62                             & 13.00                    \\
    LLaVA &  32.64  &  60.20  &  51.80  &  49.69  &  87.35  &  34.85  &  65.00  &  83.69  &  57.54  &  21.40  &  37.57  &  14.20 \\
    \midrule
    Human baseline & {\ul {79.50}}                            & 59.50                            & \textbf{98.67}                            &  52.78                            & 73.33                             & {\ul {46.67}}                             & 63.33                             & 85.00                            & 65.00                            & \textbf{60.00}                             & 35.00                            & \textbf{40.00}                   \\
    \bottomrule
    \end{tabular}}
\end{table*}
"""
