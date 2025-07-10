import pandas as pd
import os

model_list = [
    "GPT4o",
    "GPT4V",
    "QwenVLMax",
    "QwenVLPlus",
    "Gemini1_5Flash"
]

dataset_list = [
    "hdt_with_prompt",
    "cgs_with_prompt",
    "vds_with_prompt"
]

dataset_metric_mapping = {
    "hdt_with_prompt": "household/commercial damage  detection",
    "cgs_with_prompt": "crop growth status identification",
    "vds_with_prompt": "vehicle damage detection"
}

results = []

for model in model_list:
    for dataset in dataset_list:
        metric_name = dataset_metric_mapping[dataset]
        
        folder = model
        csv_filename = f"{model}_{dataset}_acc.csv"
        csv_path = os.path.join(folder, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            value = None
        else:
            df = pd.read_csv(csv_path)
            
            row = df.loc[df["split"] == "none"].iloc[0]
            
            value = row.get(metric_name, None)
        
        results.append({
            "Model": model,
            "Dataset": dataset,
            "Metric": metric_name,
            "Value": value
        })

df = pd.DataFrame(results)

print(df)



# Note that the output is not directly a table. Instead, the output is the source result for generating the table in latex. 
# To generate the table, please replace the result in the following latex code.

# \begin{table}
#     \centering
#     \caption{Results of enhanced insurance-related prompts on LVLMs performance across selected tasks. The values represent accuracy (\%), and changes in performance are highlighted in \textcolor{green}{green} for improvements and \textcolor{red}{red} for declines.}
#     \label{enhanced_prompt_results}
#     \adjustbox{max width=0.48\textwidth}{
#     \begin{tabular}{lccc}
#         \toprule
#         \textbf{Model}        & \textbf{House Damage Type Detection} & \textbf{Crop Growth Stage Detection} & \textbf{Vehicle Damage Severity Detection} \\
#         \midrule
#         GPT-4o         & 48.00/\textbf{57.00} (\textcolor{green}{+9})   & 32.00/\textbf{51.00} (\textcolor{green}{+19}) & 68.00/\textbf{80.00} (\textcolor{green}{+12})  \\
#         GPT-4V         & 33.00/\textbf{40.00} (\textcolor{green}{+7})   & 22.00/\textbf{52.00} (\textcolor{green}{+30}) & 68.00/\textbf{77.00} (\textcolor{green}{+9})   \\
#         Gemini 1.5 Flash   & 33.00/\textbf{47.00} (\textcolor{green}{+14})   & 28.00/\textbf{57.00} (\textcolor{green}{+29}) & 68.00/\textbf{68.00} (-)  \\
#         Qwen-VL-Max    & 27.00/\textbf{42.00} (\textcolor{green}{+15})  & 30.00/\textbf{58.00} (\textcolor{green}{+28}) & 72.00/\textbf{61.00} (\textcolor{red}{-11})  \\
#         Qwen-VL-Plus   & 35.00/\textbf{38.00} (\textcolor{green}{+3})   & 22.00/\textbf{60.00} (\textcolor{green}{+38}) & 68.00/\textbf{58.00} (\textcolor{red}{-10})  \\
#         \bottomrule        
#     \end{tabular}}
# \end{table}