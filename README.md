# INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance
This repository is the official implementation of INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance ([arXiv link](https://arxiv.org/abs/2406.09105)).

> INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance ([arXiv link](https://arxiv.org/abs/2406.09105))  
> Chenwei Lin<sup>\*</sup>, Hanjia Lyu, Xian Xu, Jiebo Luo  

## Introduction
INS-MMBench is the first comprehensive LVLMs benchmark for the insurance domain, it covers four representative insurance types: auto, property, health, and agricultural insurance and key insurance stages such as risk underwriting, risk monitoring and claim processing. INS-MMBench comprises a total of 2.2K thoroughly designed multiple-choice questions, covering 12 meta-tasks and 22 fundamental tasks.
![task overview](assets/task_overview.png)

## Evaluation Results Overview
Overall, GPT-4o outperforms all other models, emerging as the top-performing LVLM on the INS-MMBench with a score of 72.91. This is the only model with an overall score exceeding 70, underscoring the challenging nature of the INS-MMBench. Most LVLMs scored below 60, and some even underperformed relative to a random guess baseline of 25 in certain insurance categories, indicating significant potential for improvement in applying LVLMs within the insurance domain. Besides, we have following observations:

- LVLMs show significant variance across different types of insurance.
- LVLMs show significant variance across different meta-tasks.
- Narrowing gap between open-source and closed-source LVLMs.
![result](assets/result_across_insurance_type.png)

![result](assets/result_across_meta_task.png)

## Quick start
### Step 1: Installation
To set up the project, run the following commands:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd INS-MMBench
pip install -r requirements.txt
```bash

### Step 2: Configuration

#### 1. VLM Configuration
- All VLMs are configured in `vlmeval/config.py`.
- For some VLMs, you need to configure the **code root** (e.g., MiniGPT-4, PandaGPT) or the **model_weight root** (e.g., LLaVA-v1-7B) before conducting the evaluation.
- During evaluation, use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM.
- For models not listed in `supported_VLM`, you can customize the configuration.

#### 2. Benchmark Download
- The datasets can be downloaded through the provided Dropbox link (**[Link]**).
- Place the dataset folder in the default path `$HOME/LMUData` or a custom path.

#### 3. `.env` File Setup
- Update the `.env` file with necessary information such as API keys, base URLs, and other settings required for model integration.

---

### Step 3: Evaluation

- Use the `run.py` script for evaluation. 
- You can execute it via `$VLMEvalKit/run.py` or create a soft-link to use the script anywhere.

#### Arguments
| Argument     | Type        | Description                                                                                      |
|--------------|-------------|--------------------------------------------------------------------------------------------------|
| `--data`     | `list[str]` | Specify dataset names supported in VLMEvalKit (defined in `vlmeval/utils/data_util.py`).          |
| `--model`    | `list[str]` | Specify VLM names supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`).    |
| `--mode`     | `str`       | Evaluation mode: `"all"` (default) for both inference and evaluation; `"infer"` for inference only. |
| `--nproc`    | `int`       | Number of threads for OpenAI API calling (default: 4).                                           |
| `--verbose`  | `bool`      | Enable verbose logging.                                                                          |

#### Example Command
```bash
python run.py --data INS-MMBench --model GPT4o --verbose --nproc 4

The final results will be saved in a folder named after the model used for evaluation.

## 💐 Acknowledgement
We express our sincere gratitude to the following projects:
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) provides useful out-of-the-box tools and implements many advanced models. Thanks for their selfless dedication.

## 🖊️ Citation 
If you find our work useful in your project or research, please use the following BibTeX entry to cite our paper. Thanks!
```
@article{insmmbench,
    title={INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs’ Performance in Insurance}, 
    author={Lin, Chenwei and Lyu, Hanjia and Xu, Xian and Luo, Jiebo},
    journal={arXiv preprint arXiv:2406.09105},
    year={2024}
}
```
