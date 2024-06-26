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
