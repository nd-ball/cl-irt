# PUDF Framework for Curriculum Learning

This repository contains the implementation of the Psychology-based Unified Dynamic Framework (PUDF) for Curriculum Learning, as presented in our paper titled *"A Psychology-based Unified Dynamic Framework for Curriculum Learning."*

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [PUDF Workflow](#pudf-workflow)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Generating Data Difficulty](#generating-data-difficulty)
   - [Running Experiments](#running-experiments)
   - [Benchmarking](#benchmarking)
   - [Ablation Study](#ablation-study)
6. [Results](#results)
7. [Citation](#citation)
8. [License](#license)

## Introduction

The PUDF framework is designed to enhance the performance of machine learning models by dynamically adjusting the training curriculum based on the difficulty of the training data and the model's ability. It leverages Item Response Theory (IRT) and Artificial Crowds (AC) to automatically estimate the difficulty of training examples and the ability of models during the training process.

## Features

- Automatic difficulty estimation for training data
- Dynamic curriculum adjustment during training
- Implementation for GLUE and SuperGLUE benchmarks
- Comparison with state-of-the-art curriculum learning methods
- Ablation study capabilities

## PUDF Workflow

The PUDF framework consists of two main steps:

1. IRT-AC for the Difficulty Measurement (DM)
2. DDS-MAE and PLM Fine-tuning for the Training Strategy (TS)

![PUDF Workflow](workflow_PUDF.jpg)

Figure 1: Workflow of the PUDF. The process consists of two main steps: 1) IRT-AC for the DM, 2) DDS-MAE and PLM Fine-tuning for the TS.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/pudf-framework.git
   cd pudf-framework
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

### Generating Data Difficulty

Before running the experiments, generate the difficulty scores for the training data:

```bash
cd gen_difficulty
bash gen_respon_256.sh
```

This step generates difficulty scores for the training dataset using Artificial Crowds (AC).

### Running Experiments

#### Baseline Models

To run baseline experiments:

```bash
cd baseline_GLUE
bash glue_deberta.sh

cd ../baseline_SuperGLUE
bash superglue_deberta.sh
```

#### PUDF-Enhanced Models

To run PUDF-enhanced models:

```bash
cd PUDF_GLUE
bash glue_deberta.sh

cd ../PUDF_SuperGLUE
bash superglue_deberta.sh
```

### Benchmarking

To compare PUDF with other state-of-the-art curriculum learning methods:

```bash
cd benchmark_CL_GLUE
bash compare_cl_methods.sh
```

### Ablation Study

To perform an ablation study and analyze the contributions of different components of PUDF:

```bash
cd ablation_study
bash run_ablation.sh
```

## Results

Our experiments show that PUDF significantly improves the performance of various pre-trained language models (PLMs) on the GLUE benchmark. Here are some key results:

### Performance Comparison on GLUE Benchmark

![Performance Comparison](main_results1.jpg)

Table 1: Comparison of the performance of PLMs with and without PUDF on the GLUE benchmark. The table shows accuracy (Acc.) and training time in minutes (TT), with standard deviations (std) calculated from three repeated experiments in parentheses. Bold results indicate those that are statistically significantly better (p < 0.05) than the baseline.

### Comparison with Other Curriculum Learning Methods

![CL Methods Comparison](main_results2.jpg)

Table 2: Comparison of different CL methods on the DeBERTaV3 model on GLUE benchmark. The table shows accuracy (Acc.) and training time in minutes (TT). dSL and dWR denote sentence length and word rarity, respectively. L and R represent the linear and root functions, respectively. Trans. and RL. denote transfer-teacher and reinforcement learning-based CL methods. Best results are bolded. PUDF significantly outperforms baseline and other CL methods (p < 0.05), except those marked by †.

These results demonstrate that PUDF consistently improves both the accuracy and training efficiency of various PLMs across different GLUE tasks.

## Citation

If you use this framework in your research, please cite our paper:

```
[Insert citation details here]
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
