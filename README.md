
<h1 align="center"> 🦢GOSE 
</h1>
<div align="center">
     
   [![Awesome](https://awesome.re/badge.svg)]() 
   [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
   ![](https://img.shields.io/github/last-commit/chenxn2020/GOSE?color=green) 
   ![](https://img.shields.io/badge/PRs-Welcome-red) 
</div>

## *👋 News!*

- Code for the paper [`Global Structure Knowledge-Guided Relation Extraction Method for Visually-Rich Document`](https://arxiv.org/abs/2305.13850).

- Congratulations! Our work has been accepted by the EMNLP2023 Findings conference.

## Quick Links

* [Setup](#setup)
* [Model Preparation](#model-preparation)
* [Train GOSE](#train-gose)
  * [Language-specific Fine-tuning](#lsf)
  * [Multilingual fine-tuning](#mf)
* [Acknowledgment](#acknowledgment)

## Setup

<a id="setup"></a>

We check the reproducibility under this environment.

+ Python 3.8.18
+ CUDA 11.1

To run the codes, you need to install the requirements:

```bash
git clone https://github.com/chenxn2020/GOSE.git
cd GOSE

conda create -n gose python=3.8
conda activate gose
pip install -r requirements.txt

```

## Model Preparation

<a id="model-preparation"></a>

We utilize [LayoutXLM](https://github.com/microsoft/unilm/blob/master/layoutxlm/README.md) and [LiLT](https://github.com/jpWang/LiLT/tree/main) as our backbone.
You can download the models and place them under the `GOSE/`.

## Train GOSE

<a id="train-gose"></a>

We provide example scripts for explaining the usage of our code. You can kindly run the following commands.

+ ### Language-specific Fine-tuning

```bash
# Current path:  */GOSE
bash standard.sh
```

+ ### Multilingual fine-tuning

```bash
# Current path:  */GOSE
bash multi.sh
```

## Acknowledgment

The repository benefits greatly from [unilm/layoutlmft](https://github.com/microsoft/unilm/tree/master/layoutlmft) and [LiLT](https://github.com/jpWang/LiLT/tree/main). Thanks a lot for their excellent work.

## Citation

If our paper helps your research, please cite it in your publication(s):

```
@article{DBLP:journals/corr/abs-2305-13850,
  author       = {Xiangnan Chen and
                  Juncheng Li and
                  Duo Dong and
                  Qian Xiao and
                  Jun Lin and
                  Xiaozhong Liu and
                  Siliang Tang},
  title        = {Global Structure Knowledge-Guided Relation Extraction Method for Visually-Rich
                  Document},
  journal      = {CoRR},
  volume       = {abs/2305.13850},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.13850},
  doi          = {10.48550/ARXIV.2305.13850},
  eprinttype    = {arXiv},
  eprint       = {2305.13850},
  timestamp    = {Mon, 05 Jun 2023 15:42:15 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-13850.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
