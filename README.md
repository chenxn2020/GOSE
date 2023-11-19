# GOSE
Code for the EMNLP2023 (Findings) paper "Global Structure Knowledge-Guided Relation Extraction Method for Visually-Rich Document""

环境安装
conda create -n gose python=3.8
conda activate gose
pip install -r requirements.txt
python -m pip install detectron2==0.5 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install -e .
pip install packaging==21.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install timm



预训练模型下载
https://github.com/jpWang/LiLT
lilt-infoxlm-base

finetune
bash gose.sh [language]
e.g.  base gose.sh en  #在funsd数据集上finetune
