# SE-VGAE
SE-VGAE: Unsupervised Disentangled Representation Learning for Interpretable Architectural Layout Design Graph Generation

Official code and instructions for [**"SE-VGAE: Unsupervised Disentangled Representation Learning for Interpretable Architectural Layout Design Graph Generation"**](https://arxiv.org/html/2406.17418v1)

# Introduction
This project introduces an unsupervised disentangled representation learning framework, Style-based Edge-augmented Variational Graph Auto-Encoder (SE-VGAE), aiming to generate architectural layout in the form of attributed adjacency multi-graphs while prioritizing representation disentanglement. The framework is designed with three alternative pipelines, each integrating a transformer-based edge-augmented encoder, a latent space disentanglement module, and a style-based decoder. These components collectively facilitate the decomposition of latent factors influencing architectural layout graph generation, enhancing generation fidelity and diversity.

# Environment
- Linux
- [Pytorch 1.8.1+cu113](https://pytorch.org/get-started/previous-versions/#linux-and-windows-48)
- CUDA-supported GPU with at least 24 GB memory size is required for training.

# Quick start
## Download the repository
```
git clone https://github.com/JanineCHEN/SE-VGAE.git
cd SE-VGAE
```
## Setup virtual environment
```
conda env create -f environment.yml
conda activate sevgae
conda install conda-forge::graph-tool
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
```

## Dataset
For downloading the dataset, please refer to <a href="https://github.com/JanineCHEN/SE-VGAE/tree/main/data">data</a>.

## Training
For training the FP4S model, please run:
```
python main.py
```
For customized configuration, please refer to <a href="https://github.com/JanineCHEN/SE-VGAE/tree/main/utils/config.py">config</a>.

### Acknowledgement
The computational work for this article was performed on resources of the National Supercomputing Centre, Singapore (https://www.nscc.sg). The data sources used in this study are also gratefully acknowledged. This research was supported by the President’s Graduate Fellowship of the National University of Singapore and the Singapore Data Science Consortium (SDSC) Dissertation Research Fellowship.

