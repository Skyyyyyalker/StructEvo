# StructEvo

## Getting Started

### Setup Environment

Our work run on Python 3.11 and CUDA 12.2. You can set up the environment by running command:
```
pip install -r requirements.txt
```
We strongly recommend to create virtual environment by [conda](https://www.anaconda.com/).

### DownLoad Checkpoints

- Download pretrained weights of ESM-2-8m from [here](https://huggingface.co/facebook/esm2_t6_8M_UR50D), and put it under `ckpts/esm2_t6_8M_UR50D`.
- Download pretrained weights of ESM-2-650m from [here](https://huggingface.co/facebook/esm2_t33_650M_UR50D), and put it under `ckpts/esm2_t33_650M_UR50D`.
- Download pretrained weights of ProSST from [here](https://huggingface.co/AI4Protein/ProSST-2048), and put it under `ckpts/ProSST-2048`.
- The required dataset and checkpoints of [GGS](https://github.com/kirjner/GGS) are already included in the repository, under directory `data` and `ckpts/ggs`.


## Full-length Benchmarks

- Run StructEvo on AAV and GFP benchmarks:
```
bash run_full-length.sh [AAV/GFP] [medium/hard]
```
for example, `bash run_full-length.sh AAV medium`

- Evaluate results on AAV and GFP benchmarks:
```
bash evaluate_full-length.sh [AAV/GFP] [medium/hard]
```
for example, `bash evaluate_full-length.sh AAV medium`


## Combinatorial Benchmarks

- The initialization of combinatorial benchmarks rely on [CLADE](https://www.nature.com/articles/s43588-021-00168-y). Download CLADE repository from [here](https://github.com/WeilabMSU/CLADE). Follow the instruction in CLADE and put the initial pool under `./candidates/[GB1 or PhoQ]/[seed]/round_0.csv`. You can find a process script converting file format at `./structevo/process_clade.py`.

- Run StructEvo on GB1 and PhoQ benchmarks:
```
bash run_4-site.sh [GB1/PhoQ]
```
for example, `bash run_4-site.sh GB1`

- Evaluate results on GB1 and PhoQ benchmarks:
```
bash evaluate_4-site.sh [GB1/PhoQ]
```
for example, `bash evaluate_4-site.sh GB1`
