## EASING: Semi-supervised Node Importance Estimation with Informative Distribution Modeling for Uncertainty Regularization

![替代文本](Framework.jpg)

This repository provides the PyTorch implementation for our WWW'25 paper [[EASING](https://arxiv.org/abs/2503.20697)].


## 🔧 Requirements

The codes are built with the following main dependencies:

* `python=3.8.18`
* `torch=1.8.1+cu111`
* `dgl-cu111=0.6.1`
* `scikit-learn=1.3.2`
* `scipy=1.8.0`
* `tqdm=4.66.1`
* `numpy=1.22.4`


## 💾 Datasets

### Download

The datasets used in our paper can be downloaded from the following link. 

* **Dataset Link:** [Google Drive](https://drive.google.com/drive/folders/1kOKnQx6erIL3BJU4Vg4ntjU4PP0v156F?usp=sharing)

After downloading, place the contents into a directory, which will be referred to as `{your path}` in the usage instructions.

### File Descriptions

The datasets contain the following key files:

* **For FB15K and TMDB5K:**
    * `xx_rel.pk`: Contains graph data and structural features.
    * `xx_lang.pk`: Contains semantic features.
        * (Replace `xx` with `fb15k` for FB15K and `tmdb` for TMDB5K)
    * `idx_1000`: folder for node IDs and labels 

* **For IMDB:**
    * `imdb_s_rel_150000.pk`: Contains graph data.
    * `imdb_s_node2vec_new_150000.pk`: Contains structure features (e.g., from Node2Vec).
    * `imdb_s_lang_150000.pk`: Contains semantic features.
    * `idx_1000`: folder for node IDs and labels 


### Dataset Organization 
The dataset organization under "/datasets" is:
```
├── datasets
│   ├── FB15k
│   │   ├── dataset_split
│   │   │   ├── idx_1000
│   │   │   │   └── ...
│   │   ├── fb_lang.pk
│   │   ├── fb15k_rel.pk
│   ├── TMDB5K
│   │   └── ...
│   ├── IMDB
│   │   └── ...
└── utils
└── ...
```


## 🚀 Usage

To training/evaluating our model, please execute the following commands, after specifying the "--data_path {your path}":

### FB15K

```bash
python easing/main_easing.py --dataset FB15K \
--data_path {your path} \
--graph_data FB15K/fb15k_rel.pk --semantic_data FB15K/fb_lang.pk \
--residual --norm --spm --train_num 1.0 \
--samp_ssl 5 --w_ulb 1.0 --loss_beta 0.0 --centrality_gamma 0.9 \
--REnt_bool --unc_layers 2
```

### TMDB15K

```bash
python easing/main_easing.py --dataset TMDB5K \
--data_path {your path} \
--graph_data TMDB5K/tmdb_rel.pk --semantic_data TMDB5K/tmdb_lang.pk \
--residual --norm --spm --train_num 1.0 \
--samp_ssl 5 --w_ulb 1.0 --loss_beta 0.0 --centrality_gamma 0.9 \
--REnt_bool --unc_layers 1
```

### IMDB

```bash
python easing/main_easing.py --dataset IMDB \
--data_path {your path} \
--graph_data IMDB/imdb_s_rel_150000.pk --semantic_data IMDB/imdb_s_lang_150000.pk \
--structure_data IMDB/imdb_s_node2vec_new_150000.pk \
--residual --norm --spm --train_num 1.0 \
 --samp_ssl 5 --w_ulb 1.0 --loss_beta 0.0 --centrality_gamma 0.6 \
 --REnt_bool --unc_layers 1
```


## 📄 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{chen2025semi,
  title={Semi-supervised node importance estimation with informative distribution modeling for uncertainty regularization},
  author={Chen, Yankai and Wang, Taotao and Fang, Yixiang and Xiao, Yunyu},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={3108--3118},
  year={2025}
}
```