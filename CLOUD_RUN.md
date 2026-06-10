# Cloud Run

Use this when you want a GPU-backed rerun of the more credible experiment pipeline in `credible_experiment_runner.py`.

## 1. Create env

```powershell
conda create -y -n hcfcuda python=3.10
conda run -n hcfcuda python -m pip install --upgrade pip
conda run -n hcfcuda python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n hcfcuda python -m pip install numpy scipy scikit-learn matplotlib networkx
```

## 2. Smoke test GPU

```powershell
conda run -n hcfcuda python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 3. Run one dataset first

```powershell
conda run -n hcfcuda python credible_experiment_runner.py --datasets DBLP --epochs 180
```

## 4. Full run

```powershell
conda run -n hcfcuda python credible_experiment_runner.py --datasets ACM DBLP Yelp --epochs 180
```

## Outputs

All new outputs go to `results_hcfnet_credible/`:

- `summary.json`
- `*_metrics.json`
- `*_sir.png`
- `*_si.png`
- `*_hcf_meta.json`
