# RNADiffusion: Generative RNA Secondary Structure Prediction using DiT

## 1.Please use the .yml file to create your environment

```sh
conda env create -f requirements.yml
```

## 2. During training, the hyperparameters are set as follows

```sh
python train.py --device cuda:0
                --diffusion_dim 8
                --diffusion_steps 1000
                --cond_dim 8
                --batch_size 1
                --dp_rate 0.1
                --lr 0.0001
                --warmup 5
                --seed 2024
                --log_wandb False
                --epochs 50
                --eval_every 5
```

## 3. Data

```sh
python data/batching_data.py
python data/generating_final_data.py
```
