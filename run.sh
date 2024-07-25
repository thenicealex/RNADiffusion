python main.py  --device cuda:3 \
                --diffusion_dim 8 \
                --diffusion_steps 20 \
                --cond_dim 8 \
                --dataset bpRNAnew \
                --batch_size 1 \
                --dp_rate 0.1 \
                --lr 0.0001 \
                --warmup 5 \
                --seed 2024 \
                --log_wandb False \
                --epochs 200 \
                --eval_every 5 \
                --u_conditioner_ckpt /home/fkli/RNAm/ufold_train_alldata.pt \
                --esm_conditioner_ckpt /home/fkli/RNAm/RNA-ESM2-trans-2a100-mappro-KDNY-epoch_06-valid_F1_0.564.ckpt \
