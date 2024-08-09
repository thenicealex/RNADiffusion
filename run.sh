UNETPATH="/home/fkli/RNAm/ufold_train_alldata.pt"
ESM2PATH="/home/fkli/RNAm/RNA-ESM2-trans-2a100-mappro-KDNY-epoch_06-valid_F1_0.564.ckpt"

python main.py  --device cuda:0 \
                --diffusion_dim 8 \
                --diffusion_steps 20 \
                --cond_dim 8 \
                --batch_size 1 \
                --dp_rate 0.1 \
                --lr 0.0001 \
                --warmup 5 \
                --upsampling False \
                --seed 2024 \
                --epochs 3 \
                --eval_every 1 \
                --dry_run \
                --log_wandb False \
                --u_conditioner_ckpt $UNETPATH \
                --esm_conditioner_ckpt $ESM2PATH \
