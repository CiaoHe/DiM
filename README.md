# Install

## Install Mamba-ssm
Refer to [Vision Mamba](https://github.com/hustvl/Vim/tree/main)
```bash
# Install causal_conv1d and mamba
pip install -e causal_conv1d>=1.1.0
pip install -e mamba-1p1p1
```

# Train
```bash
# Assign HF_HOME to the cache directory
export HF_HOME="/comp_robot/rentianhe/caohe/cache"

# Train with 4 GPUs
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_mamba.py --model DiM-S/2 --feature-path /shared_space/caohe/DATA/imagenet1k/train_vae --lr 5e-4
```
* feature-path: the path to the pre-extracted features (use SD-VAE compression rate as 8)
* model: the model name (Now supported `DiM-[S|M|L|XL]/[2|4|8]`)
* [default] batch size : global batch size = 256 (constant for all models, suggested in DiT)
* [optional] --lr : learning rate (default 1e-4, for mamba, 5e-4 is suggested)
* [optional] --ckpt-every: default to 50_000 (= 10 epochs)

The ckpt will be saved under `results/xxx-MODEL_VERSION/checkpoints/`


# Evaluation
## Sampling scripts
```bash
# Assign HF_HOME to the cache directory
export HF_HOME="/comp_robot/rentianhe/caohe/cache"
# 
MODEL_VERSION=005-DiM-S-2 # The model version
CKPT=0100000 # The checkpoint number
torchrun --nnodes=1 --nproc_per_node=4 sample_ddp.py --model DiM-S/2 --num-fid-samples 50000 --ckpt results/$MODEL_VERSION/checkpoints/$CKPT.pt --sample-dir samples/$MODEL_VERSION-$CKPT --per-proc-batch-size 64
```

## FID evaluation

First need to install related packages following [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations)
```bash
IN_50K_REFERENCE=/shared_space/caohe/DATA/imagenet1k/VIRTUAL_imagenet256_labeled.npz
# ! adapt the path to the sample features path
python evaluations/evaluator.py $IN_50K_REFERENCE samples/DiM-B-2-ckpt-0100000/DiM-B-2-0100000-size-256-vae-ema-cfg-1.5-seed-0.npz
```