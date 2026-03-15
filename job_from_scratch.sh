#!/bin/bash
# Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

export PATH="$HOME/miniconda3/bin:$PATH"

# Create and activate conda env
if ! conda env list | grep -q olmoe; then
    conda create -y -n olmoe python=3.10
fi
source ~/miniconda3/bin/activate
conda activate olmoe

# Install dependencies
pip install --upgrade pip
pip install wandb torchmetrics datasets scikit-learn safetensors msgspec ninja

# Install CUDA-specific packages
export TORCH_CUDA_ARCH_LIST="8.6"
pip uninstall stanford-stk megablocks -y
pip install stanford-stk --no-binary stanford-stk --force-reinstall --no-cache-dir
pip install megablocks --no-binary megablocks --force-reinstall --no-cache-dir

# Set WANDB API key if available
export WANDB_MODE=online
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env 2>/dev/null | cut -d '=' -f 2)

# Download and tokenize data, then delete original
mkdir -p data data_all

# Install OLMo package and dolma
git submodule update --init --recursive
cd OLMo
pip install -e .
pip install dolma
cd ..

for file in wiki-001 c4-train.00000-of-01024 c4-train.00001-of-01024; do
    case $file in
        wiki-001)
            url="https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/wiki/wiki-0001.json.gz?download=true"
            ;;
        pes2o-0000)
            url="https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/pes2o/pes2o-0000.json.gz?download=true"
            ;;
        c4-train.00000-of-01024)
            url="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz?download=true"
            ;;
        c4-train.00001-of-01024)
            url="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00001-of-01024.json.gz?download=true"
            ;;
    esac

    wget -O data/${file}.json.gz "$url"

    dolma tokens \
    --documents data/${file}.json.gz \
    --destination data_all/${file}.npy \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 2

    rm data/${file}.json.gz
done


# Launch your training job (edit as needed)
python OLMo/scripts/train.py configs/olmoe-small.yml