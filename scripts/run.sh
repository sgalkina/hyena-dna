module load tools computerome_utils/2.0
module load cuda/toolkit/11.8.0
module load gcc/11.1.0
module load anaconda3/2023.09-0
module load git-lfs/3.3.0

source ~/.bashrc
conda activate hyena-dna

# Pretrain of 96 genomes with input size 2048
/home/projects/matrix/people/svekut/.conda/hyena-dna/bin/python -m train \
    wandb.group="hyena-2048-ep100" \
    experiment=hg38/gtdb_hyena trainer.max_epochs=100 \
    model.d_model=128 \
    model.n_layer=2 \
    dataset.batch_size=256 \
    train.global_batch_size=256 \
    dataset.max_length=2048 \
    optimizer.lr=6e-4 \
    trainer.devices=2



# Use the tiny-1k model pretrained on the human genome
# /home/projects/matrix/people/svekut/.conda/hyena-dna/bin/python -m train \
#     wandb.group="hyena-tiny" \
#     experiment=hg38/gtdb_hyena trainer.max_epochs=100 \
#     model.d_model=128 \
#     model.n_layer=2 \
#     dataset.batch_size=256 \
#     train.global_batch_size=256 \
#     dataset.max_length=1024 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=/home/projects/matrix/data/dna_transformers/hyenadna-tiny-1k-seqlen/weights.ckpt
