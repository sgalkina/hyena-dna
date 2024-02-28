# For speed
# conda install conda-libmamba-solver
# /home/nmb127/.conda/envs/hyena-dna/bin/conda config --set solver libmamba

module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate hyena-dna

/home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
    wandb.group="hyena-cami-continue" \
    experiment=hg38/cami_hyena trainer.max_epochs=100 \
    dataset.fasta_path=/home/nmb127/code/data/contigs_2kbp.fna.gz \
    dataset.fasta_taxonomy=/home/nmb127/code/data/gt_tax_urog.csv \
    model.d_model=256 \
    model.n_layer=6 \
    model.d_inner=1024 \
    dataset.batch_size=16 \
    train.global_batch_size=256 \
    dataset.max_length=16384 \
    optimizer.lr=6e-4 \
    trainer.devices=1 \
    train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-02-07/11-48-54-696866/checkpoints/last.ckpt
