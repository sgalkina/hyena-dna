# For speed
# conda install conda-libmamba-solver
# /home/nmb127/.conda/envs/hyena-dna/bin/conda config --set solver libmamba

module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate hyena-dna

/home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
    wandb.group="hyena-16k-ep100-family-large" \
    experiment=hg38/gtdb_hyena trainer.max_epochs=100 \
    dataset.fasta_paths=/home/nmb127/code/data/802_genome_paths.txt \
    dataset.fasta_root=/home/nmb127/code/data/ \
    model.d_model=256 \
    model.n_layer=3 \
    model.d_inner=1024 \
    dataset.batch_size=32 \
    train.global_batch_size=256 \
    dataset.max_length=16384 \
    optimizer.lr=6e-4 \
    trainer.devices=1
