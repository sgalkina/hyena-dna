# For speed
# conda install conda-libmamba-solver
# /home/nmb127/.conda/envs/hyena-dna/bin/conda config --set solver libmamba

module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate hyena-dna

/home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
    wandb.group="16k-all-continue" \
    experiment=hg38/gtdb_hyena trainer.max_epochs=100 \
    dataset.fasta_paths=/home/nmb127/code/data/genomes_paths.txt \
    dataset.fasta_root=/home/nmb127/code/data/ \
    model.d_model=256 \
    model.n_layer=6 \
    model.d_inner=1024 \
    dataset.batch_size=16 \
    train.global_batch_size=256 \
    dataset.max_length=16384 \
    optimizer.lr=6e-4 \
    trainer.devices=1 \
    train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-02-26/14-52-00-272725/checkpoints/last.ckpt
    # train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-02-07/11-48-54-696866/checkpoints/last.ckpt
    # train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-31/09-25-20-255142/checkpoints/test/loss.ckpt
    # train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-27/13-16-28-929017/checkpoints/test/loss.ckpt
    # train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-26/10-12-56-210089/checkpoints/test/loss.ckpt
    # train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-24/16-33-52-539694/checkpoints/last.ckpt
