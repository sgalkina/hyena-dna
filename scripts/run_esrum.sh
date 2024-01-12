module load cuda/11.8
module load gcc/13.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate hyena-dna

/home/projects/matrix/people/svekut/.conda/hyena-dna/bin/python -m train \
    wandb.group="hyena-tiny" \
    experiment=hg38/gtdb_hyena trainer.max_epochs=100 \
    model.d_model=128 \
    model.n_layer=2 \
    dataset.batch_size=256 \
    train.global_batch_size=256 \
    dataset.max_length=1024 \
    optimizer.lr=6e-4 \
    trainer.devices=1 \
    train.pretrained_model_path=/home/projects/matrix/data/dna_transformers/hyenadna-tiny-1k-seqlen/weights.ckpt
