# For speed
# conda install conda-libmamba-solver
# /home/nmb127/.conda/envs/hyena-dna/bin/conda config --set solver libmamba

module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate hyena-dna


/home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
    wandb.group="hyena-16k-ep100-family-all-clas" \
    experiment=hg38/gtdb_classification trainer.max_epochs=100 \
    dataset.fasta_paths=/home/nmb127/code/data/genomes_paths.txt \
    dataset.fasta_root=/home/nmb127/code/data/ \
    model.d_model=256 \
    model.n_layer=3 \
    model.d_inner=1024 \
    dataset.batch_size=32 \
    train.global_batch_size=256 \
    dataset.max_length=16384 \
    optimizer.lr=6e-4 \
    trainer.devices=1 \
    train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-24/16-33-52-539694/checkpoints/last.ckpt

# /home/nmb127/.conda/envs/hyena-dna/bin/python huggingface.py /home/nmb127/code/hyena-dna/outputs/2024-01-24/16-33-52-539694/checkpoints/last.ckpt /home/nmb127/code/data 3


# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-family-large-clas" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/802_genome_paths.txt \
#     dataset.fasta_root=/home/nmb127/code/data/ \
#     model.d_model=256 \
#     model.n_layer=3 \
#     model.d_inner=1024 \
#     dataset.batch_size=32 \
#     train.global_batch_size=256 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-22/15-38-30-641387/checkpoints/last.ckpt


# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-longshort-short-clas" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/genomes_paths_filenames.txt \
#     dataset.fasta_root=/home/nmb127/code/data/to_copy/ \
#     model.d_model=256 \
#     model.n_layer=3 \
#     model.d_inner=1024 \
#     dataset.batch_size=16 \
#     train.global_batch_size=256 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=8 \
#     train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-19/10-39-51-872331/checkpoints/test/loss.ckpt


# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-long-short-clas" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/genomes_paths_filenames.txt \
#     dataset.fasta_root=/home/nmb127/code/data/to_copy/ \
#     model.d_model=256 \
#     model.n_layer=3 \
#     model.d_inner=1024 \
#     dataset.batch_size=32 \
#     train.global_batch_size=256 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-18/12-22-03-732858/checkpoints/test/loss.ckpt


# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-clas-no-pretrain" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/genomes_paths_filenames.txt \
#     dataset.fasta_root=/home/nmb127/code/data/to_copy/ \
#     model.d_model=256 \
#     model.n_layer=3 \
#     model.d_inner=1024 \
#     dataset.batch_size=16 \
#     train.global_batch_size=16 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=null \
#     train.pretrained_model_state_hook=null


# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-clas" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/genomes_paths_filenames.txt \
#     dataset.fasta_root=/home/nmb127/code/data/to_copy/ \
#     model.d_model=256 \
#     model.n_layer=2 \
#     model.d_inner=1024 \
#     dataset.batch_size=16 \
#     train.global_batch_size=16 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-15/11-54-06-127062/checkpoints/test/loss.ckpt


# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-clas" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/genomes_paths_filenames.txt \
#     dataset.fasta_root=/home/nmb127/code/data/to_copy/ \
#     model.d_model=256 \
#     model.n_layer=3 \
#     model.d_inner=1024 \
#     dataset.batch_size=16 \
#     train.global_batch_size=16 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-16/09-42-30-091109/checkpoints/test/loss.ckpt



# /home/nmb127/.conda/envs/hyena-dna/bin/python -m train \
#     wandb.group="hyena-16k-ep100-clas" \
#     experiment=hg38/gtdb_classification trainer.max_epochs=100 \
#     dataset.fasta_paths=/home/nmb127/code/data/genomes_paths_filenames.txt \
#     dataset.fasta_root=/home/nmb127/code/data/to_copy/ \
#     model.d_model=256 \
#     model.n_layer=4 \
#     model.d_inner=1024 \
#     dataset.batch_size=16 \
#     train.global_batch_size=16 \
#     dataset.max_length=16384 \
#     optimizer.lr=6e-4 \
#     trainer.devices=1 \
#     train.pretrained_model_path=/home/nmb127/code/hyena-dna/outputs/2024-01-16/10-00-00-599178/checkpoints/test/loss.ckpt
    
