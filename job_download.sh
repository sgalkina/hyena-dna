#!/bin/bash
#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

# ./curl_gtdb.sh
curl -o /home/nmb127/code/data/gtdb_genomes_reps_r207-2.tar.gz https://data.gtdb.ecogenomic.org/releases/release207/207.0/genomic_files_reps/gtdb_genomes_reps_r207.tar.gz
