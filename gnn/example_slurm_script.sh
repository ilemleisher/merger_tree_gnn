#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --job-name=[job name]              # Job name (keep to <= 8 characters)
#SBATCH --account=[account]            # Account to charge
#SBATCH --partition=gpu                 # Partition to run on
#SBATCH --ntasks=1                      # Run on a single CPU
#SBATCH --mem=64gb                       # Job memory request
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --output=out_%j.log             # Standard output and error log
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=[email]               # Where to send mail
pwd; hostname; date

module purge
module load miniforge
conda activate
module load cuda/12.4.1
conda activate env3 #<--- change to your environment with the dependencies

cd merger_tree_gnn

python gnn/GNN_Script.py data/merger_tree_graph_data SnapNum SubhaloSFR StellarMass DMMass  #<--- add or remove desired node features

date
