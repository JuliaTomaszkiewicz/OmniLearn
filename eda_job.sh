#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --job-name=alice_eda
#SBATCH --output=alice_eda_%j.out

module load python

echo "Starting analysis..."
python -u eda.py --FILES_TO_ANALYSE "../data/run_oo_batch_*/AnalysisResults_trees.root"
python -u eda.py --FILES_TO_ANALYSE "../data/run_pbpb_batch_*/AnalysisResults_trees.root"
echo "Analysis complete."
