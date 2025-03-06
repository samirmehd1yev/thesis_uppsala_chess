#!/bin/bash
#SBATCH -A uppmax2025-2-265
#SBATCH -p node
#SBATCH -n 20
#SBATCH -t 02:00:00
#SBATCH -J chess_analysis
#SBATCH --output=logs/chess_db_analysis.log
#SBATCH --error=logs/chess_db_analysis.err

# Load required modules
module load python3/3.12.7

# Activate virtual environment
source /proj/chess/thesis_chess_code/my_env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print system info
echo "Running on: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Start time: $(date)"

# Run the fixed Python script
cd /proj/chess/thesis_uppsala_chess_samir/src
python3 test.py --depth 18 --cpus 20 --stockfish /crex/proj/chess/stockfish/src/stockfish

# Print end time
echo "End time: $(date)"

# Deactivate virtual environment
deactivate
