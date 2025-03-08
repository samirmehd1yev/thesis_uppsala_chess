#!/bin/bash
#SBATCH -A uppmax2025-2-265
#SBATCH -p node
#SBATCH -n 20
#SBATCH -t 06:00:00
#SBATCH -J chess_eval
#SBATCH --output=logs/chess_eval_%j.log
#SBATCH --error=logs/chess_eval_%j.err

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

# Define paths
INPUT_CSV="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_cleaned_final2_only_players_random_100.csv"
OUTPUT_CSV="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_evaluated.csv"
STOCKFISH_PATH="/crex/proj/chess/stockfish/src/stockfish"

# Run the evaluation script
python3 /proj/chess/thesis_uppsala_chess_samir/src/chess_evaluation_pipeline.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_CSV" \
    --stockfish "$STOCKFISH_PATH" \
    --depth 20 \
    --multipv 3 \
    --workers 20 \
    --threads 1 \
    --hash 128 \
    --chunk-size 100 \
    --resume

# Print end time
echo "End time: $(date)"
echo "CPU times used:"
uptime

# Deactivate virtual environment
deactivate