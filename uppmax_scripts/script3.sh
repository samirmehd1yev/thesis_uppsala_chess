#!/bin/bash
#SBATCH -A uppmax2025-2-265
#SBATCH -p node
#SBATCH -n 40
#SBATCH -t 06:00:00
#SBATCH -J chess_analysis
#SBATCH --output=logs/chess_analysis_%j.log
#SBATCH --error=logs/chess_analysis_%j.err

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
CSV_PATH="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_clean_1950_final.csv"
OUTPUT_DIR="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase"
STOCKFISH_PATH="/crex/proj/chess/stockfish/src/stockfish"


# Run the script first with 500 games to test performance
python3 /proj/chess/thesis_uppsala_chess_samir/src/analyze_chess_games.py \
  --csv "$CSV_PATH" \
  --output "$OUTPUT_DIR" \
  --stockfish "$STOCKFISH_PATH" \
  --depth 18 \
  --processes 40 \
  --threads 1 \
  --hash 128 \
  --batch 100 \
  --limit 500

# Print end time
echo "End time: $(date)"
echo "CPU times used:"
uptime

# Deactivate virtual environment
deactivate