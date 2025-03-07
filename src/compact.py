#!/usr/bin/env python3
"""
Chess Game Evaluation Compactor

This script compacts the verbose chess evaluation data into a more efficient format.
"""

import pandas as pd
import json
import os
import sys
from tqdm import tqdm
import ast

def compact_chess_data(input_path, output_path):
    """
    Compacts chess evaluation data from a verbose format to a more efficient format.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the output CSV file
    """
    print(f"Reading data from {input_path}...")
    
    # Read the CSV in chunks to handle large files
    chunk_size = 1000
    
    # First let's check the actual column names
    sample = pd.read_csv(input_path, nrows=1)
    print(f"Actual columns in the CSV: {list(sample.columns)}")
    
    chunks = pd.read_csv(input_path, chunksize=chunk_size)
    
    all_data = []
    total_chunks = 0
    
    # Count total rows for progress bar
    # Read the first chunk to get row count and handle potential memory issues with large files
    total_rows = 0
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        total_rows += len(chunk)
        break  # Only read the first chunk to estimate
    
    # Check if output file exists and delete it
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing output file: {output_path}")
    
    # Process and save in chunks to avoid memory issues
    chunk_count = 0
    processed_total = 0
    
    with tqdm(total=total_rows, desc="Processing games") as pbar:
        for chunk_idx, chunk in enumerate(chunks):
            chunk_data = []
            
            for _, row in chunk.iterrows():
                # Process each game
                try:
                    if pd.notna(row.get('evaluations')) and pd.notna(row.get('top_moves')):
                        # Parse JSON data
                        try:
                            evaluations = json.loads(row['evaluations'])
                            top_moves = json.loads(row['top_moves'])
                            
                            # Extract compact evaluations (just the cp/mate values)
                            compact_evals = []
                            for pos in evaluations:
                                if 'eval' in pos:
                                    eval_info = pos['eval']
                                    if eval_info['type'] == 'cp':
                                        compact_evals.append(eval_info['value'])
                                    elif eval_info['type'] == 'mate':
                                        # Use #N for mate in N, -#N for mate against in N
                                        if eval_info['value'] > 0:
                                            compact_evals.append(f"#{eval_info['value']}")
                                        else:
                                            compact_evals.append(f"-#{abs(eval_info['value'])}")
                                else:
                                    compact_evals.append(None)
                            
                            # Extract compact top moves
                            compact_top_moves = []
                            for pos_moves in top_moves:
                                pos_data = []
                                for move_data in pos_moves:
                                    move = move_data.get('move', {}).get('san', '')
                                    score = move_data.get('score', {})
                                    
                                    if score.get('type') == 'cp':
                                        score_val = score.get('value', 0)
                                    elif score.get('type') == 'mate':
                                        mate_val = score.get('value', 0)
                                        if mate_val > 0:
                                            score_val = f"#{mate_val}"
                                        else:
                                            score_val = f"-#{abs(mate_val)}"
                                    else:
                                        score_val = 0
                                        
                                    pos_data.append([move, score_val])
                                compact_top_moves.append(pos_data)
                            
                            # Add to processed data
                            new_row = row.to_dict()
                            new_row['compact_evaluations'] = str(compact_evals)
                            new_row['compact_top_moves'] = str(compact_top_moves)
                            chunk_data.append(new_row)
                        except json.JSONDecodeError:
                            # Skip rows with invalid JSON
                            pass
                except Exception as e:
                    print(f"Error processing row: {e}")
                
                pbar.update(1)
            
            # Save this chunk to CSV
            if chunk_data:
                chunk_df = pd.DataFrame(chunk_data)
                
                # Drop the original columns to save space
                if 'evaluations' in chunk_df.columns:
                    chunk_df = chunk_df.drop(columns=['evaluations', 'top_moves'])
                
                write_header = (chunk_idx == 0)
                mode = 'w' if chunk_idx == 0 else 'a'
                
                chunk_df.to_csv(output_path, mode=mode, header=write_header, index=False)
                
                chunk_count += 1
                processed_count = len(chunk_data)
                processed_total += processed_count 
                print(f"Saved chunk {chunk_count} with {processed_count} games")
    
    print(f"Done! Processed {processed_total} games, saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compact Chess Evaluation Data')
    parser.add_argument('--input', type=str, 
                        default="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_evaluated.csv",
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, 
                        default="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_evaluated_pretty.csv",
                        help='Path to output CSV file')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of rows to process in each chunk')
    
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Chunk size: {args.chunk_size}")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist!")
        sys.exit(1)
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    compact_chess_data(args.input, args.output)
    
    # Chess Game Evaluation CSV Description

# This CSV file contains chess games with compact evaluation data that has been processed from Stockfish engine analysis.

# ## Key Columns

# - **game metadata**: event, site, date, white, black, result, eco, etc.
# - **moves**: The PGN notation of the moves played in the game
# - **compact_evaluations**: A list of evaluation scores for each position in the game, measured in centipawns from White's perspective (positive values favor White, negative values favor Black). Mate scores are represented as "#N" for mate in N moves.
# - **compact_top_moves**: A nested structure showing the top 3 candidate moves at each position with their evaluation scores. Format: [[move1, score1], [move2, score2], [move3, score3]]

# ## Example Usage

# This CSV provides a space-efficient way to analyze chess games with engine evaluations. The compact format makes it easier to:

# - Identify critical positions where evaluation changes significantly
# - Compare played moves against engine recommendations
# - Study patterns in different openings and players' decision making
# - Analyze game quality through evaluation stability

# The data is particularly valuable for chess researchers, coaches, and data scientists looking to extract patterns from large collections of games.