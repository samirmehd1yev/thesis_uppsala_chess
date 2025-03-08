#!/usr/bin/env python3
"""
Chess Game Feature Extraction Script

This script extracts features from evaluated chess games in the
/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_evaluated_pretty.csv
file and saves them to a new CSV with additional feature columns.

Uses the same approach as in the GameAnalyzer class to extract features.
"""

import pandas as pd
import chess
import chess.pgn
import io
import os
import sys
import logging
import ast
import numpy as np
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import traceback
import math

# Import from existing codebase
from features.extractor import FeatureExtractor
from analysis.game_analyzer import GameAnalyzer
from models.data_classes import Info, FeatureVector
from models.enums import Judgment
from analysis.phase_detector import GamePhaseDetector
from analysis.king_safety import KingSafetyEvaluator
from analysis.move_analyzer import MoveAnalyzer
from analysis.sharpness_analyzer import WdlSharpnessAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_feature_extraction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('chess_feature_extraction')

# Define file paths
INPUT_CSV = "/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_evaluated_pretty.csv"
OUTPUT_CSV = "/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_with_features.csv"

class GameFeatureExtractor:
    """Class to extract features from evaluated chess games."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.sharpness_analyzer = WdlSharpnessAnalyzer()
    
    def parse_eval_list(self, eval_str):
        """
        Parse the compact evaluations from string representation to a list.
        """
        try:
            if pd.isna(eval_str) or not eval_str:
                return []
            
            # Parse using ast.literal_eval for safety
            eval_list = ast.literal_eval(eval_str)
            return eval_list
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error parsing evaluation list: {e}")
            return []

    def parse_top_moves(self, top_moves_str):
        """
        Parse the compact top moves from string representation to a nested list.
        """
        try:
            if pd.isna(top_moves_str) or not top_moves_str:
                return []
            
            # Parse using ast.literal_eval for safety
            top_moves = ast.literal_eval(top_moves_str)
            return top_moves
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error parsing top moves: {e}")
            return []

    def create_info_objects(self, eval_list, top_moves_list):
        """
        Create Info objects from parsed evaluation and top moves lists.
        """
        info_objects = []
        
        for ply, eval_value in enumerate(eval_list):
            # Create eval dictionary first
            if isinstance(eval_value, str) and eval_value.startswith('#'):
                # Positive mate score
                mate_value = int(eval_value[1:])
                eval_dict = {"type": "mate", "value": mate_value}
            elif isinstance(eval_value, str) and eval_value.startswith('-#'):
                # Negative mate score
                mate_value = -int(eval_value[2:])
                eval_dict = {"type": "mate", "value": mate_value}
            else:
                # Regular centipawn evaluation
                try:
                    cp_value = int(eval_value) if eval_value is not None else 0
                    eval_dict = {"type": "cp", "value": cp_value}
                except (ValueError, TypeError):
                    # Default to 0 if conversion fails
                    eval_dict = {"type": "cp", "value": 0}
            
            # Create Info object with required eval parameter
            info = Info(ply=ply, eval=eval_dict)
            
            # Add top moves if available for this position
            if ply < len(top_moves_list):
                top_moves = top_moves_list[ply]
                # Extract move UCIs for the top 3 moves
                moves = []
                
                # For each [move, score] pair
                for move_data in top_moves:
                    if len(move_data) >= 2:
                        move = move_data[0]  # The move in SAN format
                        moves.append(move)
                
                info.variation = moves
            
            info_objects.append(info)
        
        return info_objects

    def pgn_to_board_positions(self, pgn_text):
        """
        Convert PGN text to a list of board positions.
        """
        try:
            # Parse PGN
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            
            if not game:
                return [], None
            
            positions = []
            board = game.board()
            
            # Add initial position
            positions.append(board.copy())
            
            # Add position after each move
            for move in game.mainline_moves():
                board.push(move)
                positions.append(board.copy())
            
            return positions, game
        except Exception as e:
            logger.error(f"Error converting PGN to board positions: {e}")
            return [], None

    def calculate_position_sharpness(self, positions, evals):
        """
        Calculate sharpness scores for all positions in the game.
        Mirrors GameAnalyzer.calculate_position_sharpness
        """
        sharpness_scores = []
        
        for i, (board, eval_info) in enumerate(zip(positions, evals)):
            if board and eval_info:
                sharpness = self.sharpness_analyzer.calculate_position_sharpness(board, eval_info)
                sharpness_scores.append(sharpness)
            else:
                # Default values for missing positions/evaluations
                sharpness_scores.append({'sharpness': 0.0, 'white_sharpness': 0.0, 'black_sharpness': 0.0})
                
        return sharpness_scores

    def calculate_move_accuracies(self, positions, evals, mainline_moves):
        """
        Calculate accuracy for each move based on the change in winning percentages.
        Mirrors GameAnalyzer._calculate_move_accuracies
        """
        move_accuracies = []
        
        # We need at least 2 positions to calculate accuracy for 1 move
        if len(positions) < 2 or len(evals) < 2 or not mainline_moves:
            return move_accuracies
            
        for i in range(len(positions) - 1):  # We calculate accuracy for each move except the last position
            # Skip if we don't have the move for this position
            if i >= len(mainline_moves):
                continue
                
            # Determine whose move it was
            player_color = "white" if positions[i].turn == chess.WHITE else "black"
            
            # Get evaluation before the move
            eval_before = evals[i]
            eval_after = evals[i+1]
            
            # Convert the eval format from {'type': 'cp', 'value': X} to {'cp': X} or {'mate': X}
            if hasattr(eval_before, 'eval') and eval_before.eval:
                eval_type_before = eval_before.eval.get('type')
                eval_value_before = eval_before.eval.get('value')
                score_dict_before = {eval_type_before: eval_value_before} if eval_type_before and eval_value_before is not None else {"cp": 0}
            else:
                score_dict_before = {"cp": 0}
                
            if hasattr(eval_after, 'eval') and eval_after.eval:
                eval_type_after = eval_after.eval.get('type')
                eval_value_after = eval_after.eval.get('value')
                score_dict_after = {eval_type_after: eval_value_after} if eval_type_after and eval_value_after is not None else {"cp": 0}
            else:
                score_dict_after = {"cp": 0}
            
            # Get win percentages from the player's perspective
            win_percent_before = MoveAnalyzer.pov_chances(player_color, score_dict_before)
            win_percent_after = MoveAnalyzer.pov_chances(player_color, score_dict_after)
            
            # Check if the move was a top engine move
            is_top_move = False
            if hasattr(eval_before, 'variation') and eval_before.variation:
                # Get the actual move played
                actual_move = mainline_moves[i]
                
                # Check if the move played matches the top engine move
                top_engine_move = eval_before.variation[0] if eval_before.variation else None
                is_top_move = actual_move.uci() == top_engine_move
            
            # Calculate accuracy for this move
            accuracy = MoveAnalyzer.calculate_move_accuracy(win_percent_before, win_percent_after, is_top_move)
            
            move_accuracies.append({
                "move_number": (i // 2) + 1 if i % 2 == 0 else i // 2 + 1,  # Chess move numbering
                "player": player_color,
                "accuracy": accuracy,
                "win_percent_before": win_percent_before * 100,  # Convert to percentage
                "win_percent_after": win_percent_after * 100,    # Convert to percentage
                "is_top_move": is_top_move,  # Add this information to the output
            })
            
        return move_accuracies

    def calculate_player_accuracies(self, move_accuracies, positions):
        """
        Calculate overall accuracy for white and black players using Lichess approach.
        Mirrors GameAnalyzer._calculate_player_accuracies
        """
        if not move_accuracies:
            return 0.0, 0.0
            
        # Get all win percentages for volatility calculation
        all_win_percents = []
        
        # Add the initial position (like Lichess's Cp.initial)
        all_win_percents.append(0.5)  # 50% win chance for starting position
        
        # Add all win percentages from moves
        for acc in move_accuracies:
            if "win_percent_before" in acc and acc["win_percent_before"] is not None:
                all_win_percents.append(acc["win_percent_before"] / 100.0)  # Convert to 0-1 range
            if "win_percent_after" in acc and acc["win_percent_after"] is not None:
                all_win_percents.append(acc["win_percent_after"] / 100.0)  # Convert to 0-1 range
                
        # Calculate window size based on game length (like Lichess)
        window_size = min(max(2, len(positions) // 10), 8)  # Between 2 and 8 positions
        
        # Create windows for volatility calculation
        windows = []
        
        # First add window_size - 2 copies of the first window
        first_window = all_win_percents[:window_size]
        for _ in range(min(window_size - 2, len(first_window) - 2)):
            windows.append(first_window)
            
        # Then add all sliding windows
        for i in range(len(all_win_percents) - window_size + 1):
            windows.append(all_win_percents[i:i+window_size])
            
        # Calculate standard deviation for each window (volatility/complexity weight)
        weights = []
        for window in windows:
            if len(window) >= 2:
                # Calculate standard deviation (Lichess's Maths.standardDeviation)
                mean = sum(window) / len(window)
                variance = sum((x - mean) ** 2 for x in window) / len(window)
                stdev = math.sqrt(variance)
                # Clamp between 0.5 and 12 (like Lichess)
                weight = max(0.5, min(12, stdev))
                weights.append(weight)
            else:
                weights.append(0.5)  # Default minimum weight
                
        # Pair moves with weights (similar to Lichess's zip(allWinPercents.sliding(2), weights))
        weighted_accuracies_by_color = {"white": [], "black": []}
        
        for i, acc in enumerate(move_accuracies):
            if i < len(weights):
                weight = weights[i]
                color = acc["player"]
                accuracy = acc["accuracy"]
                weighted_accuracies_by_color[color].append((accuracy, weight))
                
        def weighted_mean(weighted_values):
            """Calculate weighted mean like Lichess's Maths.weightedMean"""
            if not weighted_values:
                return 0.0
            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight == 0:
                return 0.0
            return sum(value * weight for value, weight in weighted_values) / total_weight
            
        def harmonic_mean(values):
            """Calculate harmonic mean like Lichess's Maths.harmonicMean"""
            if not values:
                return 0.0
            # Filter out zeros to avoid division by zero
            non_zero_values = [v for v in values if v > 0]
            if not non_zero_values:
                return 0.0
            return len(non_zero_values) / sum(1.0 / v for v in non_zero_values)
            
        # Calculate white accuracy (exactly like Lichess)
        white_weighted = weighted_mean(weighted_accuracies_by_color["white"])
        white_harmonic = harmonic_mean([acc for acc, _ in weighted_accuracies_by_color["white"]])
        white_accuracy = (white_weighted + white_harmonic) / 2
        
        # Calculate black accuracy (exactly like Lichess)
        black_weighted = weighted_mean(weighted_accuracies_by_color["black"])
        black_harmonic = harmonic_mean([acc for acc, _ in weighted_accuracies_by_color["black"]])
        black_accuracy = (black_weighted + black_harmonic) / 2
        
        return white_accuracy, black_accuracy

    def analyze_move(self, prev_info, curr_info, prev_board, curr_board, move, top_moves):
        """
        Analyze a single move using MoveAnalyzer.
        Similar to GameAnalyzer._analyze_move_worker
        """
        try:
            # Analyze the move with detailed information
            judgment, debug_reason = MoveAnalyzer.analyze_move_with_top_moves(
                prev_info, curr_info, 
                prev_board=prev_board, 
                curr_board=curr_board, 
                move=move,
                top_moves=top_moves,
                debug=True
            )
            
            return judgment
        except Exception as e:
            logger.error(f"Error analyzing move: {e}")
            return Judgment.GOOD  # Default to GOOD on error

    def process_game(self, game_row):
        """
        Process a single game and extract features.
        """
        try:
            # Extract data from row
            pgn_text = game_row.get('moves', '')
            eval_list = self.parse_eval_list(game_row.get('compact_evaluations', '[]'))
            top_moves_list = self.parse_top_moves(game_row.get('compact_top_moves', '[]'))
            
            # Skip games with no moves or evaluations
            if not pgn_text or not eval_list:
                logger.warning(f"Skipping game: Missing moves or evaluations")
                return None
            
            # Convert PGN to board positions and game object
            positions, game = self.pgn_to_board_positions(pgn_text)
            
            # Skip games with parsing errors
            if not positions or not game:
                logger.warning(f"Skipping game: Failed to parse positions")
                return None
            
            # Create Info objects
            info_objects = self.create_info_objects(eval_list, top_moves_list)
            
            # Skip games with mismatched evaluation counts
            if len(positions) != len(info_objects):
                logger.warning(f"Skipping game: Mismatched position and evaluation counts "
                              f"(positions: {len(positions)}, evals: {len(info_objects)})")
                return None
            
            # Get the actual moves played
            mainline_moves = list(game.mainline_moves())
            
            # Calculate judgments for moves
            judgments = []
            for i in range(1, len(info_objects)):
                if i-1 >= len(mainline_moves):
                    break
                
                prev_info = info_objects[i-1]
                curr_info = info_objects[i]
                prev_board = positions[i-1]
                curr_board = positions[i]
                move = mainline_moves[i-1]
                
                # Get top moves if available
                top_moves = prev_info.variation if hasattr(prev_info, 'variation') else None
                
                # Analyze the move
                judgment = self.analyze_move(prev_info, curr_info, prev_board, curr_board, move, top_moves)
                judgments.append(judgment)
            
            # Extract features using the feature extractor
            features = self.feature_extractor.extract_features(game, info_objects, judgments)
            
            # Calculate sharpness scores
            sharpness_scores = self.calculate_position_sharpness(positions, info_objects)
            cumulative_sharpness = self.sharpness_analyzer.calculate_cumulative_sharpness(sharpness_scores)
            
            # Calculate move accuracies
            move_accuracies = self.calculate_move_accuracies(positions, info_objects, mainline_moves)
            
            # Calculate overall player accuracies
            white_accuracy, black_accuracy = self.calculate_player_accuracies(move_accuracies, positions)
            
            # Update accuracy values in features
            features.white_accuracy = white_accuracy
            features.black_accuracy = black_accuracy
            
            # Add features to game data
            result = game_row.copy()
            for key, value in features.__dict__.items():
                result[key] = value
            
            # Add summary sharpness metrics
            result['overall_sharpness'] = cumulative_sharpness['sharpness']
            result['white_sharpness'] = cumulative_sharpness['white_sharpness']
            result['black_sharpness'] = cumulative_sharpness['black_sharpness']
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            traceback.print_exc()
            return None

def process_chunk(chunk_df, extractor, num_processes=1):
    """Process a chunk of games in parallel."""
    # Convert DataFrame to list of dictionaries for parallel processing
    games_data = chunk_df.to_dict('records')
    
    # Process games in parallel
    processed_chunk = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for result in tqdm(executor.map(extractor.process_game, games_data), 
                          total=len(games_data), 
                          desc="Processing games"):
            if result is not None:
                processed_chunk.append(result)
    
    return processed_chunk

def main():
    """Main function to extract features from games."""
    start_time = datetime.now()
    logger.info(f"Starting feature extraction at {start_time}")
    
    # Initialize the feature extractor
    extractor = GameFeatureExtractor()
    
    # Read input CSV
    logger.info(f"Reading input CSV from {INPUT_CSV}")
    try:
        # Read in chunks for memory efficiency
        chunk_size = 1000
        chunks = pd.read_csv(INPUT_CSV, chunksize=chunk_size)
        total_games = 0
        processed_games = 0
        output_exists = False
        
        # Get CPU count for parallel processing
        num_processes = min(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {num_processes} processes for parallel processing")
        
        # Columns to exclude from output
        columns_to_exclude = [
            'time_control', 'import_date', 'source', 'moves', 'eval_info', 
            'clock_info', 'avg_elo', 'elo_difference', 'move_count', 
            'has_clock_info', 'has_eval_info', 'has_analysis', 
            'compact_evaluations', 'compact_top_moves'
        ]
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx+1} with {len(chunk)} games")
            
            # Process the chunk in parallel
            processed_chunk = process_chunk(chunk, extractor, num_processes)
            
            total_games += len(chunk)
            processed_games += len(processed_chunk)
            
            # Convert back to DataFrame
            if processed_chunk:
                result_df = pd.DataFrame(processed_chunk)
                
                # Remove excluded columns if they exist
                columns_to_drop = [col for col in columns_to_exclude if col in result_df.columns]
                if columns_to_drop:
                    result_df = result_df.drop(columns=columns_to_drop)
                    logger.info(f"Dropped columns: {columns_to_drop}")
                
                # Save to CSV (append mode after first chunk)
                mode = 'a' if output_exists else 'w'
                header = not output_exists
                result_df.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)
                output_exists = True
                
                logger.info(f"Saved {len(processed_chunk)} games to CSV")
            
            # Log progress
            logger.info(f"Processed {processed_games}/{total_games} games so far")
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        logger.info(f"Feature extraction completed at {end_time}")
        logger.info(f"Total processing time: {processing_time}")
        logger.info(f"Total games processed: {processed_games}/{total_games}")
        logger.info(f"Results saved to: {OUTPUT_CSV}")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()