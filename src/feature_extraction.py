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
import argparse

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
INPUT_CSV = "../data/processed/lumbrasgigabase/chess_games_evaluated_pretty.csv"
OUTPUT_CSV = "../data/processed/lumbrasgigabase/chess_games_with_features.csv"

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
        Format in CSV: [[move1, score1], [move2, score2], [move3, score3]]
        """
        try:
            if pd.isna(top_moves_str) or not top_moves_str:
                return []
            
            # Parse using ast.literal_eval for safety
            top_moves = ast.literal_eval(top_moves_str)
            
            # Verify format and log errors if needed
            if not isinstance(top_moves, list):
                logger.warning(f"Invalid top moves format: expected list, got {type(top_moves)}")
                return []
                
            # Add verbose logging for debugging
            for i, pos_moves in enumerate(top_moves[:3]):
                if pos_moves:
                    logger.debug(f"Position {i} top moves (first 3): {pos_moves[:3]}")
            
            return top_moves
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error parsing top moves: {e}")
            return []

    def cp_to_wdl(self, eval_dict: dict, ply: int = 30) -> dict:
        """
        Convert centipawn/mate evaluation to WDL (Win/Draw/Loss) probabilities
        using chess.engine.Score's built-in WDL calculation.
        
        Args:
            eval_dict: Evaluation dictionary with 'type' and 'value' keys
            ply: Current ply (used for WDL calculation scaling)
            
        Returns:
            Dictionary with 'wins', 'draws', and 'losses' probabilities (0-1 range)
        """
        # Import chess.engine here for Score
        import chess.engine
        
        # Default result with 1.0 probability for draws
        result = {"wins": 0.0, "draws": 1.0, "losses": 0.0}
        
        try:
            # Convert our eval_dict to a chess.engine.Score object
            score = None
            if eval_dict.get('type') == 'mate':
                mate_value = eval_dict.get('value', 0)
                score = chess.engine.Mate(mate_value)
            elif eval_dict.get('type') == 'cp':
                cp_value = eval_dict.get('value', 0)
                score = chess.engine.Cp(cp_value)
            
            if score is not None:
                # Use the built-in WDL calculation from the chess library
                # This is the same model Stockfish uses
                wdl_model = "sf"  # Use Stockfish's WDL model
                wdl = score.wdl(model=wdl_model, ply=ply)
                
                # The WDL values from Stockfish are in permille (0-1000)
                # Convert to 0-1 range for better handling in sharpness calculation
                return {
                    "wins": wdl.wins / 1000.0,
                    "draws": wdl.draws / 1000.0,
                    "losses": wdl.losses / 1000.0
                }
        except Exception as e:
            logger.error(f"Error calculating WDL: {e}")
        
        return result

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
                logger.debug(f"Position {ply}: Mate in {mate_value} for White")
            elif isinstance(eval_value, str) and eval_value.startswith('-#'):
                # Negative mate score
                mate_value = -int(eval_value[2:])
                eval_dict = {"type": "mate", "value": mate_value}
                logger.debug(f"Position {ply}: Mate in {-mate_value} for Black")
            else:
                # Regular centipawn evaluation
                try:
                    cp_value = int(eval_value) if eval_value is not None else 0
                    eval_dict = {"type": "cp", "value": cp_value}
                except (ValueError, TypeError):
                    # Default to 0 if conversion fails
                    eval_dict = {"type": "cp", "value": 0}
                    logger.warning(f"Position {ply}: Could not parse evaluation: {eval_value}, defaulting to 0")
            
            # Calculate WDL from evaluation
            wdl_dict = self.cp_to_wdl(eval_dict, ply)
            
            # Create Info object with required eval parameter
            info = Info(ply=ply, eval=eval_dict, wdl=wdl_dict)
            
            # Add top moves if available for this position
            if ply < len(top_moves_list) and top_moves_list[ply]:
                top_moves = top_moves_list[ply]
                # Extract the move strings (first element of each pair)
                moves = []
                
                # Create multipv structure for is_only_good_move function
                multipv_data = []
                
                # Format of compact_top_moves: [[move1, score1], [move2, score2], [move3, score3]]
                for move_data in top_moves:
                    if isinstance(move_data, list) and len(move_data) >= 2:
                        # Get the move string and score
                        move_str = str(move_data[0])
                        score_value = move_data[1]
                        moves.append(move_str)
                        
                        # Create score dict for multipv
                        if isinstance(score_value, str) and score_value.startswith('#'):
                            # Positive mate score
                            mate_value = int(score_value[1:])
                            score_dict = {"mate": mate_value}
                        elif isinstance(score_value, str) and score_value.startswith('-#'):
                            # Negative mate score
                            mate_value = -int(score_value[2:])
                            score_dict = {"mate": mate_value}
                        else:
                            # Regular centipawn evaluation
                            try:
                                cp_value = int(score_value) if score_value is not None else 0
                                score_dict = {"cp": cp_value}
                            except (ValueError, TypeError):
                                # Default to 0 if conversion fails
                                score_dict = {"cp": 0}
                        
                        # Add to multipv list
                        multipv_data.append({"move": move_str, "score": score_dict})
                
                # Log the extracted top moves for debugging
                if moves:
                    logger.debug(f"Position {ply}: Top moves: {moves}")
                
                info.variation = moves
                
                # Set multipv data if we have it
                if multipv_data:
                    info.multipv = multipv_data
            else:
                logger.debug(f"Position {ply}: No top moves available")
            
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
            # Debug move and top moves
            self.debug_move_analysis(prev_info, prev_board, move, top_moves)
            
            # Analyze the move with detailed information
            judgment, debug_reason = MoveAnalyzer.analyze_move_with_top_moves(
                prev_info, curr_info, 
                prev_board=prev_board, 
                curr_board=curr_board, 
                move=move,
                top_moves=top_moves,
                debug=True
            )
            
            # Log the judgment for debugging
            logger.debug(f"Move {move.uci()} judgment: {judgment.name} - {debug_reason}")
            
            return judgment
        except Exception as e:
            logger.error(f"Error analyzing move: {e}")
            traceback.print_exc()
            return Judgment.GOOD  # Default to GOOD on error
            
    def debug_move_analysis(self, prev_info, prev_board, move, top_moves):
        """Debug helper for move analysis issues"""
        if not move or not top_moves:
            return
            
        move_uci = move.uci()
        is_match = False
        
        # Check for direct match with UCI format
        if move_uci in top_moves:
            is_match = True
            logger.debug(f"MATCH (UCI): Move {move_uci} found in top moves: {top_moves}")
        
        # Check for potential SAN format match
        elif prev_board:
            try:
                move_san = prev_board.san(move)
                if move_san in top_moves:
                    is_match = True
                    logger.debug(f"MATCH (SAN): Move {move_san} found as SAN in top moves: {top_moves}")
            except Exception as e:
                logger.warning(f"Error converting move to SAN: {e}")
        
        # Log debug info if no match
        if not is_match:
            logger.debug(f"NO MATCH: Played move {move_uci} not found in top moves: {top_moves}")
            if top_moves:
                logger.debug(f"Top move formats - UCI: {move_uci}, SAN: {prev_board.san(move) if prev_board else 'N/A'}")
                logger.debug(f"First top move type: {type(top_moves[0])}, value: {top_moves[0]}")
        
        return is_match

    def process_game(self, game_row):
        """
        Process a single game and extract features.
        """
        try:
            # Extract data from row
            pgn_text = game_row.get('moves', '')
            eval_list = self.parse_eval_list(game_row.get('compact_evaluations', '[]'))
            # logger.debug(f"DEBUG: eval_list: {eval_list}")
            top_moves_list = self.parse_top_moves(game_row.get('compact_top_moves', '[]'))
            
            # Skip games with no moves or evaluations
            if not pgn_text or not eval_list:
                logger.warning(f"Skipping game: Missing moves or evaluations")
                return None
            
            # Convert PGN to board positions and game object
            positions, game = self.pgn_to_board_positions(pgn_text)
            # logger.debug(f"DEBUG: positions: {positions}")
            
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract features from chess games')
    parser.add_argument('--test', action='store_true', help='Process only the first game for testing')
    parser.add_argument('--debug', action='store_true', help='Print detailed debug information like in test.py')
    parser.add_argument('--input', type=str, default=INPUT_CSV, help='Input CSV file path')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV, help='Output CSV file path')
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting feature extraction at {start_time}")
    
    # Configure logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - verbose logging activated")
    
    # Initialize the feature extractor
    extractor = GameFeatureExtractor()
    
    # Read input CSV
    input_csv = args.input
    output_csv = args.output
    logger.info(f"Reading input CSV from {input_csv}")
    try:
        # For debug mode, read the entire CSV and process a specific row
        if args.debug:
            logger.info("Debug mode: Reading entire CSV file")
            df = pd.read_csv(input_csv)
            row_to_process = 22 # Default row number for debug mode
            logger.info(f"Processing row {row_to_process} in debug mode")
            if row_to_process < len(df):
                game_row = df.iloc[row_to_process].to_dict()
                result = extractor.process_game(game_row)
                processed_games = 1 if result is not None else 0
                processed_chunk = [result] if result is not None else []
                
                # Print detailed info in debug mode
                if result is not None:
                    print_game_analysis(result, game_row)
                    
                # Save to CSV
                if processed_chunk:
                    # Columns to exclude from output
                    columns_to_exclude = [
                        'time_control', 'import_date', 'source', 'moves', 'eval_info', 
                        'clock_info', 'avg_elo', 'elo_difference', 'move_count', 
                        'has_clock_info', 'has_eval_info', 'has_analysis', 
                        'compact_evaluations', 'compact_top_moves'
                    ]
                    
                    result_df = pd.DataFrame(processed_chunk)
                    
                    # Remove excluded columns if they exist
                    columns_to_drop = [col for col in columns_to_exclude if col in result_df.columns]
                    if columns_to_drop:
                        result_df = result_df.drop(columns=columns_to_drop)
                        logger.info(f"Dropped columns: {columns_to_drop}")
                    
                    # Save to CSV
                    # result_df.to_csv(output_csv, mode='w', header=True, index=False)
                    # logger.info(f"Saved debug game to CSV at {output_csv}")
            else:
                logger.error(f"Row {row_to_process} does not exist in the CSV. CSV has {len(df)} rows.")
                return
                
            # Set total for debug mode
            total_games = 1
        else:
            # Read in chunks for memory efficiency
            chunk_size = 1 if args.test else 1000
            chunks = pd.read_csv(input_csv, chunksize=chunk_size)
            total_games = 0
            processed_games = 0
            output_exists = False
            
            # Get CPU count for parallel processing
            num_processes = 1 if args.test else max(1, multiprocessing.cpu_count() - 1)
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
                
                # Process the chunk
                if args.test:
                    # Process just the first game directly for testing
                    first_game = chunk.iloc[0].to_dict()
                    result = extractor.process_game(first_game)
                    processed_chunk = [result] if result is not None else []
                else:
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
                    result_df.to_csv(output_csv, mode=mode, header=header, index=False)
                    output_exists = True
                    
                    logger.info(f"Saved {len(processed_chunk)} games to CSV")
                
                # Log progress
                logger.info(f"Processed {processed_games}/{total_games} games so far")
                
                # Exit after first chunk if in test mode
                if args.test:
                    logger.info("Test mode: exiting after processing first game")
                    break
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        logger.info(f"Feature extraction completed at {end_time}")
        logger.info(f"Total processing time: {processing_time}")
        logger.info(f"Total games processed: {processed_games}/{total_games}")
        logger.info(f"Results saved to: {output_csv}")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)

def print_game_analysis(result, game_data):
    """Print detailed analysis of a game similar to test.py"""
    try:
        # Try to import colorama for colored output
        from colorama import Fore, Back, Style, init
        init()  # Initialize colorama
    except ImportError:
        # Create dummy colorama classes if not available
        class DummyFore:
            def __getattr__(self, name):
                return ""
        class DummyStyle:
            def __getattr__(self, name):
                return ""
        Fore = DummyFore()
        Style = DummyStyle()
    
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}GAME ANALYSIS{Style.RESET_ALL}")
    print("="*80)
    
    # Print game metadata
    print(f"{Fore.CYAN}{Style.BRIGHT}Game Metadata:{Style.RESET_ALL}")
    for key in ['white', 'black', 'result', 'event', 'date', 'eco']:
        if key in game_data:
            print(f"  {key.title()}: {game_data[key]}")
    
    # Print feature summary
    print_feature_summary(result)
    
    # Print judgment summary if available
    if 'white_brilliant_count' in result:
        print_judgment_summary(result)
    
    # Print sharpness summary if available
    if 'overall_sharpness' in result and 'white_sharpness' in result and 'black_sharpness' in result:
        print_sharpness_summary({
            'sharpness': result['overall_sharpness'],
            'white_sharpness': result['white_sharpness'],
            'black_sharpness': result['black_sharpness']
        })
    
    print("\n" + "="*80)

def print_feature_summary(features):
    """Print the feature summary with formatting"""
    try:
        from colorama import Fore, Back, Style, init
        init()
    except ImportError:
        class DummyFore:
            def __getattr__(self, name):
                return ""
        class DummyStyle:
            def __getattr__(self, name):
                return ""
        Fore = DummyFore()
        Style = DummyStyle()
    
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}GAME FEATURE SUMMARY{Style.RESET_ALL}")
    print("="*80)
    
    # Organize features into categories
    categories = {
        "Game Phase": [
            "total_moves", "opening_length", "middlegame_length", "endgame_length"
        ],
        "Material/Position - White": [
            "white_material_changes", "white_piece_mobility_avg", 
            "white_pawn_structure_changes", "white_center_control_avg"
        ],
        "Material/Position - Black": [
            "black_material_changes", "black_piece_mobility_avg", 
            "black_pawn_structure_changes", "black_center_control_avg"
        ],
        "White Move Quality": [
            "white_brilliant_count", "white_great_count", "white_good_moves",
            "white_inaccuracy_count", "white_mistake_count", "white_blunder_count",
            "white_sacrifice_count", "white_avg_eval_change", "white_eval_volatility",
            "white_accuracy"
        ],
        "Black Move Quality": [
            "black_brilliant_count", "black_great_count", "black_good_moves",
            "black_inaccuracy_count", "black_mistake_count", "black_blunder_count",
            "black_sacrifice_count", "black_avg_eval_change", "black_eval_volatility",
            "black_accuracy"
        ]
    }
    
    # Print each category of features
    for category, feature_names in categories.items():
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{category}:{Style.RESET_ALL}")
        for name in feature_names:
            if name in features:
                value = features[name]
                
                # Apply color formatting
                if "brilliant" in name or "great" in name:
                    # Highlight brilliant/great moves in green
                    formatted_value = f"{Fore.GREEN}{int(value)}{Style.RESET_ALL}"
                elif "inaccuracy" in name:
                    # Highlight inaccuracies in yellow
                    formatted_value = f"{Fore.YELLOW}{int(value)}{Style.RESET_ALL}"
                elif "mistake" in name or "blunder" in name:
                    # Highlight mistakes/blunders in red
                    formatted_value = f"{Fore.RED}{int(value)}{Style.RESET_ALL}"
                elif "accuracy" in name:
                    # Format accuracy with color based on value
                    acc_value = float(value)
                    if acc_value >= 90:
                        color = Fore.GREEN
                    elif acc_value >= 80:
                        color = Fore.CYAN
                    elif acc_value >= 70:
                        color = Fore.BLUE
                    elif acc_value >= 60:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                    formatted_value = f"{color}{acc_value:.1f}%{Style.RESET_ALL}"
                elif name.endswith('_count') or name.endswith('_changes') or name == 'total_moves':
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{float(value):.2f}"
                
                # Format the name nicely
                pretty_name = name.replace('_', ' ').title()
                print(f"  {pretty_name}: {formatted_value}")
            else:
                print(f"  {name.replace('_', ' ').title()}: N/A")

def print_judgment_summary(features):
    """Print a summary of the move judgments"""
    try:
        from colorama import Fore, Back, Style, init
        init()
    except ImportError:
        class DummyFore:
            def __getattr__(self, name):
                return ""
        class DummyStyle:
            def __getattr__(self, name):
                return ""
        Fore = DummyFore()
        Style = DummyStyle()
    
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}MOVE JUDGMENT SUMMARY{Style.RESET_ALL}")
    print("="*80)
    
    # White judgments
    white_total = sum([
        features.get('white_brilliant_count', 0),
        features.get('white_great_count', 0),
        features.get('white_good_moves', 0),
        features.get('white_inaccuracy_count', 0),
        features.get('white_mistake_count', 0),
        features.get('white_blunder_count', 0)
    ])
    
    # Black judgments
    black_total = sum([
        features.get('black_brilliant_count', 0),
        features.get('black_great_count', 0),
        features.get('black_good_moves', 0),
        features.get('black_inaccuracy_count', 0),
        features.get('black_mistake_count', 0),
        features.get('black_blunder_count', 0)
    ])
    
    # Print White judgments
    print(f"\n{Fore.WHITE}{Style.BRIGHT}White Moves:{Style.RESET_ALL}")
    if white_total > 0:
        brilliant = features.get('white_brilliant_count', 0)
        great = features.get('white_great_count', 0)
        good = features.get('white_good_moves', 0)
        inaccuracy = features.get('white_inaccuracy_count', 0)
        mistake = features.get('white_mistake_count', 0)
        blunder = features.get('white_blunder_count', 0)
        
        print(f"  {Fore.GREEN}Brilliant: {brilliant} ({brilliant*100/white_total:.1f}%){Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Great: {great} ({great*100/white_total:.1f}%){Style.RESET_ALL}")
        print(f"  Good: {good} ({good*100/white_total:.1f}%)")
        print(f"  {Fore.YELLOW}Inaccuracy: {inaccuracy} ({inaccuracy*100/white_total:.1f}%){Style.RESET_ALL}")
        print(f"  {Fore.RED}Mistake: {mistake} ({mistake*100/white_total:.1f}%){Style.RESET_ALL}")
        print(f"  {Fore.RED}Blunder: {blunder} ({blunder*100/white_total:.1f}%){Style.RESET_ALL}")
    else:
        print("  No move judgments available")
    
    # Print Black judgments
    print(f"\n{Fore.BLACK}{Style.BRIGHT}Black Moves:{Style.RESET_ALL}")
    if black_total > 0:
        brilliant = features.get('black_brilliant_count', 0)
        great = features.get('black_great_count', 0)
        good = features.get('black_good_moves', 0)
        inaccuracy = features.get('black_inaccuracy_count', 0)
        mistake = features.get('black_mistake_count', 0)
        blunder = features.get('black_blunder_count', 0)
        
        print(f"  {Fore.GREEN}Brilliant: {brilliant} ({brilliant*100/black_total:.1f}%){Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Great: {great} ({great*100/black_total:.1f}%){Style.RESET_ALL}")
        print(f"  Good: {good} ({good*100/black_total:.1f}%)")
        print(f"  {Fore.YELLOW}Inaccuracy: {inaccuracy} ({inaccuracy*100/black_total:.1f}%){Style.RESET_ALL}")
        print(f"  {Fore.RED}Mistake: {mistake} ({mistake*100/black_total:.1f}%){Style.RESET_ALL}")
        print(f"  {Fore.RED}Blunder: {blunder} ({blunder*100/black_total:.1f}%){Style.RESET_ALL}")
    else:
        print("  No move judgments available")

def print_sharpness_summary(cumulative_sharpness):
    """Print the sharpness analysis summary"""
    try:
        from colorama import Fore, Back, Style, init
        init()
    except ImportError:
        class DummyFore:
            def __getattr__(self, name):
                return ""
        class DummyStyle:
            def __getattr__(self, name):
                return ""
        Fore = DummyFore()
        Style = DummyStyle()
    
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}SHARPNESS ANALYSIS{Style.RESET_ALL}")
    print("="*80)
    
    def sharpness_color(value):
        """Return appropriate color for sharpness value"""
        if value > 80:
            return Fore.RED  # Very sharp
        elif value > 60:
            return Fore.YELLOW  # Sharp
        elif value > 40:
            return Fore.GREEN  # Moderately sharp
        else:
            return Fore.BLUE  # Not sharp
    
    overall = cumulative_sharpness.get('sharpness', 0)
    white = cumulative_sharpness.get('white_sharpness', 0)
    black = cumulative_sharpness.get('black_sharpness', 0)
    
    print(f"  Overall Sharpness: {sharpness_color(overall)}{overall:.1f}{Style.RESET_ALL}")
    print(f"  White Sharpness: {sharpness_color(white)}{white:.1f}{Style.RESET_ALL}")
    print(f"  Black Sharpness: {sharpness_color(black)}{black:.1f}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()