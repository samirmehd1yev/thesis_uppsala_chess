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
from models.enums import Judgment
from analysis.move_analyzer import MoveAnalyzer
from analysis.sharpness_analyzer import WdlSharpnessAnalyzer
from tools.eco_database_loader import eco_loader
from models.data_classes import Info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/chess_feature_extraction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('chess_feature_extraction')

# Define file paths
INPUT_CSV = "../data/processed/lumbrasgigabase/chess_games_evaluated_pretty.csv"
OUTPUT_CSV = "../data/processed/lumbrasgigabase/chess_games_with_features.csv"

class GameFeatureExtractor:
    """Class to extract features from evaluated chess games."""
    
    def __init__(self, wdl_engine="lc0"):
        """
        Initialize the feature extractor with engine options.
        
        Args:
            wdl_engine: Engine to use for WDL calculations. Options: "lc0" or "sf"
        """
        self.feature_extractor = FeatureExtractor()
        self.wdl_engine = wdl_engine
        
        # Initialize sharpness analyzer based on the selected engine
        if wdl_engine.lower() == "lc0":
            self.logger = logging.getLogger('chess_feature_extraction')
            self.logger.info("Using Leela Chess Zero for WDL calculations")
            self.sharpness_analyzer = WdlSharpnessAnalyzer(nodes=10000)
        elif wdl_engine.lower() == "sf":
            self.logger = logging.getLogger('chess_feature_extraction')
            self.logger.info("Using Stockfish for WDL calculations")
            # For Stockfish WDL, we'll use the chess.engine.Score WDL model
            # Create a partial sharpness analyzer that only has the calculate_sharpness method
            from types import SimpleNamespace
            self.sharpness_analyzer = SimpleNamespace(
                calculate_sharpness=self._calculate_sharpness_from_wdl
            )
        else:
            raise ValueError(f"Unknown WDL engine: {wdl_engine}. Use 'lc0' or 'sf'.")
            
        # Load ECO database for opening identification
        self.eco_loader = eco_loader  # Use the singleton instance
    
    def _calculate_sharpness_from_wdl(self, board, wdl):
        """
        Calculate sharpness score using the same formula as in WdlSharpnessAnalyzer.
        
        Args:
            board: Chess board position
            wdl: Tuple of (win, draw, loss) probabilities in range 0-1
            
        Returns:
            Float representing the sharpness score
        """
        # Use the same formula as in WdlSharpnessAnalyzer
        W = min(max(wdl[0], 0.0001), 0.9999)
        L = min(max(wdl[2], 0.0001), 0.9999)
        
        try:
            return (max(2/(np.log((1/W)-1) + np.log((1/L)-1)), 0))**2
        except Exception as e:
            self.logger.error(f"Error calculating sharpness: {e}")
            return 0.0
            
    def calculate_cumulative_sharpness(self, sharpness_scores):
        """
        Calculate cumulative sharpness over positions.
        Same logic as WdlSharpnessAnalyzer.calculate_cumulative_sharpness.
        
        Args:
            sharpness_scores: List of sharpness score dictionaries
            
        Returns:
            Dictionary with cumulative sharpness scores
        """
        if not sharpness_scores:
            return {'sharpness': 0.0, 'white_sharpness': 0.0, 'black_sharpness': 0.0}
        
        # Separate scores for white and black positions
        white_positions = []
        black_positions = []
        
        for i, score in enumerate(sharpness_scores):
            # Even ply numbers (0, 2, 4...) are positions where it's White's turn to move
            # Odd ply numbers (1, 3, 5...) are positions where it's Black's turn to move
            if i % 2 == 0:
                white_positions.append(score.get('sharpness', 0.0))
            else:
                black_positions.append(score.get('sharpness', 0.0))
        
        # Calculate cumulative values
        white_cumulative = sum(white_positions)
        black_cumulative = sum(black_positions)
        
        # Overall cumulative is the sum of white and black cumulative values
        overall_cumulative = white_cumulative + black_cumulative
        
        return {
            'sharpness': overall_cumulative,
            'white_sharpness': white_cumulative,
            'black_sharpness': black_cumulative
        }

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
            self.logger.error(f"Error calculating WDL: {e}")
        
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
                
                # # Log the extracted top moves for debugging
                # if moves:
                #     logger.debug(f"Position {ply}: Top moves: {moves}")
                
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
                if self.wdl_engine.lower() == "lc0":
                    # Use LC0 for WDL and sharpness calculation
                    sharpness = self.sharpness_analyzer.calculate_position_sharpness(board, eval_info)
                    sharpness_scores.append(sharpness)
                else:
                    # Use Stockfish WDL model 
                    if hasattr(eval_info, 'eval') and eval_info.eval:
                        # Calculate WDL using chess.engine.Score
                        wdl_dict = self.cp_to_wdl(eval_info.eval, i)
                        
                        # Calculate sharpness
                        wins = wdl_dict.get('wins', 0)
                        draws = wdl_dict.get('draws', 0)
                        losses = wdl_dict.get('losses', 0)
                        
                        if self.sharpness_analyzer:
                            # If we have a sharpness analyzer, use it
                            sharpness = self.sharpness_analyzer.calculate_sharpness(board, (wins, draws, losses))
                        else:
                            # Otherwise use a basic entropy-based formula
                            W = min(max(wins, 0.0001), 0.9999)
                            L = min(max(losses, 0.0001), 0.9999)
                            try:
                                sharpness = (max(2/(np.log((1/W)-1) + np.log((1/L)-1)), 0))**2
                                # print(f"Sharpness: {sharpness}")
                            except:
                                # print(f"Error calculating sharpness: {e}")
                                sharpness = 0.0
                        
                        # Create result dict
                        result = {
                            'sharpness': sharpness,
                            'white_sharpness': sharpness if board.turn == chess.WHITE else 0.0,
                            'black_sharpness': sharpness if board.turn == chess.BLACK else 0.0
                        }
                        
                        sharpness_scores.append(result)
                    else:
                        # Default values for missing evaluations
                        sharpness_scores.append({'sharpness': 0.0, 'white_sharpness': 0.0, 'black_sharpness': 0.0})
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

    def calculate_player_accuracies(self, move_accuracies, positions, features=None):
        """
        Calculate overall accuracy for white and black players using Lichess approach.
        Uses the implementation from GameAnalyzer._calculate_player_accuracies
        for consistency and to avoid code duplication.
        
        Args:
            move_accuracies: List of dictionaries containing move accuracy information
            positions: List of chess board positions
            features: Optional FeatureVector with phase information
            
        Returns:
            Tuple of (white_accuracy, black_accuracy, phase_accuracies)
        """
        # Import GameAnalyzer dynamically to avoid circular imports
        from analysis.game_analyzer import GameAnalyzer
        
        # Extract phase boundaries from features if available
        phase_info = None
        if features is not None and hasattr(features, 'opening_length') and hasattr(features, 'middlegame_length'):
            # Convert normalized phase lengths to actual move numbers
            total_moves = int(features.total_moves)
            opening_end = max(1, int(features.opening_length * total_moves))
            middlegame_end = min(total_moves, int((features.opening_length + features.middlegame_length) * total_moves))
            
            phase_info = {
                'opening_end': opening_end,
                'middlegame_end': middlegame_end
            }
            
            logger.debug(f"Using actual phase detection: opening_end={opening_end}, middlegame_end={middlegame_end}")
        
        # Use the implementation from GameAnalyzer
        white_accuracy, black_accuracy, phase_accuracies = GameAnalyzer._calculate_player_accuracies(
            None,  # self is None since we're calling a class method
            move_accuracies, 
            positions,
            phase_info
        )
        
        return white_accuracy, black_accuracy, phase_accuracies

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

    def calculate_opening_novelty_score(self, game, features, eco_code=None):
        """
        Calculate the opening novelty score based on how early the game deviates from known theory.
        
        The score is calculated as: (number of moves matching ECO lines) / (total opening moves)
        This represents how long the players stayed in "book" during the opening phase.
        
        Args:
            game: The chess.pgn.Game object
            features: The FeatureVector with opening_length already calculated
            eco_code: The ECO code from the CSV data, if available
            
        Returns:
            float: The opening novelty score (0.0 to 1.0), eco_code, and opening_name
        """
        try:
            # Calculate total moves in opening phase
            total_moves = features.total_moves
            opening_length = features.opening_length
            total_opening_moves = int(opening_length * total_moves)
            
            if total_opening_moves <= 0:
                return 0.0, "", "", 0
                
            # Convert game moves to UCI format for comparison
            game_moves_uci = []
            board = chess.Board()
            current_node = game
            
            while not current_node.is_end():
                next_node = current_node.variations[0]
                move = next_node.move
                game_moves_uci.append(move.uci())
                board.push(move)
                current_node = next_node
            
            # If we have an ECO code from the CSV, use it instead of finding a match
            if eco_code:
                # Use the FeatureExtractor's calculate_opening_novelty_score method
                opening_novelty_score, opening_name, matched_eco, matching_plies = self.feature_extractor.calculate_opening_novelty_score(
                    eco_code, game_moves_uci, opening_length, total_moves
                )
                
                logger.debug(f"Using ECO code from CSV: {eco_code}")
                logger.debug(f"Matched ECO code: {matched_eco}")
                logger.debug(f"Opening name: {opening_name}")
                logger.debug(f"Opening novelty score: {opening_novelty_score:.2f}")
                logger.debug(f"Matching plies: {matching_plies}")
                
                return opening_novelty_score, matched_eco, opening_name, matching_plies
            
            return 0.0, "Unknown", "Unknown Opening", 0
            
        except Exception as e:
            logger.error(f"Error calculating opening novelty score: {e}")
            return 0.0, "", "Unknown Opening", 0

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
            cumulative_sharpness = self.calculate_cumulative_sharpness(sharpness_scores)
            
            # Calculate move accuracies
            move_accuracies = self.calculate_move_accuracies(positions, info_objects, mainline_moves)
            
            # Calculate overall player accuracies and phase-specific accuracies
            white_accuracy, black_accuracy, phase_accuracies = self.calculate_player_accuracies(
                move_accuracies, positions, features
            )
            
            # Calculate move statistics
            check_freq_white, check_freq_black, white_castle_move, black_castle_move = self.feature_extractor._calculate_move_statistics(positions, mainline_moves)
            
            # Calculate opening novelty score and get opening information
            opening_novelty_score, eco_code, opening_name, matching_plies = self.calculate_opening_novelty_score(game, features, game_row.get('eco'))
            features.opening_novelty_score = opening_novelty_score
            features.opening_name = opening_name
            features.eco = eco_code  # Store the matched ECO code
            # features.matching_plies = matching_plies  # Store the matching plies for accurate deviation calculation
            
            # Update accuracy values in features
            features.white_accuracy = white_accuracy
            features.black_accuracy = black_accuracy
            features.white_opening_accuracy = phase_accuracies['white']['opening']
            features.white_middlegame_accuracy = phase_accuracies['white']['middlegame']
            features.white_endgame_accuracy = phase_accuracies['white']['endgame']
            features.black_opening_accuracy = phase_accuracies['black']['opening']
            features.black_middlegame_accuracy = phase_accuracies['black']['middlegame']
            features.black_endgame_accuracy = phase_accuracies['black']['endgame']
            
            # Add features to game data
            result = game_row.copy()
            for key, value in features.__dict__.items():
                result[key] = value
            
            # Add summary sharpness metrics
            result['white_sharpness'] = cumulative_sharpness['white_sharpness']
            result['black_sharpness'] = cumulative_sharpness['black_sharpness']
            
            # Add move statistics
            result['white_check_frequency'] = check_freq_white
            result['black_check_frequency'] = check_freq_black
            result['white_castle_move'] = white_castle_move
            result['black_castle_move'] = black_castle_move
            
            # Add ECO code if not already in the data
            if not result.get('eco') and eco_code:
                result['eco'] = eco_code
                
            # Add opening name if not already in the data
            if not result.get('opening') and opening_name:
                result['opening'] = opening_name
            
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
    parser.add_argument('--wdl', type=str, choices=['lc0', 'sf'], default='sf',
                        help='Engine to use for WDL calculations: lc0 (Leela Chess Zero) or sf (Stockfish)')
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Starting feature extraction at {start_time}")
    
    # Configure logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - verbose logging activated")
    
    # Initialize the feature extractor with the selected WDL engine
    extractor = GameFeatureExtractor(wdl_engine=args.wdl)
    
    # Read input CSV
    input_csv = args.input
    output_csv = args.output
    logger.info(f"Reading input CSV from {input_csv}")
    try:
        # For debug mode, read the entire CSV and process a specific row
        if args.debug:
            logger.info("Debug mode: Reading entire CSV file")
            df = pd.read_csv(input_csv)
            row_to_process = 11 # Default row number for debug mode
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
                        'time_control', 'import_date', 'source', 'eval_info', 
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
                    result_df.to_csv(output_csv, mode='w', header=True, index=False)
                    logger.info(f"Saved debug game to CSV at {output_csv}")
            else:
                logger.error(f"Row {row_to_process} does not exist in the CSV. CSV has {len(df)} rows.")
                return
                
            # Set total for debug mode
            total_games = 1
        else:
            # Read in chunks for memory efficiency
            chunk_size = 1 if args.test else 10000
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

def print_accuracy_summary(features):
    """Print a summary of player accuracies by phase"""
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
    print(f"{Fore.BLUE}{Style.BRIGHT}PLAYER ACCURACY BY PHASE{Style.RESET_ALL}")
    print("="*80)
    
    # Create a nice table-like output
    print(f"{'Player':<8} | {'Overall':<10} | {'Opening':<10} | {'Middlegame':<10} | {'Endgame':<10}")
    print("-"*60)
    
    # Get accuracy values
    white_overall = features.get('white_accuracy', 0.0)
    white_opening = features.get('white_opening_accuracy', 0.0)
    white_middlegame = features.get('white_middlegame_accuracy', 0.0)
    white_endgame = features.get('white_endgame_accuracy', 0.0)
    
    black_overall = features.get('black_accuracy', 0.0)
    black_opening = features.get('black_opening_accuracy', 0.0)
    black_middlegame = features.get('black_middlegame_accuracy', 0.0)
    black_endgame = features.get('black_endgame_accuracy', 0.0)
    
    # Helper function to color-code accuracy values
    def color_accuracy(acc):
        if acc >= 90:
            return f"{Fore.GREEN}{acc:.1f}%{Style.RESET_ALL}"
        elif acc >= 80:
            return f"{Fore.CYAN}{acc:.1f}%{Style.RESET_ALL}"
        elif acc >= 70:
            return f"{Fore.BLUE}{acc:.1f}%{Style.RESET_ALL}"
        elif acc >= 60:
            return f"{Fore.YELLOW}{acc:.1f}%{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{acc:.1f}%{Style.RESET_ALL}"
    
    # Print White's accuracies with better alignment
    print(f"{Fore.WHITE}{Style.BRIGHT}White{Style.RESET_ALL}    | {color_accuracy(white_overall)}      | {color_accuracy(white_opening)}      | {color_accuracy(white_middlegame)}      | {color_accuracy(white_endgame)}")
    
    # Print Black's accuracies with better alignment
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Black{Style.RESET_ALL}    | {color_accuracy(black_overall)}      | {color_accuracy(black_opening)}      | {color_accuracy(black_middlegame)}      | {color_accuracy(black_endgame)}")

def print_opening_summary(features, game_data):
    """Print a summary of the opening phase including novelty information"""
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
    print(f"{Fore.BLUE}{Style.BRIGHT}OPENING ANALYSIS{Style.RESET_ALL}")
    print("="*80)
    
    # Get ECO code and opening name from features or game data
    eco_code = features.get('eco', game_data.get('eco', 'Unknown'))
    opening_name = features.get('opening_name', game_data.get('opening', 'Unknown Opening'))
    
    # Calculate opening information
    total_moves = features.get('total_moves', 0)
    opening_length = features.get('opening_length', 0)
    total_opening_moves = int(opening_length * total_moves)
    
    # Get opening novelty score
    novelty_score = features.get('opening_novelty_score', 0.0)
    
    # Get the matching plies directly from features if available
    # matching_plies = features.get('matching_plies', 0)
    
    # Calculate how many moves matched theory before deviation
    # The novelty score is the ratio of matching moves to total opening moves
    # So multiply by total_opening_moves to get the actual number of matching moves
    matching_moves = int(novelty_score * total_opening_moves)
    
    # Display opening information
    print(f"{Fore.CYAN}{Style.BRIGHT}Opening Information:{Style.RESET_ALL}")
    print(f"  ECO Code: {Fore.YELLOW}{eco_code}{Style.RESET_ALL}")
    print(f"  Opening: {Fore.YELLOW}{opening_name}{Style.RESET_ALL}")
    print(f"  Opening Length: {total_opening_moves} moves ({opening_length:.0%} of game)")
    
    # Display theory deviation
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Opening Theory:{Style.RESET_ALL}")
    if matching_moves > 0:
        if matching_moves == total_opening_moves:
            print(f"  {Fore.GREEN}Players followed established theory for the entire opening phase: {matching_moves}/{total_opening_moves} moves{Style.RESET_ALL}")
        else:
            # Calculate the deviation point in terms of move number and player
            # Use the stored matching_plies if available, otherwise estimate from matching_moves
            # if matching_plies > 0:
            #     # Use the actual matching plies from the feature extractor
            #     actual_matching_plies = matching_plies
            # else:
            #     # Fallback calculation if matching_plies not provided
            #     actual_matching_plies = matching_moves * 2
            
            # Add 1 to get the deviation ply (1-indexed)
            deviation_ply = matching_moves + 1
            
            # Calculate whether it's White's or Black's move at the deviation point
            is_white_deviation = (deviation_ply % 2 == 1)
            
            # Calculate the full move number (1. e4 e5 is move 1)
            deviation_move_number = (deviation_ply + 1) // 2 if is_white_deviation else deviation_ply // 2
            
            deviating_player = "White" if is_white_deviation else "Black"
            player_color = Fore.WHITE if is_white_deviation else Fore.MAGENTA
            
            print(f"  Players followed established theory for {Fore.YELLOW}{matching_moves}/{total_opening_moves}{Style.RESET_ALL} moves")
            print(f"  First deviation from theory occurred on move {Fore.YELLOW}{deviation_move_number}{Style.RESET_ALL}" + 
                  f" by {player_color}{Style.BRIGHT}{deviating_player}{Style.RESET_ALL}")
            print(f"  Opening novelty score: {novelty_score:.2f}")
    else:
        print(f"  {Fore.RED}No established opening theory was followed{Style.RESET_ALL}")
        print(f"  Either unknown opening or immediate deviation")
        print(f"  Opening novelty score: {novelty_score:.2f}")

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
    
    # Print opening summary (new)
    print_opening_summary(result, game_data)
    
    # Print sharpness summary if available
    if  'white_sharpness' in result and 'black_sharpness' in result:
        print_sharpness_summary({
            'white_sharpness': result['white_sharpness'],
            'black_sharpness': result['black_sharpness']
        })
    
    # Print accuracy summary
    print_accuracy_summary(result)
    
    # Print move statistics if available
    print_move_statistics(result)
    
    # Print player style analysis if available
    if 'white_initiative_ratio' in result and 'black_initiative_ratio' in result:
        print_player_type_analysis(result)
    
    # Print feature summary
    print_feature_summary(result)
    
    # Print judgment summary if available
    if 'white_brilliant_count' in result:
        print_judgment_summary(result)

def print_feature_summary(features):
    """Print a summary of the extracted features"""
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
    
    categories = {
        "Game Structure": [
            "total_moves", "opening_length", "middlegame_length", "endgame_length"
        ],
        "Opening Development": [
            "opening_novelty_score", "opening_name", "white_minor_piece_development", "black_minor_piece_development", 
            "white_queen_development", "black_queen_development"
        ],
        "Material Dynamics": [
            "white_material_changes", "black_material_changes", "material_balance_std"
        ],
        "Positional Control": [
            "white_piece_mobility_avg", "black_piece_mobility_avg",
            "white_center_control_avg", "black_center_control_avg",
            "white_space_advantage", "black_space_advantage",
        ],
        "King Safety": [
            "white_king_safety", "black_king_safety",
            "white_vulnerability_spikes", "black_vulnerability_spikes"
        ],
        "White Move Quality": [
            "white_brilliant_count", "white_great_count", "white_good_moves",
            "white_inaccuracy_count", "white_mistake_count", "white_blunder_count",
            "white_sacrifice_count", "white_avg_eval_change",
            "white_accuracy"
        ],
        "White Accuracy by Phase": [
            "white_opening_accuracy", "white_middlegame_accuracy", "white_endgame_accuracy"
        ],
        "Black Move Quality": [
            "black_brilliant_count", "black_great_count", "black_good_moves",
            "black_inaccuracy_count", "black_mistake_count", "black_blunder_count",
            "black_sacrifice_count", "black_avg_eval_change",
            "black_accuracy"
        ],
        "Black Accuracy by Phase": [
            "black_opening_accuracy", "black_middlegame_accuracy", "black_endgame_accuracy"
        ],
        "Move Statistics - White": [
             "white_check_frequency", "white_castle_move"
        ],
        "Move Statistics - Black": [
             "black_check_frequency", "black_castle_move"
        ],
        # New Player Type Features
        "Activist Style - White": [
            "white_initiative_ratio", "white_forcing_sequence_length", "white_counterplay_ratio"
        ],
        "Activist Style - Black": [
            "black_initiative_ratio", "black_forcing_sequence_length", "black_counterplay_ratio"
        ],
        "Theorist Style - White": [
            "opening_theory_adherence", "white_structural_consistency", "white_pattern_adherence"
        ],
        "Theorist Style - Black": [
            "opening_theory_adherence", "black_structural_consistency", "black_pattern_adherence"
        ],
        "Reflector Style - White": [
            "white_piece_harmony", "white_prophylactic_ratio", "white_exchange_sacrifice_ratio"
        ],
        "Reflector Style - Black": [
            "black_piece_harmony", "black_prophylactic_ratio", "black_exchange_sacrifice_ratio"
        ],
        "Pragmatist Style - White": [
            "white_concrete_calculation", "white_defensive_precision", 
            "white_objective_decision_ratio", "white_evaluation_clarity"
        ],
        "Pragmatist Style - Black": [
            "black_concrete_calculation", "black_defensive_precision", 
            "black_objective_decision_ratio", "black_evaluation_clarity"
        ]
    }
    
    # Print each category of features
    for category, feature_names in categories.items():
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{category}:{Style.RESET_ALL}")
        for name in feature_names:
            if name in features:
                value = features[name]
                # Format opening_novelty_score as a fraction (e.g., "2/11")
                if name == "opening_novelty_score" and 'total_moves' in features and 'opening_length' in features:
                    total_opening_moves = int(features['opening_length'] * features['total_moves'])
                    theoretical_moves = int(value * total_opening_moves)
                    if features.get('eco', ''):
                        display_value = f"{theoretical_moves}/{total_opening_moves} ({value:.2f}) - ECO: {features.get('eco', '')}"
                    else:
                        display_value = f"{theoretical_moves}/{total_opening_moves} ({value:.2f})"
                elif name == "opening_name":
                    display_value = f"{value}"
                elif isinstance(value, float):
                    if name.endswith('_accuracy'):
                        display_value = f"{value:.1f}%"
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                print(f"  {name}: {display_value}")
    print("")

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
    
    white = cumulative_sharpness.get('white_sharpness', 0)
    black = cumulative_sharpness.get('black_sharpness', 0)
    
    print(f"  White Sharpness: {sharpness_color(white)}{white:.1f}{Style.RESET_ALL}")
    print(f"  Black Sharpness: {sharpness_color(black)}{black:.1f}{Style.RESET_ALL}")

def print_move_statistics(result):
    """Print move statistics like captures, checks, and castling"""
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
    print(f"{Fore.BLUE}{Style.BRIGHT}MOVE STATISTICS{Style.RESET_ALL}")
    print("="*80)
    
    
    # Print check frequency for white and black
    if 'white_check_frequency' in result:
        check_freq_white = result['white_check_frequency']
        print(f"  White Check Frequency: {check_freq_white:.2f}")
    
    if 'black_check_frequency' in result:
        check_freq_black = result['black_check_frequency']
        print(f"  Black Check Frequency: {check_freq_black:.2f}")
    
    # Print prophylactic move frequency for white and black
    if 'white_prophylactic_frequency' in result:
        white_prophylactic = result['white_prophylactic_frequency']
        print(f"  White Prophylactic Frequency: {white_prophylactic:.2f}")
    
    if 'black_prophylactic_frequency' in result:
        black_prophylactic = result['black_prophylactic_frequency']
        print(f"  Black Prophylactic Frequency: {black_prophylactic:.2f}")
    
    # Print castling timing using actual move numbers
    if 'white_castle_move' in result:
        castle_white = result['white_castle_move']
        if isinstance(castle_white, float) and castle_white > 0:
            total_moves = result.get('total_moves', 0)
            if total_moves > 0:
                castle_move_number = int(castle_white * total_moves)
                print(f"  White Castling: Move {castle_move_number} ({castle_white:.2f} of game)")
            else:
                print(f"  White Castling: {castle_white:.2f} of game")
        elif castle_white > 0:
            print(f"  White Castling: Move {castle_white}")
        else:
            print(f"  White Castling: Did not castle")
    
    if 'black_castle_move' in result:
        castle_black = result['black_castle_move']
        if isinstance(castle_black, float) and castle_black > 0:
            total_moves = result.get('total_moves', 0)
            if total_moves > 0:
                castle_move_number = int(castle_black * total_moves)
                print(f"  Black Castling: Move {castle_move_number} ({castle_black:.2f} of game)")
            else:
                print(f"  Black Castling: {castle_black:.2f} of game")
        elif castle_black > 0:
            print(f"  Black Castling: Move {castle_black}")
        else:
            print(f"  Black Castling: Did not castle")

def print_player_type_analysis(features):
    """Print an analysis of player types based on the extracted features"""
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
    print(f"{Fore.BLUE}{Style.BRIGHT}PLAYER STYLE ANALYSIS{Style.RESET_ALL}")
    print("="*80)
    
    # Check if player style features are available
    if 'white_initiative_ratio' not in features or 'black_initiative_ratio' not in features:
        print("Player style features not available.")
        return
    
    # Calculate style scores for White
    white_activist_score = (
        features.get('white_initiative_ratio', 0) * 0.4 +
        features.get('white_forcing_sequence_length', 0) * 0.3 +
        features.get('white_counterplay_ratio', 0) * 0.3
    ) * 100
    
    white_theorist_score = (
        features.get('opening_theory_adherence', 0) * 0.4 +
        features.get('white_structural_consistency', 0) * 0.3 +
        features.get('white_pattern_adherence', 0) * 0.3
    ) * 100
    
    white_reflector_score = (
        features.get('white_piece_harmony', 0) * 0.4 +
        features.get('white_prophylactic_ratio', 0) * 0.3 +
        features.get('white_exchange_sacrifice_ratio', 0) * 0.3
    ) * 100
    
    white_pragmatist_score = (
        features.get('white_concrete_calculation', 0) * 0.3 +
        features.get('white_defensive_precision', 0) * 0.4 +
        features.get('white_objective_decision_ratio', 0) * 0.2 +
        features.get('white_evaluation_clarity', 0) * 0.1
    ) * 100
    
    # Calculate style scores for Black
    black_activist_score = (
        features.get('black_initiative_ratio', 0) * 0.4 +
        features.get('black_forcing_sequence_length', 0) * 0.3 +
        features.get('black_counterplay_ratio', 0) * 0.3
    ) * 100
    
    black_theorist_score = (
        features.get('opening_theory_adherence', 0) * 0.4 +
        features.get('black_structural_consistency', 0) * 0.3 +
        features.get('black_pattern_adherence', 0) * 0.3
    ) * 100
    
    black_reflector_score = (
        features.get('black_piece_harmony', 0) * 0.4 +
        features.get('black_prophylactic_ratio', 0) * 0.3 +
        features.get('black_exchange_sacrifice_ratio', 0) * 0.3
    ) * 100
    
    black_pragmatist_score = (
        features.get('black_concrete_calculation', 0) * 0.3 +
        features.get('black_defensive_precision', 0) * 0.4 +
        features.get('black_objective_decision_ratio', 0) * 0.2 +
        features.get('black_evaluation_clarity', 0) * 0.1
    ) * 100
    
    # Normalize scores
    white_total = max(1, white_activist_score + white_theorist_score + white_reflector_score + white_pragmatist_score)
    black_total = max(1, black_activist_score + black_theorist_score + black_reflector_score + black_pragmatist_score)
    
    white_activist_pct = (white_activist_score / white_total) * 100
    white_theorist_pct = (white_theorist_score / white_total) * 100
    white_reflector_pct = (white_reflector_score / white_total) * 100
    white_pragmatist_pct = (white_pragmatist_score / white_total) * 100
    
    black_activist_pct = (black_activist_score / black_total) * 100
    black_theorist_pct = (black_theorist_score / black_total) * 100
    black_reflector_pct = (black_reflector_score / black_total) * 100
    black_pragmatist_pct = (black_pragmatist_score / black_total) * 100
    
    # Determine primary and secondary styles
    white_styles = [
        ("Activist", white_activist_pct),
        ("Theorist", white_theorist_pct),
        ("Reflector", white_reflector_pct),
        ("Pragmatist", white_pragmatist_pct)
    ]
    white_styles.sort(key=lambda x: x[1], reverse=True)
    
    black_styles = [
        ("Activist", black_activist_pct),
        ("Theorist", black_theorist_pct),
        ("Reflector", black_reflector_pct),
        ("Pragmatist", black_pragmatist_pct)
    ]
    black_styles.sort(key=lambda x: x[1], reverse=True)
    
    # Print style analysis
    print(f"{Fore.WHITE}{Style.BRIGHT}White Player Style:{Style.RESET_ALL}")
    print(f"  Primary: {Fore.CYAN}{white_styles[0][0]}{Style.RESET_ALL} ({white_styles[0][1]:.1f}%)")
    print(f"  Secondary: {Fore.CYAN}{white_styles[1][0]}{Style.RESET_ALL} ({white_styles[1][1]:.1f}%)")
    print(f"  Style Breakdown:")
    print(f"    - Activist: {white_activist_pct:.1f}% (Dynamic, tactical, initiative-driven)")
    print(f"    - Theorist: {white_theorist_pct:.1f}% (Principled, structured, theoretical)")
    print(f"    - Reflector: {white_reflector_pct:.1f}% (Subtle, positional, harmonious)")
    print(f"    - Pragmatist: {white_pragmatist_pct:.1f}% (Calculating, practical, concrete)")
    
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}Black Player Style:{Style.RESET_ALL}")
    print(f"  Primary: {Fore.CYAN}{black_styles[0][0]}{Style.RESET_ALL} ({black_styles[0][1]:.1f}%)")
    print(f"  Secondary: {Fore.CYAN}{black_styles[1][0]}{Style.RESET_ALL} ({black_styles[1][1]:.1f}%)")
    print(f"  Style Breakdown:")
    print(f"    - Activist: {black_activist_pct:.1f}% (Dynamic, tactical, initiative-driven)")
    print(f"    - Theorist: {black_theorist_pct:.1f}% (Principled, structured, theoretical)")
    print(f"    - Reflector: {black_reflector_pct:.1f}% (Subtle, positional, harmonious)")
    print(f"    - Pragmatist: {black_pragmatist_pct:.1f}% (Calculating, practical, concrete)")
    
    # Print style descriptions
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Player Style Descriptions:{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Activist:{Style.RESET_ALL} Dynamic player who values initiative and tactical play.")
    print(f"    • Seeks complications and tactical opportunities")
    print(f"    • Willing to sacrifice material for initiative")
    print(f"    • Good at finding counterplay in difficult positions")
    print(f"    • Examples: Mikhail Tal, Garry Kasparov, Alexei Shirov")
    
    print(f"\n  {Fore.CYAN}Theorist:{Style.RESET_ALL} Principled player who values preparation and structure.")
    print(f"    • Strong opening preparation and theory knowledge")
    print(f"    • Maintains consistent pawn structures")
    print(f"    • Follows established principles and patterns")
    print(f"    • Examples: Wilhelm Steinitz, Vladimir Kramnik, Anatoly Karpov")
    
    print(f"\n  {Fore.CYAN}Reflector:{Style.RESET_ALL} Positional player who values harmony and prophylaxis.")
    print(f"    • Focuses on piece coordination and harmony")
    print(f"    • Makes preventive moves before threats materialize")
    print(f"    • Willing to make positional exchanges and sacrifices")
    print(f"    • Examples: Tigran Petrosian, Aron Nimzowitsch, Bent Larsen")
    
    print(f"\n  {Fore.CYAN}Pragmatist:{Style.RESET_ALL} Practical player who values concrete calculation and defense.")
    print(f"    • Strong defensive skills and practical decision-making")
    print(f"    • Makes objective, principle-based choices")
    print(f"    • Excellent at calculated evaluations and endgames")
    print(f"    • Examples: Emanuel Lasker, Mikhail Botvinnik, Magnus Carlsen")

if __name__ == "__main__":
    main()