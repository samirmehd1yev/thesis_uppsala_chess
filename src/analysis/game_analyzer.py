# src/analysis/game_analyzer.py
import chess
import chess.pgn
import io
from typing import Dict, List, Tuple, Optional, Any
import logging
import multiprocessing
from multiprocessing import Pool, Manager
import time
from tqdm import tqdm
import os
import math

from features.extractor import FeatureExtractor
from analysis.stockfish_handler import StockfishHandler
from analysis.move_analyzer import MoveAnalyzer
from models.data_classes import Info
from models.enums import Judgment
from analysis.sharpness_analyzer import WdlSharpnessAnalyzer

logger = logging.getLogger('chess_analyzer')

class GameAnalyzer:
    """
    A class for analyzing chess games with Stockfish and extracting features.
    """
    
    def __init__(self, stockfish_path: str = "stockfish", analysis_depth: int = 16, 
                 threads: int = None, hash_size: int = 128, num_cpus: int = None):
        """
        Initialize the GameAnalyzer.
        
        Args:
            stockfish_path: Path to stockfish engine executable
            analysis_depth: Depth for Stockfish analysis
            threads: Number of threads for each Stockfish instance
            hash_size: Hash table size in MB for Stockfish
            num_cpus: Number of CPU cores to use for parallel analysis (defaults to cores-1)
        """
        self.stockfish_path = stockfish_path
        self.analysis_depth = analysis_depth
        self.threads = threads or 1
        self.hash_size = hash_size
        
        # Set number of CPUs for parallel processing
        if num_cpus is None:
            self.num_cpus = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        else:
            self.num_cpus = max(1, min(num_cpus, multiprocessing.cpu_count()))
            
        self.feature_extractor = FeatureExtractor()
        
        # Add sharpness analyzer
        self.sharpness_analyzer = WdlSharpnessAnalyzer()
    
    def calculate_position_sharpness(self, positions: List[chess.Board], evals: List[Info]) -> List[Dict[str, float]]:
        """
        Calculate sharpness scores for all positions in the game.
        
        Args:
            positions: List of chess board positions
            evals: List of evaluation info objects
        
        Returns:
            List of dictionaries with sharpness scores for each position
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
        
    def analyze_pgn(self, pgn_content: str) -> Dict[str, Any]:
        """
        Analyze a game from PGN string.
        
        Args:
            pgn_content: PGN formatted string containing a chess game
            
        Returns:
            Dictionary with analysis results including:
            - game: The chess.pgn.Game object
            - evals: List of position evaluations
            - judgments: List of move judgments
            - features: FeatureVector with extracted features
            - top_moves: List of top moves for each position
            - sharpness: List of sharpness scores for each position
            - cumulative_sharpness: Cumulative sharpness scores for the game
            - move_accuracies: List of accuracy scores for each move
            - white_accuracy: Overall accuracy for white player
            - black_accuracy: Overall accuracy for black player
            - phase_accuracies: Accuracy metrics broken down by game phase
        """
        logger.info("Starting game analysis...")
        start_time = time.time()
        
        # Parse the game from PGN
        pgn_io = io.StringIO(pgn_content)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            logger.error("Failed to parse game!")
            return None
            
        logger.info("Game parsed successfully")
        
        # Get positions from the game
        positions = self.feature_extractor._get_positions(game)
        
        try:
            # PHASE 1: Evaluate positions in parallel
            logger.info("Evaluating positions with Stockfish...")
            evals = self._evaluate_positions_parallel(positions)
            logger.info(f"Position evaluation completed in {time.time() - start_time:.2f} seconds")
            
            # PHASE 2: Analyze moves in parallel
            logger.info("Analyzing moves...")
            phase2_start = time.time()
            mainline_moves = list(game.mainline_moves())
            judgments = self._analyze_moves_parallel(positions, mainline_moves, evals)
            logger.info(f"Move analysis completed in {time.time() - phase2_start:.2f} seconds")
            
            # Extract features
            logger.info("Extracting game features...")
            features = self.feature_extractor.extract_features(game, evals, judgments)
            
            # Collect top moves for each position
            top_moves = [eval_info.variation for eval_info in evals if eval_info and hasattr(eval_info, 'variation')]
            
            # Calculate position sharpness
            logger.info("Calculating position sharpness...")
            sharpness_scores = self.calculate_position_sharpness(positions, evals)
            cumulative_sharpness = self.sharpness_analyzer.calculate_cumulative_sharpness(sharpness_scores)
            print(f"Debug: Overall cumulative sharpness: {cumulative_sharpness['sharpness']:.2f}")
            print(f"Debug: White's cumulative sharpness: {cumulative_sharpness['white_sharpness']:.2f} (positions where White is to move)")
            print(f"Debug: Black's cumulative sharpness: {cumulative_sharpness['black_sharpness']:.2f} (positions where Black is to move)")
            
            # Calculate move accuracies
            logger.info("Calculating move accuracies...")
            move_accuracies = self._calculate_move_accuracies(positions, evals, mainline_moves)
            
            # Extract phase boundaries from features
            phase_info = None
            if hasattr(features, 'opening_length') and hasattr(features, 'middlegame_length'):
                # Convert normalized phase lengths to actual move numbers
                total_moves = int(features.total_moves)
                opening_end = max(1, int(features.opening_length * total_moves))
                middlegame_end = min(total_moves, int((features.opening_length + features.middlegame_length) * total_moves))
                
                phase_info = {
                    'opening_end': opening_end,
                    'middlegame_end': middlegame_end
                }
                
                logger.info(f"Extracted phase boundaries from features: opening_end={opening_end}, middlegame_end={middlegame_end}")
            
            # Calculate overall player accuracies and phase-specific accuracies
            white_accuracy, black_accuracy, phase_accuracies = self._calculate_player_accuracies(
                move_accuracies, positions, phase_info
            )
            print(f"Debug: White's accuracy: {white_accuracy:.1f}%")
            print(f"Debug: Black's accuracy: {black_accuracy:.1f}%")
            print(f"Debug: White's opening accuracy: {phase_accuracies['white']['opening']:.1f}%")
            print(f"Debug: White's middlegame accuracy: {phase_accuracies['white']['middlegame']:.1f}%")
            print(f"Debug: White's endgame accuracy: {phase_accuracies['white']['endgame']:.1f}%")
            print(f"Debug: Black's opening accuracy: {phase_accuracies['black']['opening']:.1f}%")
            print(f"Debug: Black's middlegame accuracy: {phase_accuracies['black']['middlegame']:.1f}%")
            print(f"Debug: Black's endgame accuracy: {phase_accuracies['black']['endgame']:.1f}%")
            
            # Set accuracy values in the feature vector
            features.white_accuracy = white_accuracy
            features.black_accuracy = black_accuracy
            features.white_opening_accuracy = phase_accuracies['white']['opening']
            features.white_middlegame_accuracy = phase_accuracies['white']['middlegame']
            features.white_endgame_accuracy = phase_accuracies['white']['endgame']
            features.black_opening_accuracy = phase_accuracies['black']['opening']
            features.black_middlegame_accuracy = phase_accuracies['black']['middlegame']
            features.black_endgame_accuracy = phase_accuracies['black']['endgame']
            
            logger.info(f"Total analysis completed in {time.time() - start_time:.2f} seconds")
            
            return {
                "game": game,
                "evals": evals,
                "judgments": judgments,
                "features": features,
                "top_moves": top_moves,
                "sharpness": sharpness_scores,
                "cumulative_sharpness": cumulative_sharpness,
                "move_accuracies": move_accuracies,
                "white_accuracy": white_accuracy,
                "black_accuracy": black_accuracy,
                "phase_accuracies": phase_accuracies
            }
        
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_positions_parallel(self, positions: List[chess.Board]) -> List[Info]:
        """
        Evaluate chess positions in parallel using multiple Stockfish instances.
        
        Args:
            positions: List of chess board positions to evaluate
            
        Returns:
            List of Info objects with evaluations
        """
        with Manager() as manager:
            # Create a shared dictionary to store results
            result_dict = manager.dict()
            
            # Prepare arguments for parallel evaluation
            eval_args = [(positions[i], i, self.stockfish_path, self.analysis_depth, 
                         self.threads, self.hash_size, result_dict) 
                        for i in range(len(positions))]
            
            # Evaluate positions in parallel
            with Pool(processes=self.num_cpus) as pool:
                results = list(tqdm(
                    pool.imap(self._evaluate_position_worker, eval_args), 
                    total=len(positions),
                    desc="Evaluating positions"
                ))
            
            # Convert shared dictionary to ordered list
            evals = [result_dict.get(i) for i in range(len(positions))]
            
            # Fill in any missing evaluations with neutral values
            for i in range(len(evals)):
                if evals[i] is None:
                    logger.warning(f"Missing evaluation at ply {i}, using default")
                    evals[i] = Info(ply=i, eval={"type": "cp", "value": 0}, variation=[])
            
            return evals
    
    @staticmethod
    def _evaluate_position_worker(args):
        """
        Worker function for position evaluation in parallel processing.
        
        Args:
            args: Tuple containing (position, ply, stockfish_path, depth, threads, hash_size, result_dict)
            
        Returns:
            True if successful, False otherwise
        """
        position, ply, stockfish_path, depth, threads, hash_size, result_dict = args
        
        # Create a new Stockfish instance for this process
        stockfish = StockfishHandler(
            path=stockfish_path, 
            depth=depth,
            threads=threads,
            hash_size=hash_size
        )
        
        try:
            # Evaluate the position
            result = stockfish.evaluate_position(position, ply)
            # print(f"Debug: Result: {result}")
            
            # Store result in shared dictionary
            result_dict[ply] = result
            
            # Close the stockfish engine
            stockfish.close()
            
            return True
        except Exception as e:
            logger.error(f"Error evaluating position at ply {ply}: {e}")
            try:
                stockfish.close()
            except:
                pass
            return False
    
    def _analyze_moves_parallel(self, positions: List[chess.Board], 
                                moves: List[chess.Move], 
                                evals: List[Info]) -> List[Judgment]:
        """
        Analyze moves in parallel using the MoveAnalyzer.
        
        Args:
            positions: List of board positions
            moves: List of moves played
            evals: List of position evaluations
            
        Returns:
            List of move judgments
        """
        # Prepare arguments for parallel move analysis
        analysis_args = []
        for i in range(1, len(evals)):
            move_idx = i - 1
            if move_idx < len(moves):
                prev_eval = evals[i-1]
                curr_eval = evals[i]
                move = moves[move_idx]
                prev_board = positions[move_idx]
                curr_board = positions[move_idx + 1]
                player_id = 'white' if (i-1) % 2 == 0 else 'black'
                
                analysis_args.append((prev_eval, curr_eval, prev_board, curr_board, move, player_id, move_idx))
        
        # Analyze moves in parallel
        with Pool(processes=self.num_cpus) as pool:
            judgments = list(tqdm(
                pool.imap(self._analyze_move_worker, analysis_args),
                total=len(analysis_args),
                desc="Analyzing moves"
            ))
        
        return judgments
    
    @staticmethod
    def _analyze_move_worker(args):
        """
        Worker function for move analysis in parallel processing.
        
        Args:
            args: Tuple containing (prev_eval, curr_eval, prev_board, curr_board, move, player_id, move_idx)
            
        Returns:
            Judgment for the move
        """
        prev_eval, curr_eval, prev_board, curr_board, move, player_id, move_idx = args
        
        try:
            # Get top moves if available
            top_moves = prev_eval.variation if prev_eval and hasattr(prev_eval, 'variation') else None
            
            # Analyze the move with detailed information
            judgment, debug_reason = MoveAnalyzer.analyze_move_with_top_moves(
                prev_eval, curr_eval, 
                prev_board=prev_board, 
                curr_board=curr_board, 
                move=move,
                top_moves=top_moves,
                debug=True
            )
            
            # Log the analysis result with debug information
            move_san = prev_board.san(move) if prev_board and move else move.uci()
            logger.debug(f"Move {move_san}: {judgment} | REASON: {debug_reason}")
            
            return judgment
        except Exception as e:
            logger.error(f"Error analyzing move: {e}")
            return Judgment.GOOD  # Default to GOOD on error
    
    def _calculate_move_accuracies(self, positions: List[chess.Board], evals: List[Info], mainline_moves: List[chess.Move]) -> List[Dict[str, float]]:
        """
        Calculate accuracy for each move based on the change in winning percentages.
        
        Args:
            positions: List of chess board positions
            evals: List of evaluation information for each position
            mainline_moves: List of moves played in the game
            
        Returns:
            List of dictionaries containing move accuracy information
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
    
    def _calculate_player_accuracies(self, move_accuracies: List[Dict[str, float]], positions: List[chess.Board], 
                               phase_info: Dict[str, int] = None) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
        """
        Calculate overall accuracy for white and black players using Lichess approach.
        Also calculates separate accuracy metrics for opening, middlegame, and endgame phases.
        
        Computes a mix of weighted mean (based on position volatility) and harmonic mean
        of individual move accuracies.
        
        Args:
            move_accuracies: List of dictionaries containing move accuracy information
            positions: List of chess board positions
            phase_info: Dictionary with phase boundary information (opening_end, middlegame_end)
            
        Returns:
            Tuple of (white_accuracy, black_accuracy, phase_accuracies)
            where phase_accuracies is a dictionary with phase-specific accuracy metrics
        """
        if not move_accuracies:
            empty_phases = {
                "white": {"opening": 0.0, "middlegame": 0.0, "endgame": 0.0},
                "black": {"opening": 0.0, "middlegame": 0.0, "endgame": 0.0}
            }
            return 0.0, 0.0, empty_phases
            
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
        
        # Determine game phases based on actual phase detection or fallback to move numbers
        total_moves = len(move_accuracies)
        
        if phase_info and 'opening_end' in phase_info and 'middlegame_end' in phase_info:
            # Use actual phase detection
            opening_end = phase_info['opening_end']
            middlegame_end = phase_info['middlegame_end']
            logger.debug(f"Using actual phase detection: opening_end={opening_end}, middlegame_end={middlegame_end}")
        else:
            # Fallback to simplified approach
            opening_end = min(15, total_moves // 3)  # First 15 moves or 1/3 of the game
            middlegame_end = total_moves - max(10, total_moves // 4)  # Last 10 moves or 1/4 of the game
            logger.debug(f"Using simplified phase detection: opening_end={opening_end}, middlegame_end={middlegame_end}")
        
        # Track phase-specific accuracies
        weighted_accuracies_by_phase = {
            "white": {"opening": [], "middlegame": [], "endgame": []},
            "black": {"opening": [], "middlegame": [], "endgame": []}
        }
        
        # Process move accuracies
        for i, acc in enumerate(move_accuracies):
            if i < len(weights):
                weight = weights[i]
                color = acc["player"]
                accuracy = acc["accuracy"]
                move_number = acc["move_number"]
                
                # Add to overall accuracy
                weighted_accuracies_by_color[color].append((accuracy, weight))
                
                # Determine phase and add to phase-specific accuracy
                if move_number <= opening_end:
                    phase = "opening"
                elif move_number > middlegame_end:
                    phase = "endgame"
                else:
                    phase = "middlegame"
                
                # Add accuracy to the appropriate phase
                weighted_accuracies_by_phase[color][phase].append((accuracy, weight))
                
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
        
        def calculate_combined_accuracy(weighted_values):
            """Calculate combined accuracy from weighted values (weighted + harmonic mean)"""
            if not weighted_values:
                return 0.0
            weighted_mean_value = weighted_mean(weighted_values)
            harmonic_mean_value = harmonic_mean([acc for acc, _ in weighted_values])
            return (weighted_mean_value + harmonic_mean_value) / 2
            
        # Calculate overall accuracies
        white_accuracy = calculate_combined_accuracy(weighted_accuracies_by_color["white"])
        black_accuracy = calculate_combined_accuracy(weighted_accuracies_by_color["black"])
        
        # Calculate phase-specific accuracies
        phase_accuracies = {
            "white": {
                "opening": calculate_combined_accuracy(weighted_accuracies_by_phase["white"]["opening"]),
                "middlegame": calculate_combined_accuracy(weighted_accuracies_by_phase["white"]["middlegame"]),
                "endgame": calculate_combined_accuracy(weighted_accuracies_by_phase["white"]["endgame"])
            },
            "black": {
                "opening": calculate_combined_accuracy(weighted_accuracies_by_phase["black"]["opening"]),
                "middlegame": calculate_combined_accuracy(weighted_accuracies_by_phase["black"]["middlegame"]),
                "endgame": calculate_combined_accuracy(weighted_accuracies_by_phase["black"]["endgame"])
            }
        }
        
        return white_accuracy, black_accuracy, phase_accuracies
    
    def format_analysis_report(self, analysis_result: Dict[str, Any], html: bool = False) -> str:
        """
        Format the analysis results as a text or HTML report.
        
        Args:
            analysis_result: Dict with analysis data
            html: Whether to format as HTML
            
        Returns:
            String with formatted analysis report
        """
        if not analysis_result:
            return "No analysis available."
            
        game = analysis_result.get("game")
        evals = analysis_result.get("evals", [])
        judgments = analysis_result.get("judgments", [])
        features = analysis_result.get("features")
        
        # Generate a simple text report
        report = []
        report.append("Game Analysis Report")
        report.append("===================")
        
        # Add game headers
        if game:
            headers = dict(game.headers)
            report.append(f"Event: {headers.get('Event', 'Unknown')}")
            report.append(f"White: {headers.get('White', 'Unknown')}")
            report.append(f"Black: {headers.get('Black', 'Unknown')}")
            report.append(f"Result: {headers.get('Result', 'Unknown')}")
            report.append("")
        
        # Add feature summary
        if features:
            report.append("Game Feature Summary")
            report.append("-------------------")
            
            # Add player accuracies
            report.append("Player Accuracies:")
            white_accuracy = analysis_result.get("white_accuracy", 0.0)
            black_accuracy = analysis_result.get("black_accuracy", 0.0)
            phase_accuracies = analysis_result.get("phase_accuracies", None)
            
            report.append(f"  White Overall Accuracy: {white_accuracy:.1f}%")
            report.append(f"  Black Overall Accuracy: {black_accuracy:.1f}%")
            
            # Add phase-specific accuracies if available
            if phase_accuracies:
                # White phase-specific accuracies
                report.append("")
                report.append("  White Accuracy by Phase:")
                report.append(f"    Opening: {phase_accuracies['white']['opening']:.1f}%")
                report.append(f"    Middlegame: {phase_accuracies['white']['middlegame']:.1f}%")
                report.append(f"    Endgame: {phase_accuracies['white']['endgame']:.1f}%")
                
                # Black phase-specific accuracies
                report.append("")
                report.append("  Black Accuracy by Phase:")
                report.append(f"    Opening: {phase_accuracies['black']['opening']:.1f}%")
                report.append(f"    Middlegame: {phase_accuracies['black']['middlegame']:.1f}%")
                report.append(f"    Endgame: {phase_accuracies['black']['endgame']:.1f}%")
            
            report.append("")
            
            # Game phase
            report.append("Game Phase:")
            report.append(f"  Total Moves: {features.total_moves:.0f}")
            report.append(f"  Opening Length: {features.opening_length:.0f}")
            report.append(f"  Middlegame Length: {features.middlegame_length:.0f}")
            report.append(f"  Endgame Length: {features.endgame_length:.0f}")
            report.append("")
            
            # Material/Position
            report.append("Material & Position:")
            report.append(f"  Material Balance Changes: {features.material_balance_changes:.2f}")
            report.append(f"  Piece Mobility Average: {features.piece_mobility_avg:.2f}")
            report.append(f"  Pawn Structure Changes: {features.pawn_structure_changes:.2f}")
            report.append(f"  Center Control Average: {features.center_control_avg:.2f}")
            report.append("")
            
            # White Move Quality
            report.append("White Move Quality:")
            report.append(f"  Brilliant Moves: {features.white_brilliant_count:.0f}")
            report.append(f"  Great Moves: {features.white_great_count:.0f}")
            report.append(f"  Good Moves: {features.white_good_moves:.0f}")
            report.append(f"  Inaccuracies: {features.white_inaccuracy_count:.0f}")
            report.append(f"  Mistakes: {features.white_mistake_count:.0f}")
            report.append(f"  Blunders: {features.white_blunder_count:.0f}")
            report.append(f"  Sacrifices: {features.white_sacrifice_count:.0f}")
            report.append("")
            
            # Black Move Quality
            report.append("Black Move Quality:")
            report.append(f"  Brilliant Moves: {features.black_brilliant_count:.0f}")
            report.append(f"  Great Moves: {features.black_great_count:.0f}")
            report.append(f"  Good Moves: {features.black_good_moves:.0f}")
            report.append(f"  Inaccuracies: {features.black_inaccuracy_count:.0f}")
            report.append(f"  Mistakes: {features.black_mistake_count:.0f}")
            report.append(f"  Blunders: {features.black_blunder_count:.0f}")
            report.append(f"  Sacrifices: {features.black_sacrifice_count:.0f}")
            
        # Add move-by-move analysis
        if evals and judgments:
            report.append("Move-by-Move Analysis")
            report.append("--------------------")
            move_accuracies = analysis_result.get("move_accuracies", [])
            accuracy_map = {(m["move_number"], m["player"]): m["accuracy"] for m in move_accuracies}
            
            board = chess.Board()
            node = game.game()
            
            move_num = 1
            current_player = "white"
            moves_analyzed = 0
            
            while not node.is_end() and moves_analyzed < len(judgments):
                next_node = node.variations[0]
                move = next_node.move
                san = board.san(move)
                
                # Add move number and player indicator
                if current_player == "white":
                    move_str = f"{move_num}. {san}"
                else:
                    move_str = f"{move_num}... {san}"
                
                # Get judgment for the move
                judgment = judgments[moves_analyzed] if moves_analyzed < len(judgments) else None
                
                # Get accuracy for the move
                accuracy = accuracy_map.get((move_num, current_player), 0.0)
                
                if judgment:
                    report.append(f"{move_str}: {judgment.name} (Accuracy: {accuracy:.1f}%)")
                else:
                    report.append(f"{move_str}")
                
                # Update for next move
                board.push(move)
                node = next_node
                moves_analyzed += 1
                
                # Update move number and player
                if current_player == "black":
                    move_num += 1
                current_player = "black" if current_player == "white" else "white"
            
            report.append("")
        
        # Add detailed move accuracy table
        move_accuracies = analysis_result.get("move_accuracies", [])
        if move_accuracies:
            report.append("Detailed Move Accuracies")
            report.append("------------------------")
            
            # Table header
            if html:
                report.append("<table border='1'>")
                report.append("<tr><th>Move</th><th>Player</th><th>Accuracy</th><th>Win% Before</th><th>Win% After</th></tr>")
            else:
                report.append(f"{'Move':<6} {'Player':<8} {'Accuracy':<10} {'Win% Before':<12} {'Win% After':<12}")
                report.append("-" * 60)
            
            # Table rows
            for move in move_accuracies:
                move_num = move["move_number"]
                player = move["player"].capitalize()
                accuracy = move["accuracy"]
                win_before = move["win_percent_before"]
                win_after = move["win_percent_after"]
                
                if html:
                    report.append(f"<tr><td>{move_num}{'' if player == 'White' else '...'}</td>"
                                 f"<td>{player}</td>"
                                 f"<td>{accuracy:.1f}%</td>"
                                 f"<td>{win_before:.1f}%</td>"
                                 f"<td>{win_after:.1f}%</td></tr>")
                else:
                    move_str = f"{move_num}{'' if player == 'White' else '...'}"
                    report.append(f"{move_str:<6} {player:<8} {accuracy:.1f}%{'':<5} {win_before:.1f}%{'':<5} {win_after:.1f}%")
            
            if html:
                report.append("</table>")
            
            report.append("")
        
        # Format as HTML if needed
        if html:
            html_report = "<html><head><style>"
            html_report += "body { font-family: Arial, sans-serif; }"
            html_report += "h1, h2 { color: #333; }"
            html_report += "table { border-collapse: collapse; width: 100%; }"
            html_report += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
            html_report += "th { background-color: #f2f2f2; }"
            html_report += "</style></head><body>"
            
            # Convert text report to HTML
            html_report += "<h1>Game Analysis Report</h1>"
            
            # Add headers
            if game:
                headers = dict(game.headers)
                html_report += f"<p><strong>Event:</strong> {headers.get('Event', 'Unknown')}</p>"
                html_report += f"<p><strong>White:</strong> {headers.get('White', 'Unknown')}</p>"
                html_report += f"<p><strong>Black:</strong> {headers.get('Black', 'Unknown')}</p>"
                html_report += f"<p><strong>Result:</strong> {headers.get('Result', 'Unknown')}</p>"
            
            # Add features as HTML tables
            if features:
                html_report += "<h2>Game Feature Summary</h2>"
                
                # Player accuracies table
                html_report += "<h3>Player Accuracies</h3>"
                html_report += "<table>"
                html_report += "<tr><th>Player</th><th>Overall</th><th>Opening</th><th>Middlegame</th><th>Endgame</th></tr>"
                
                # Use phase_accuracies if available
                if phase_accuracies:
                    html_report += f"<tr><td>White</td><td>{white_accuracy:.1f}%</td>"
                    html_report += f"<td>{phase_accuracies['white']['opening']:.1f}%</td>"
                    html_report += f"<td>{phase_accuracies['white']['middlegame']:.1f}%</td>"
                    html_report += f"<td>{phase_accuracies['white']['endgame']:.1f}%</td></tr>"
                    
                    html_report += f"<tr><td>Black</td><td>{black_accuracy:.1f}%</td>"
                    html_report += f"<td>{phase_accuracies['black']['opening']:.1f}%</td>"
                    html_report += f"<td>{phase_accuracies['black']['middlegame']:.1f}%</td>"
                    html_report += f"<td>{phase_accuracies['black']['endgame']:.1f}%</td></tr>"
                else:
                    html_report += f"<tr><td>White</td><td>{white_accuracy:.1f}%</td><td>-</td><td>-</td><td>-</td></tr>"
                    html_report += f"<tr><td>Black</td><td>{black_accuracy:.1f}%</td><td>-</td><td>-</td><td>-</td></tr>"
                
                html_report += "</table>"
                
                # Game phase table
                html_report += "<h3>Game Phase</h3>"
                html_report += "<table><tr><th>Metric</th><th>Value</th></tr>"
                html_report += f"<tr><td>Total Moves</td><td>{features.total_moves:.0f}</td></tr>"
                html_report += f"<tr><td>Opening Length</td><td>{features.opening_length:.0f}</td></tr>"
                html_report += f"<tr><td>Middlegame Length</td><td>{features.middlegame_length:.0f}</td></tr>"
                html_report += f"<tr><td>Endgame Length</td><td>{features.endgame_length:.0f}</td></tr>"
                html_report += "</table>"
                
                # Material table
                html_report += "<h3>Material & Position</h3>"
                html_report += "<table><tr><th>Metric</th><th>Value</th></tr>"
                html_report += f"<tr><td>Material Balance Changes</td><td>{features.material_balance_changes:.2f}</td></tr>"
                html_report += f"<tr><td>Piece Mobility Average</td><td>{features.piece_mobility_avg:.2f}</td></tr>"
                html_report += f"<tr><td>Pawn Structure Changes</td><td>{features.pawn_structure_changes:.2f}</td></tr>"
                html_report += f"<tr><td>Center Control Average</td><td>{features.center_control_avg:.2f}</td></tr>"
                html_report += "</table>"
                
                # Move quality comparison table
                html_report += "<h3>Move Quality Comparison</h3>"
                html_report += "<table><tr><th>Metric</th><th>White</th><th>Black</th></tr>"
                html_report += f"<tr><td>Brilliant Moves</td><td>{features.white_brilliant_count:.0f}</td><td>{features.black_brilliant_count:.0f}</td></tr>"
                html_report += f"<tr><td>Great Moves</td><td>{features.white_great_count:.0f}</td><td>{features.black_great_count:.0f}</td></tr>"
                html_report += f"<tr><td>Good Moves</td><td>{features.white_good_moves:.0f}</td><td>{features.black_good_moves:.0f}</td></tr>"
                html_report += f"<tr><td>Inaccuracies</td><td>{features.white_inaccuracy_count:.0f}</td><td>{features.black_inaccuracy_count:.0f}</td></tr>"
                html_report += f"<tr><td>Mistakes</td><td>{features.white_mistake_count:.0f}</td><td>{features.black_mistake_count:.0f}</td></tr>"
                html_report += f"<tr><td>Blunders</td><td>{features.white_blunder_count:.0f}</td><td>{features.black_blunder_count:.0f}</td></tr>"
                html_report += f"<tr><td>Sacrifices</td><td>{features.white_sacrifice_count:.0f}</td><td>{features.black_sacrifice_count:.0f}</td></tr>"
                html_report += "</table>"
            
            html_report += "</body></html>"
            return html_report
        else:
            return "\n".join(report)

# Helper function for simple analysis
def analyze_game(pgn_content: str, stockfish_path: str = "stockfish", 
                analysis_depth: int = 16, threads: int = 1) -> Dict[str, Any]:
    """
    Simple helper function to analyze a chess game.
    
    Args:
        pgn_content: PGN formatted string containing a chess game
        stockfish_path: Path to stockfish engine executable
        analysis_depth: Depth for Stockfish analysis
        threads: Number of threads for each Stockfish instance
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = GameAnalyzer(
        stockfish_path=stockfish_path,
        analysis_depth=analysis_depth,
        threads=threads
    )
    return analyzer.analyze_pgn(pgn_content) 