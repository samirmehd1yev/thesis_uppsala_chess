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
            
            logger.info(f"Total analysis completed in {time.time() - start_time:.2f} seconds")
            
            return {
                "game": game,
                "evals": evals,
                "judgments": judgments,
                "features": features,
                "top_moves": top_moves,
                "sharpness": sharpness_scores,
                "cumulative_sharpness": cumulative_sharpness
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