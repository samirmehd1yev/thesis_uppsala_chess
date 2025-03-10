"""
Chess Game Evaluation Script

This script processes chess games from a CSV file, extracts evaluations, 
and stores evaluation data and top moves with scores back into a new CSV.

Requirements:
- pandas
- python-chess
- stockfish (needs to be installed and in PATH)
"""

import os
import sys
import json
import chess
import chess.pgn
import chess.engine
import pandas as pd
import multiprocessing
import argparse
import logging
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
from pathlib import Path
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('chess_evaluator')

class ChessGameAnalyzer:
    """Class for analyzing chess games using Stockfish."""
    
    def __init__(self, stockfish_path: str = "stockfish", 
                 depth: int = 20, 
                 multipv: int = 3, 
                 threads: int = 1, 
                 hash_size: int = 128):
        """
        Initialize the analyzer with Stockfish settings.
        
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth
            multipv: Number of principal variations to calculate (top moves)
            threads: Number of threads for Stockfish
            hash_size: Hash size in MB for Stockfish
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.multipv = multipv
        self.threads = threads
        self.hash_size = hash_size
        self.engine = None
    
    def start_engine(self) -> bool:
        """Initialize and configure the Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            # Configure engine parameters
            self.engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_size
            })
            return True
        except Exception as e:
            logger.error(f"Engine initialization error: {e}")
            return False
    
    def stop_engine(self) -> None:
        """Properly close the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def analyze_game(self, pgn_text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze all positions in a chess game with progress tracking.
        
        Args:
            pgn_text: PGN notation of the chess game
            
        Returns:
            Tuple containing evaluations and top moves
        """
        # Initialize engine if not already running
        if not self.engine and not self.start_engine():
            return [], []
        
        evaluations = []
        top_moves = []
        
        try:
            # Parse PGN
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            
            if not game:
                logger.warning(f"Failed to parse game: {pgn_text[:100]}...")
                return [], []
                
            
            # Get total moves for progress tracking
            move_count = 0
            board_copy = game.board()
            for _ in game.mainline_moves():
                move_count += 1
            
            # Analyze initial position
            board = game.board()
            info = self._evaluate_position(board)
            if info:
                evaluations.append(info)
                top_moves.append(self._extract_top_moves(info))
            
            # Process each move without a progress bar
            # The progress tracking is now handled at the game level in process_chunk
            for move in game.mainline_moves():
                board.push(move)
                
                # Evaluate position
                info = self._evaluate_position(board)
                if info:
                    evaluations.append(info)
                    top_moves.append(self._extract_top_moves(info))
            
            return evaluations, top_moves
            
        except Exception as e:
            logger.error(f"Error analyzing game: {e}")
            return [], []
        
    def _evaluate_position(self, board: chess.Board) -> Dict[str, Any]:
        """
        Evaluate a single chess position.
        
        Args:
            board: Chess board position
            
        Returns:
            Dictionary with evaluation information
        """
        try:
            # Skip evaluation for terminal positions
            if board.is_game_over():
                return self._evaluate_terminal_position(board)
            
            # Analyze with multipv
            result = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth),
                multipv=self.multipv
            )
            
            # Extract the evaluation information
            eval_info = {
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "multipv": []
            }
            
            # Process each PV line
            for pv_info in result:
                pv_data = {}
                
                # Extract score - always from White's perspective
                score = pv_info["score"]
                if score.is_mate():
                    mate_score = score.white().mate()
                    pv_data["score"] = {"type": "mate", "value": mate_score}
                else:
                    cp_score = score.white().score()
                    pv_data["score"] = {"type": "cp", "value": cp_score}
                
                # Extract principal variation
                if "pv" in pv_info:
                    moves = []
                    temp_board = board.copy()
                    for pv_move in pv_info["pv"]:
                        move_uci = pv_move.uci()
                        move_san = temp_board.san(pv_move)
                        moves.append({
                            "uci": move_uci,
                            "san": move_san
                        })
                        temp_board.push(pv_move)
                    pv_data["pv"] = moves
                    
                # Add data to multipv list
                eval_info["multipv"].append(pv_data)
            
            # Set main eval to the first PV line's evaluation
            if eval_info["multipv"]:
                eval_info["eval"] = eval_info["multipv"][0]["score"]
                
                # Try to extract Win/Draw/Loss probabilities if available
                try:
                    wdl = result[0]["score"].wdl()
                    if wdl:
                        eval_info["wdl"] = {
                            "wins": wdl.wins,
                            "draws": wdl.draws,
                            "losses": wdl.losses
                        }
                except Exception:
                    # WDL might not be available, skip it
                    pass
            
            return eval_info
            
        except Exception as e:
            logger.error(f"Error evaluating position: {e}")
            return {}
    
    def _evaluate_terminal_position(self, board: chess.Board) -> Dict[str, Any]:
        """Evaluate a game-over position."""
        eval_info = {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "multipv": [],
            "terminal": True
        }
        
        # Determine the outcome
        outcome = board.outcome()
        if outcome:
            if outcome.winner is None:
                # Draw
                eval_info["eval"] = {"type": "cp", "value": 0}
                eval_info["wdl"] = {"wins": 0, "draws": 1000, "losses": 0}
            else:
                # Checkmate - always from White's perspective
                if outcome.winner == chess.WHITE:
                    # White wins (positive mate score for White)
                    eval_info["eval"] = {"type": "mate", "value": 1}  # Mate in 1 for White
                    eval_info["wdl"] = {"wins": 1000, "draws": 0, "losses": 0}
                else:
                    # Black wins (negative mate score for White)
                    eval_info["eval"] = {"type": "mate", "value": -1}  # Mate in 1 against White
                    eval_info["wdl"] = {"wins": 0, "draws": 0, "losses": 1000}
        
        return eval_info
    
    def _extract_top_moves(self, eval_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract top moves from evaluation info.
        
        Args:
            eval_info: Evaluation information dictionary
            
        Returns:
            List of top moves with their evaluations
        """
        top_moves = []
        
        if not eval_info or "multipv" not in eval_info:
            return top_moves
            
        for pv_data in eval_info.get("multipv", []):
            if "pv" in pv_data and pv_data["pv"] and len(pv_data["pv"]) > 0:
                move_data = {
                    "move": pv_data["pv"][0],
                    "score": pv_data.get("score", {})
                }
                top_moves.append(move_data)
                
        return top_moves

def format_evaluation_data(evaluations: List[Dict], top_moves: List[Dict]) -> Tuple[str, str]:
    """
    Format evaluation and top moves data for storage.
    
    Args:
        evaluations: List of evaluation dictionaries
        top_moves: List of top move dictionaries
        
    Returns:
        Tuple of (evaluations_json, top_moves_json)
    """
    try:
        evals_json = json.dumps(evaluations, separators=(',', ':'))
        moves_json = json.dumps(top_moves, separators=(',', ':'))
        return evals_json, moves_json
    except Exception as e:
        logger.error(f"Error formatting evaluation data: {e}")
        return "", ""

def process_game(args):
    """
    Process a single game for parallel execution.
    
    Args:
        args: Tuple containing game information and analyzer settings
        
    Returns:
        Dictionary with the processed game results
    """
    idx, game_data, stockfish_path, depth, multipv, threads, hash_size = args
    
    # Initialize analyzer
    analyzer = ChessGameAnalyzer(
        stockfish_path=stockfish_path,
        depth=depth,
        multipv=multipv,
        threads=threads,
        hash_size=hash_size
    )
    
    # Extract pgn moves
    pgn_text = game_data.get("moves", "")
    
    # Analyze game
    evaluations, top_moves = analyzer.analyze_game(pgn_text)
    analyzer.stop_engine()
    
    # Format for storage
    evals_json, moves_json = format_evaluation_data(evaluations, top_moves)
    
    # Return results
    return {
        "idx": idx,
        "evaluations": evals_json,
        "top_moves": moves_json,
        "has_analysis": bool(evaluations)
    }

def process_chunk(chunk_df, analyzer_settings, progress_callback=None):
    """
    Process a chunk of games in parallel with tqdm progress bar.
    
    Args:
        chunk_df: DataFrame chunk to process
        analyzer_settings: Dictionary with analyzer settings
        progress_callback: Optional callback for progress updates
        
    Returns:
        DataFrame with added evaluation data
    """
    # Prepare arguments for parallel processing
    process_args = []
    for idx, row in chunk_df.iterrows():
        if pd.isna(row['moves']) or row['moves'] == '':
            continue
            
        game_data = row.to_dict()
        process_args.append((
            idx,
            game_data,
            analyzer_settings["stockfish_path"],
            analyzer_settings["depth"],
            analyzer_settings["multipv"],
            analyzer_settings["threads"],
            analyzer_settings["hash_size"]
        ))
    
    results = {}
    
    # Process games in parallel with tqdm progress bar
    from tqdm import tqdm
    
    # Create a progress bar for tracking completed games
    total_games = len(process_args)
    pbar = tqdm(total=total_games, 
               desc=f"Analyzing games", 
               unit="game", 
               ncols=100,
               smoothing=0.1)
    
    # Process games in parallel
    with ProcessPoolExecutor(max_workers=analyzer_settings["workers"]) as executor:
        futures = {executor.submit(process_game, args): args[0] for args in process_args}
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                result = future.result(timeout=900)
                if result is None:
                    logger.warning("Game analysis timed out")
                    pbar.update(1)
                    continue
                idx = result["idx"]
                results[idx] = {
                    'evaluations': result["evaluations"],
                    'top_moves': result["top_moves"],
                    'has_analysis': result["has_analysis"]
                }
                # Update progress bar for each completed game
                pbar.update(1)
                pbar.set_postfix({"completed": f"{len(results)}/{total_games}"})
            except Exception as e:
                logger.error(f"Error processing game: {e}")
                # Still update the progress bar even if there was an error
                pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    
    # Update DataFrame with results
    chunk_df['evaluations'] = pd.Series({idx: results[idx]['evaluations'] if idx in results else None for idx in chunk_df.index})
    chunk_df['top_moves'] = pd.Series({idx: results[idx]['top_moves'] if idx in results else None for idx in chunk_df.index})
    chunk_df['has_analysis'] = pd.Series({idx: results[idx]['has_analysis'] if idx in results else False for idx in chunk_df.index})
    
    return chunk_df


def main():
    """Main function for the chess game evaluation script with improved progress monitoring."""
    parser = argparse.ArgumentParser(description='Chess Game Evaluation Script')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--stockfish', type=str, default='stockfish', help='Path to Stockfish executable')
    parser.add_argument('--depth', type=int, default=20, help='Analysis depth')
    parser.add_argument('--multipv', type=int, default=3, help='Number of principal variations (top moves)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--threads', type=int, default=1, help='Threads per Stockfish instance')
    parser.add_argument('--hash', type=int, default=128, help='Hash size in MB per Stockfish instance')
    parser.add_argument('--chunk-size', type=int, default=100, help='Chunk size for processing')
    parser.add_argument('--resume', action='store_true', help='Resume from partially processed output')
    parser.add_argument('--max-games', type=int, default=None, help='Maximum number of games to process')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars (for log files)')
    
    args = parser.parse_args()
    
    # Determine number of workers
    n_workers = args.workers
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU by default
    
    # Collect settings
    analyzer_settings = {
        "stockfish_path": args.stockfish,
        "depth": args.depth,
        "multipv": args.multipv,
        "workers": n_workers,
        "threads": args.threads,
        "hash_size": args.hash
    }
    
    logger.info(f"Starting chess evaluation pipeline")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Stockfish path: {args.stockfish}")
    logger.info(f"Analysis depth: {args.depth}")
    logger.info(f"MultiPV: {args.multipv}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Threads per worker: {args.threads}")
    logger.info(f"Hash size: {args.hash} MB")
    logger.info(f"Chunk size: {args.chunk_size}")
    
    # Check if Stockfish is available
    try:
        test_analyzer = ChessGameAnalyzer(stockfish_path=args.stockfish)
        if not test_analyzer.start_engine():
            logger.error(f"Cannot initialize Stockfish engine. Please make sure Stockfish is installed and the path is correct.")
            return 1
        test_analyzer.stop_engine()
    except Exception as e:
        logger.error(f"Stockfish test failed: {e}")
        logger.error("Please make sure Stockfish is installed and the path is correct.")
        return 1
    
    # Load the CSV file
    start_time = time.time()
    logger.info("Loading CSV file...")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} games from CSV")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return 1
    
    # Limit number of games if specified
    if args.max_games and len(df) > args.max_games:
        df = df.iloc[:args.max_games]
        logger.info(f"Limited to {args.max_games} games")
    
    # Check if we need to resume processing
    processed_games = set()
    if args.resume and os.path.exists(args.output):
        logger.info(f"Resuming from existing output: {args.output}")
        try:
            # Use error_bad_lines=False (for pandas <1.3) or on_bad_lines='skip' (pandas >=1.3)
            try:
                # For newer pandas versions
                df_output = pd.read_csv(args.output, on_bad_lines='skip')
                logger.warning("Using 'on_bad_lines=skip' to handle corrupted rows")
            except Exception as e:
                logger.error(f"Error reading output file for resuming: {e}")
                sys.exit(1)
            
            # Count the difference between expected rows and actual rows
            file_size = sum(1 for _ in open(args.output)) - 1  # -1 for header
            skipped_rows = file_size - len(df_output)
            if skipped_rows > 0:
                logger.warning(f"Skipped {skipped_rows} corrupted rows when reading the output file")
            
            # Identify which games have already been processed
            for i, row in df_output.iterrows():
                if pd.notna(row.get('evaluations')):
                    # Create a unique key for the game
                    game_key = (
                        row.get('white', ''), 
                        row.get('black', ''), 
                        row.get('date', ''), 
                        row.get('event', ''),
                        row.get('moves', '')
                    )
                    processed_games.add(game_key)
            
            logger.info(f"Found {len(processed_games)} already processed games")
        except Exception as e:
            logger.error(f"Error reading output file for resuming: {e}")
            logger.warning("Will start processing from the beginning")
            processed_games = set()  # Reset to empty set if we can't read the file at all
    
    # Filter out already processed games if resuming
    if processed_games:
        df_to_process = []
        for i, row in df.iterrows():
            game_key = (
                row.get('white', ''), 
                row.get('black', ''), 
                row.get('date', ''), 
                row.get('event', ''),
                row.get('moves', '')
            )
            if game_key not in processed_games:
                df_to_process.append(row)
        
        if df_to_process:
            df_to_process = pd.DataFrame(df_to_process)
            logger.info(f"{len(df_to_process)} games remaining to process")
        else:
            logger.info("All games have already been processed")
            return 0
    else:
        df_to_process = df
    
    # Process in chunks
    total_games = len(df_to_process)
    chunk_size = min(args.chunk_size, total_games)
    num_chunks = (total_games + chunk_size - 1) // chunk_size  # Ceiling division
    
    logger.info(f"Processing {total_games} games in {num_chunks} chunks")
    
    # Process each chunk with overall progress bar
    start_time = time.time()
    output_exists = os.path.exists(args.output) and args.resume
    
    from tqdm import tqdm
    
    # Create progress bar for total games across all chunks
    with tqdm(total=total_games, desc="Total progress", 
             unit="game", disable=args.no_progress) as total_pbar:
        
        games_completed = 0
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_games)
            chunk_size_actual = end_idx - start_idx
            
            logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks} (games {start_idx+1}-{end_idx})")
            
            # Get the chunk
            chunk_df = df_to_process.iloc[start_idx:end_idx].copy()
            
            # Process the chunk
            try:
                result_df = process_chunk(chunk_df, analyzer_settings)
                
                # Save the results
                if output_exists or chunk_idx > 0:
                    # Append to existing file
                    result_df.to_csv(args.output, mode='a', header=False, index=False)
                else:
                    # Create new file
                    result_df.to_csv(args.output, index=False)
                    output_exists = True
                    
                # Update total progress bar
                games_completed += chunk_size_actual
                total_pbar.update(chunk_size_actual)
                total_pbar.set_postfix({"completed": f"{games_completed}/{total_games}"})
                
                # Log progress
                elapsed_time = time.time() - start_time
                games_per_second = games_completed / elapsed_time if elapsed_time > 0 else 0
                
                logger.info(f"Completed {chunk_size_actual} games ({games_completed}/{total_games} total, "
                           f"{games_completed/total_games*100:.1f}%)")
                logger.info(f"Processing speed: {games_per_second:.2f} games/second")
                
                games_remaining = total_games - games_completed
                if games_remaining > 0:
                    eta_seconds = games_remaining / games_per_second if games_per_second > 0 else 0
                    eta_hours = eta_seconds / 3600
                    logger.info(f"Estimated time remaining: {eta_hours:.1f} hours")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx+1}: {e}")
    
    # Log final stats
    total_time = time.time() - start_time
    logger.info(f"\nProcessing completed in {total_time/3600:.2f} hours")
    logger.info(f"Average speed: {total_games/total_time:.2f} games/second")
    logger.info(f"Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())