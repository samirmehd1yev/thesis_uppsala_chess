#!/usr/bin/env python3
import os
import pandas as pd
import chess
import chess.pgn
import chess.engine
import io
from typing import List, Dict, Optional, Any, Tuple, Set
import logging
import multiprocessing
from multiprocessing import Pool, Lock, Manager
import time
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import numpy as np
import gc
import traceback
import signal
import sys
from datetime import datetime
import csv

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chess_analyzer')

# Global file lock for CSV operations
file_lock = multiprocessing.Lock()

class StockfishHandler:
    def __init__(self, path: str = "stockfish", depth: int = 16, threads: int = 1, hash_size: int = 128):
        """Initialize the StockfishHandler with chess.engine library."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            
            # Get available options and configure only supported ones
            config = {}
            
            # Check and set essential options
            if "Threads" in self.engine.options:
                config["Threads"] = threads
            if "Hash" in self.engine.options:
                config["Hash"] = hash_size
            if "Skill Level" in self.engine.options:
                config["Skill Level"] = 20  # Maximum skill level
                
            # Configure engine with supported parameters
            self.engine.configure(config)
            
            self.depth = depth
            self.path = path
            logger.debug(f"Engine initialized with options: {config}")
        except FileNotFoundError:
            logger.error(f"Stockfish engine not found at path: {path}")
            raise FileNotFoundError(f"Stockfish engine not found at path: {path}")
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish engine: {e}")
            raise

    def evaluate_position(self, board: chess.Board, num_variations: int = 3) -> Dict[str, Any]:
        """Evaluate position and return top moves with evaluations."""
        if not board:
            return {"eval": {"type": "cp", "value": 0}, "top_moves": []}
            
        try:
            # Get analysis with multiple variations
            analysis = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth),
                multipv=num_variations
            )
            
            # Extract evaluation from the best move
            if isinstance(analysis, list) and analysis:
                primary_eval = analysis[0]
                score = primary_eval.get("score")
                
                if score:
                    # Convert score to dictionary format
                    if score.is_mate():
                        eval_dict = {"type": "mate", "value": score.white().mate()}
                    else:
                        eval_dict = {"type": "cp", "value": score.white().score()}
                else:
                    eval_dict = {"type": "cp", "value": 0}
                    
                # Extract top moves
                top_moves = []
                for idx, info in enumerate(analysis):
                    if "pv" in info and info["pv"] and "score" in info:
                        score = info["score"]
                        if score.is_mate():
                            score_dict = {"type": "mate", "value": score.white().mate()}
                        else:
                            score_dict = {"type": "cp", "value": score.white().score()}
                            
                        top_moves.append({
                            "move": info["pv"][0].uci(),
                            "score": score_dict
                        })
                
                return {
                    "eval": eval_dict,
                    "top_moves": top_moves
                }
            else:
                return {"eval": {"type": "cp", "value": 0}, "top_moves": []}
        
        except chess.engine.EngineTerminatedError:
            # Handle engine crashes by restarting
            logger.warning(f"Engine terminated unexpectedly, restarting...")
            try:
                self.close()
                self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
                # Configure with supported options
                config = {}
                if "Threads" in self.engine.options:
                    config["Threads"] = self.engine.options.get("Threads").default
                if "Hash" in self.engine.options:
                    config["Hash"] = self.engine.options.get("Hash").default
                self.engine.configure(config)
                
                # Return a default value since the analysis failed
                return {"eval": {"type": "cp", "value": 0}, "top_moves": []}
            except Exception as e:
                logger.error(f"Failed to restart engine: {e}")
                return {"eval": {"type": "cp", "value": 0}, "top_moves": []}
        
        except Exception as e:
            logger.error(f"Error evaluating position: {e}")
            return {"eval": {"type": "cp", "value": 0}, "top_moves": []}
    
    def close(self):
        """Clean up engine resources"""
        try:
            if hasattr(self, 'engine'):
                self.engine.quit()
        except Exception as e:
            logger.error(f"Error closing Stockfish engine: {e}")

def analyze_game_worker(args: Tuple[int, Dict, str, int, int, int]) -> Dict[str, Any]:
    """Worker function for analyzing a game in parallel processing."""
    game_id, row, stockfish_path, depth, threads, hash_size = args
    
    # Initialize Stockfish handler
    try:
        stockfish = StockfishHandler(
            path=stockfish_path, 
            depth=depth,
            threads=threads,
            hash_size=hash_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize Stockfish for game {game_id}: {e}")
        return {"game_id": game_id, "status": "failed", "reason": f"stockfish init: {str(e)}"}
    
    try:
        # Get the PGN
        pgn = row.get('moves', '')
        if not isinstance(pgn, str) or not pgn:
            # Try alternative column names
            pgn = row.get('pgn', '')
            if not isinstance(pgn, str) or not pgn:
                logger.warning(f"Skipping game {game_id} due to missing or invalid PGN")
                stockfish.close()
                return {"game_id": game_id, "status": "skipped", "reason": "missing PGN"}
        
        # Parse PGN
        pgn_io = io.StringIO(pgn)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            logger.error(f"Failed to parse game {game_id}!")
            stockfish.close()
            return {"game_id": game_id, "status": "failed", "reason": "parse error"}
        
        # Initialize lists for evaluations and top moves
        evals = []
        top_moves = []
        
        # Initialize board
        board = game.board()
        
        # Analyze initial position
        init_analysis = stockfish.evaluate_position(board)
        
        # Store initial evaluation
        evals.append({
            "ply": 0,
            "eval": init_analysis["eval"]
        })
        
        # Store initial top moves
        top_moves.append({
            "ply": 0,
            "moves": init_analysis["top_moves"]
        })
        
        # Extract moves efficiently
        moves = list(game.mainline_moves())
        
        # Skip games that are excessively long (outliers)
        if len(moves) > 200:  # Most chess games are under 200 moves
            logger.warning(f"Skipping unusually long game: {game_id}")
            stockfish.close()
            return {"game_id": game_id, "status": "skipped", "reason": "excessive length"}
        
        # Analyze each position after each move
        for i, move in enumerate(moves):
            board.push(move)
            ply = i + 1
            
            # Analyze the position
            analysis = stockfish.evaluate_position(board, num_variations=3)
            
            # Store evaluation
            eval_result = {
                "ply": ply,
                "eval": analysis["eval"]
            }
            evals.append(eval_result)
            
            # Store top moves
            top_moves_result = {
                "ply": ply,
                "moves": analysis["top_moves"]
            }
            top_moves.append(top_moves_result)
        
        stockfish.close()
        
        # Put all the original row data plus the new analysis data
        row_data = row.copy()
        row_data["evaluations"] = str(evals)
        row_data["top_moves"] = str(top_moves)
        row_data["game_id"] = game_id
        row_data["status"] = "success"
        
        return row_data
    
    except Exception as e:
        logger.error(f"Error during analysis of game {game_id}: {e}")
        stockfish.close()
        return {"game_id": game_id, "status": "failed", "error": str(e)}

def prepare_output_file(csv_path: str, output_path: str):
    """Create the output CSV file with the correct header if it doesn't exist."""
    if os.path.exists(output_path):
        return
        
    # Read the header from the original file
    with open(csv_path, 'r') as f:
        header_line = f.readline().strip()
        
    # Add our new columns
    new_header = header_line + ",evaluations,top_moves\n"
    
    # Write to the output file
    with open(output_path, 'w') as f:
        f.write(new_header)
        
    logger.info(f"Created output file with header: {output_path}")

def append_results_to_csv(results: List[Dict], output_path: str, csv_columns: List[str]):
    """Append successful results directly to the CSV file."""
    successful_results = [r for r in results if r.get("status") == "success"]
    
    if not successful_results:
        logger.warning("No successful results to append to CSV")
        return
    
    logger.info(f"Appending {len(successful_results)} results to CSV")
    
    # Safely append to the CSV file
    with file_lock:
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
            for row in successful_results:
                writer.writerow(row)

def get_csv_columns(csv_path: str) -> List[str]:
    """Get the column names from the CSV file."""
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
    columns = header.split(',')
    # Add our new columns if they're not already there
    if 'evaluations' not in columns:
        columns.append('evaluations')
    if 'top_moves' not in columns:
        columns.append('top_moves')
    return columns

def process_games_parallel(
    csv_path: str, 
    output_dir: str, 
    stockfish_path: str = "stockfish", 
    depth: int = 16, 
    batch_size: int = 100, 
    limit: Optional[int] = None,
    start_from: int = 0, 
    num_processes: Optional[int] = None,
    threads_per_engine: int = 1,
    hash_size: int = 128
):
    """Process games in parallel and save results directly to CSV."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_path = os.path.join(os.path.dirname(output_dir), "chess_games_clean_1950_final_with_evals.csv")
    
    # Create checkpoint file path
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    # Set up signal handlers for graceful exit
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check if stockfish is available
        try:
            test_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            test_engine.quit()
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish at {stockfish_path}: {e}")
            print(f"ERROR: Stockfish not found at {stockfish_path}. Please check the path.")
            return
        
        # Prepare output file
        prepare_output_file(csv_path, output_path)
        
        # Get CSV columns
        csv_columns = get_csv_columns(csv_path)
        
        # Read CSV and sort by date
        logger.info(f"Reading CSV file: {csv_path}")
        
        # First, check the size of the CSV to determine if we should load it all at once
        file_size = os.path.getsize(csv_path)
        large_file = file_size > 500 * 1024 * 1024  # 500MB threshold
        
        if large_file:
            logger.info(f"Large file detected ({file_size/1024/1024:.1f} MB). Using memory-efficient loading.")
            # Count rows and get column names
            row_count = sum(1 for _ in open(csv_path)) - 1  # Subtract header
            df_sample = pd.read_csv(csv_path, nrows=1)
            has_date_column = 'date' in df_sample.columns
            
            if has_date_column and not os.path.exists(os.path.join(output_dir, "sorted_indices.npy")):
                logger.info("Pre-sorting large file by date...")
                # Create a smaller dataframe with just the indices and dates
                date_df = pd.DataFrame()
                
                for chunk in pd.read_csv(csv_path, chunksize=100000, usecols=['date']):
                    date_df = pd.concat([date_df, chunk])
                
                # Convert dates and sort
                date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')
                sorted_indices = date_df.sort_values('date').index.to_numpy()
                
                # Save sorted indices for future runs
                np.save(os.path.join(output_dir, "sorted_indices.npy"), sorted_indices)
                logger.info(f"Saved sorted indices for {len(sorted_indices)} rows")
                
                # Free memory
                del date_df
                gc.collect()
            elif has_date_column and os.path.exists(os.path.join(output_dir, "sorted_indices.npy")):
                logger.info("Loading pre-computed sorted indices...")
                sorted_indices = np.load(os.path.join(output_dir, "sorted_indices.npy"))
                logger.info(f"Loaded {len(sorted_indices)} sorted indices")
            else:
                logger.warning("No date column found or sorted indices file not present. Using original order.")
                sorted_indices = np.arange(row_count)
        else:
            logger.info(f"Loading full CSV file ({file_size/1024/1024:.1f} MB)...")
            df = pd.read_csv(csv_path)
            has_date_column = 'date' in df.columns
            
            if has_date_column:
                logger.info("Sorting games by date (ascending)...")
                # Convert date column to datetime for proper sorting
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # Sort by date ascending (oldest first)
                    df = df.sort_values(by='date').reset_index()
                    logger.info(f"Successfully sorted {len(df)} games by date")
                except Exception as e:
                    logger.warning(f"Error sorting by date: {e}. Proceeding with original order.")
            
            # Store the sorted indices
            sorted_indices = df.index.to_numpy()
            row_count = len(df)
        
        # Apply limit and start_from to the indices
        if start_from > 0:
            sorted_indices = sorted_indices[start_from:]
            
        if limit:
            sorted_indices = sorted_indices[:limit]
            
        total_rows = len(sorted_indices)
        logger.info(f"Will process {total_rows} games")
        
        # Calculate number of batches
        num_batches = (total_rows + batch_size - 1) // batch_size
        logger.info(f"Will process games in {num_batches} batches")
        
        # Check for checkpoint
        checkpoint_data = {}
        completed_batches = set()
        last_completed_batch = -1
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    completed_batches = set(checkpoint_data.get('completed_batches', []))
                    last_completed_batch = checkpoint_data.get('last_completed_batch', -1)
                    
                logger.info(f"Found checkpoint: {len(completed_batches)} batches already processed")
                logger.info(f"Last completed batch: {last_completed_batch}")
            except Exception as e:
                logger.warning(f"Error reading checkpoint file: {e}. Starting from scratch.")
        
        # Determine number of processes
        if num_processes is None:
            # Default to number of CPU cores minus 1, but ensure it's at least 1
            num_processes = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {num_processes} processes with {threads_per_engine} threads per engine")
        
        # Process each batch
        current_batch = 0
        games_processed = 0
        
        for batch_start in range(0, total_rows, batch_size):
            # Skip already processed batches
            if current_batch in completed_batches:
                logger.info(f"Skipping batch {current_batch+1}/{num_batches} (already processed)")
                current_batch += 1
                games_processed += min(batch_size, total_rows - batch_start)
                continue
                
            batch_end = min(batch_start + batch_size, total_rows)
            batch_indices = sorted_indices[batch_start:batch_end].tolist()
            
            logger.info(f"Processing batch {current_batch+1}/{num_batches} ({len(batch_indices)} games)")
            
            # Read only the rows needed for this batch
            if large_file:
                # Read rows by index for large files
                chunk_df = pd.DataFrame()
                for chunk in pd.read_csv(csv_path, chunksize=10000):
                    # Filter rows in this batch
                    batch_rows = chunk[chunk.index.isin(batch_indices)]
                    if not batch_rows.empty:
                        chunk_df = pd.concat([chunk_df, batch_rows])
                    
                    # Break early if we've found all rows
                    if len(chunk_df) >= len(batch_indices):
                        break
                
                chunk = chunk_df
            else:
                # For smaller files, use the full dataframe
                chunk = df.loc[batch_indices].copy()
            
            # Prepare arguments for parallel processing
            args_list = []
            for i, row in chunk.iterrows():
                args_list.append((
                    i,  # Use the original index as game_id
                    row.to_dict(), 
                    stockfish_path, 
                    depth,
                    threads_per_engine,
                    hash_size
                ))
            
            # Process games in parallel
            results = []
            start_time = time.time()
            
            with Pool(processes=num_processes) as pool:
                for result in tqdm(
                    pool.imap(analyze_game_worker, args_list),
                    total=len(args_list),
                    desc=f"Batch {current_batch+1}/{num_batches}"
                ):
                    results.append(result)
            
            end_time = time.time()
            batch_time = end_time - start_time
            games_per_second = len(results) / batch_time if batch_time > 0 else 0
            logger.info(f"Batch {current_batch+1} completed in {batch_time:.1f} seconds ({games_per_second:.2f} games/sec)")
            
            # Directly append results to CSV
            append_results_to_csv(results, output_path, csv_columns)
            
            # Update checkpoint
            completed_batches.add(current_batch)
            last_completed_batch = current_batch
            
            checkpoint_data = {
                'completed_batches': list(completed_batches),
                'last_completed_batch': last_completed_batch,
                'total_batches': num_batches,
                'batch_size': batch_size,
                'total_games': total_rows,
                'games_processed': games_processed + len(chunk),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Update counters
            games_processed += len(chunk)
            current_batch += 1
            
            # Optional: Pause between batches to let system resources recover
            time.sleep(1)
            
            # Force garbage collection to free memory
            gc.collect()
        
        logger.info(f"Processing completed. Processed {games_processed} games in {current_batch} batches.")
        
        # Create a completion marker file
        with open(os.path.join(output_dir, "processing_complete.txt"), 'w') as f:
            f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total games processed: {games_processed}\n")
            f.write(f"Output file: {output_path}\n")
        
        # Clear checkpoint after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed after successful completion.")
    
    except Exception as e:
        logger.error(f"Error processing games: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Analyze chess games with Stockfish')
    parser.add_argument('--csv', type=str, 
                        default="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/chess_games_clean_1950_final.csv", 
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, 
                        default="/proj/chess/thesis_uppsala_chess_samir/data/processed/lumbrasgigabase/", 
                        help='Directory to save output files')
    parser.add_argument('--stockfish', type=str, 
                        default="/proj/chess/stockfish/stockfish-ubuntu-x86-64-avx2", 
                        help='Path to Stockfish executable')
    parser.add_argument('--depth', type=int, 
                        default=18, 
                        help='Analysis depth')
    parser.add_argument('--batch', type=int, 
                        default=100, 
                        help='Batch size')
    parser.add_argument('--limit', type=int, 
                        default=None, 
                        help='Limit on the number of games to process')
    parser.add_argument('--start', type=int, 
                        default=0, 
                        help='Index to start processing from')
    parser.add_argument('--processes', type=int, 
                        default=None, 
                        help='Number of parallel processes')
    parser.add_argument('--threads', type=int, 
                        default=1, 
                        help='Threads per Stockfish engine')
    parser.add_argument('--hash', type=int, 
                        default=2048, 
                        help='Hash size in MB for Stockfish')
    
    args = parser.parse_args()
    
    # Create analysis subdirectory for intermediate files
    analysis_dir = os.path.join(args.output, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Display configuration
    print(f"Configuration:")
    print(f"  CSV file: {args.csv}")
    print(f"  Output directory: {args.output}")
    print(f"  Analysis files directory: {analysis_dir}")
    print(f"  Stockfish path: {args.stockfish}")
    print(f"  Analysis depth: {args.depth}")
    print(f"  Batch size: {args.batch}")
    print(f"  Game limit: {args.limit if args.limit else 'None (all games)'}")
    print(f"  Starting index: {args.start}")
    print(f"  Processes: {args.processes if args.processes else 'Auto'}")
    print(f"  Threads per engine: {args.threads}")
    print(f"  Hash size: {args.hash} MB")
    print()
    
    # Record start time
    start_time = time.time()
    
    # Process games
    process_games_parallel(
        args.csv, 
        analysis_dir,  # Use analysis subdirectory for intermediate files
        args.stockfish, 
        args.depth, 
        args.batch, 
        args.limit,
        args.start,
        args.processes,
        args.threads,
        args.hash
    )
    
    # Record end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Analysis completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final output saved to: {os.path.join(args.output, 'chess_games_clean_1950_final_with_evals.csv')}")

if __name__ == "__main__":
    main()