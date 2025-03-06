import os
import pandas as pd
import chess
import chess.pgn
import chess.engine
import io
from typing import List, Dict, Optional, Any, Tuple
import logging
import multiprocessing
from multiprocessing import Pool
import time
from tqdm import tqdm
import json
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_batch_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chess_batch_analyzer')

class StockfishHandler:
    def __init__(self, path: str = "stockfish", depth: int = 16, threads: int = 1, hash_size: int = 128):
        """Initialize the StockfishHandler with chess.engine library."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            # Configure engine parameters
            self.engine.configure({
                "Threads": threads,
                "Hash": hash_size
            })
            self.depth = depth
            self.path = path
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
        
        # Get moves
        moves = list(game.mainline_moves())
        
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
        
        # Create structured result
        result = {
            "game_id": game_id,
            "status": "success",
            "white": row.get('white', ''),
            "black": row.get('black', ''),
            "result": row.get('result', ''),
            "total_moves": len(moves),
            "evaluations": evals,
            "top_moves": top_moves
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error during analysis of game {game_id}: {e}")
        stockfish.close()
        return {"game_id": game_id, "status": "failed", "error": str(e)}

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
    """Process games in parallel and save results in batches, sorted by date."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint file path
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    try:
        # Check if stockfish is available
        try:
            test_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            test_engine.quit()
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish at {stockfish_path}: {e}")
            print(f"ERROR: Stockfish not found at {stockfish_path}. Please check the path.")
            return
        
        # Read CSV and sort by date
        logger.info(f"Reading CSV file: {csv_path}")
        
        # First, read a sample to check if 'date' column exists
        sample_df = pd.read_csv(csv_path, nrows=1)
        has_date_column = 'date' in sample_df.columns
        
        if has_date_column:
            logger.info("Sorting games by date (ascending)...")
            # Read the entire CSV
            df = pd.read_csv(csv_path)
            
            # Convert date column to datetime for proper sorting
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Sort by date ascending (oldest first)
                df = df.sort_values(by='date').reset_index(drop=True)
                logger.info(f"Successfully sorted {len(df)} games by date")
            except Exception as e:
                logger.warning(f"Error sorting by date: {e}. Proceeding with original order.")
        else:
            logger.warning("No 'date' column found. Proceeding with original order.")
            df = pd.read_csv(csv_path)
        
        # Apply limit and start_from
        if start_from > 0:
            df = df.iloc[start_from:]
            
        if limit:
            df = df.iloc[:limit]
            
        total_rows = len(df)
        logger.info(f"Processing {total_rows} games")
        
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
            num_processes = max(1, multiprocessing.cpu_count())
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
            chunk = df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {current_batch+1}/{num_batches} ({len(chunk)} games)")
            
            # Prepare arguments for parallel processing
            args_list = []
            for i, row in chunk.iterrows():
                game_id = i  # Use the actual dataframe index as game_id
                args_list.append((
                    game_id, 
                    row.to_dict(), 
                    stockfish_path, 
                    depth,
                    threads_per_engine,
                    hash_size
                ))
            
            # Process games in parallel
            results = []
            with Pool(processes=num_processes) as pool:
                for result in tqdm(
                    pool.imap(analyze_game_worker, args_list),
                    total=len(args_list),
                    desc=f"Batch {current_batch+1}/{num_batches}"
                ):
                    results.append(result)
            
            # Save batch results
            output_path = os.path.join(output_dir, f"chess_analysis_batch_{current_batch}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved batch {current_batch+1} results to {output_path}")
            
            # Update checkpoint
            completed_batches.add(current_batch)
            last_completed_batch = current_batch
            
            checkpoint_data = {
                'completed_batches': list(completed_batches),
                'last_completed_batch': last_completed_batch,
                'total_batches': num_batches,
                'batch_size': batch_size,
                'total_games': total_rows,
                'games_processed': games_processed + len(chunk)
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Update counters
            games_processed += len(chunk)
            current_batch += 1
            
            # Optional: Pause between batches to let system resources recover
            time.sleep(1)
        
        # Create combined CSV with evaluations and top moves
        combine_results_to_csv(output_dir, num_batches, csv_path)
        
        logger.info(f"Processing completed. Processed {games_processed} games in {current_batch} batches.")
        
        # Clear checkpoint after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed after successful completion.")
    
    except Exception as e:
        logger.error(f"Error processing games: {e}")
        raise

def combine_results_to_csv(output_dir, num_batches, csv_path):
    """Combine all batch results with the original CSV file."""
    logger.info("Combining results with original CSV...")
    
    # Output file path
    output_path = os.path.join(os.path.dirname(output_dir), "chess_games_clean_1950_final_with_evals.csv")
    
    # Check if output file already exists - we'll use it as a checkpoint
    if os.path.exists(output_path):
        logger.info(f"Found existing output file: {output_path}")
        print(f"Output file already exists. To regenerate, please delete: {output_path}")
        return
    
    # Dictionary to store results by game_id
    game_results = {}
    
    # Process each batch
    for batch_num in range(num_batches):
        batch_file = os.path.join(output_dir, f"chess_analysis_batch_{batch_num}.json")
        
        if not os.path.exists(batch_file):
            logger.warning(f"Batch file not found: {batch_file}")
            continue
        
        try:
            with open(batch_file, 'r') as f:
                batch_results = json.load(f)
            
            for result in batch_results:
                if result.get("status") != "success":
                    continue
                
                game_id = result.get("game_id")
                
                # Store evaluations and top moves as JSON strings
                evaluations_json = json.dumps(result.get("evaluations", []))
                top_moves_json = json.dumps(result.get("top_moves", []))
                
                game_results[game_id] = {
                    "evaluations": evaluations_json,
                    "top_moves": top_moves_json
                }
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
    
    # Read the original CSV in chunks to avoid memory issues
    logger.info(f"Adding evaluation data to CSV and saving to: {output_path}")
    
    # Write header in the first chunk
    first_chunk = True
    
    # Track progress
    total_rows = sum(1 for _ in pd.read_csv(csv_path, chunksize=1000))
    total_rows *= 1000  # Approximate total rows
    rows_processed = 0
    
    with tqdm(total=total_rows, desc="Adding evaluation data to CSV") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=1000):
            # Add new columns to each chunk
            chunk['evaluations'] = None
            chunk['top_moves'] = None
            
            # Update rows with results if available
            for idx, row in chunk.iterrows():
                if idx in game_results:
                    chunk.at[idx, 'evaluations'] = game_results[idx]['evaluations']
                    chunk.at[idx, 'top_moves'] = game_results[idx]['top_moves']
            
            # Write to output file
            if first_chunk:
                chunk.to_csv(output_path, index=False)
                first_chunk = False
            else:
                chunk.to_csv(output_path, mode='a', header=False, index=False)
            
            rows_processed += len(chunk)
            pbar.update(len(chunk))
    
    logger.info(f"Successfully added evaluation data to {rows_processed} rows")
    logger.info(f"Enhanced CSV saved to: {output_path}")
    
    # Create a completion marker file
    with open(os.path.join(output_dir, "processing_complete.txt"), 'w') as f:
        f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total games processed: {len(game_results)}\n")
        f.write(f"Output file: {output_path}\n")

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
                        default="stockfish", 
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
                        default=128, 
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
    
    print(f"Final output saved to: {os.path.join(args.output, 'chess_games_clean_1950_final_with_evals.csv')}")

if __name__ == "__main__":
    main()