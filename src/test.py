# test.py
import sqlite3
import chess
import chess.pgn
import io
from features.extractor import FeatureExtractor
from analysis.stockfish_handler import StockfishHandler
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

def evaluate_position_parallel(args):
    position, idx, stockfish_path = args
    # Create a new stockfish instance for each process
    stockfish = StockfishHandler(path=stockfish_path, depth=20)
    try:
        result = stockfish.evaluate_position(position, idx)
        stockfish.close()
        return result
    except Exception as e:
        stockfish.close()
        return None

def main():
    # Connect to database
    DB_PATH = "/proj/chess/thesis_chess_code/data/processed/chess_games.db"
    conn = sqlite3.connect(DB_PATH)

    # print tables list
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = conn.execute(query).fetchall()
    print("Tables:")
    print("-" * 40)
    for table in tables:
        print(table[0])

    # Get multiple games for batch processing
    query = """
    SELECT 
        moves,
        white, black,
        white_elo, black_elo,
        eco, opening,
        result
    FROM chess_games 
    WHERE moves IS NOT NULL
      AND white_elo IS NOT NULL
      AND black_elo IS NOT NULL
    LIMIT 10  # Increased to process multiple games
    """

    rows = conn.execute(query).fetchall()
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Number of CPU cores to use (adjust based on your HPC setup)
    num_cores = min(40, mp.cpu_count())  # Using up to 40 cores
    print(f"Using {num_cores} CPU cores")

    stockfish_path = "/crex/proj/chess/stockfish/src/stockfish"
    
    for row in rows:
        game = chess.pgn.read_game(io.StringIO(row[0]))
        
        # Get positions
        positions = feature_extractor._get_positions(game)
        
        # Prepare arguments for parallel processing
        eval_args = [(pos, i, stockfish_path) for i, pos in enumerate(positions)]
        
        # Parallel evaluation using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            evals = list(executor.map(evaluate_position_parallel, eval_args))
        
        print("\nEngine evaluations:")
        print(evals)

        # Extract features with engine
        print("\nFeatures with engine analysis:")
        print("-" * 40)
        features_with_engine = feature_extractor.extract_features(game, evals)
        for name, value in features_with_engine.__dict__.items():
            print(f"{name}: {value:.3f}")

    conn.close()

if __name__ == '__main__':
    main()