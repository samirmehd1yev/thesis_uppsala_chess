"""
Feature Processing Module
This module handles the processing of chess games and storage of extracted features.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import time
from game_features import ChessFeatureExtractor

def setup_logging(log_dir="../../logs"):
    """Set up logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'feature_extraction_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_features_table(conn):
    """Create the features table in the database"""
    create_features_table = """
    CREATE TABLE IF NOT EXISTS player_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        white_player TEXT,
        black_player TEXT,
        result TEXT,
        total_moves INTEGER,
        captures_per_move FLOAT,
        checks_per_move FLOAT,
        pawn_move_ratio FLOAT,
        center_move_ratio FLOAT,
        avg_material_imbalance FLOAT,
        material_volatility FLOAT,
        avg_mobility FLOAT,
        mobility_volatility FLOAT,
        avg_center_control FLOAT,
        center_control_volatility FLOAT
    )
    """
    conn.execute(create_features_table)
    conn.commit()

def process_database(db_path, batch_size=1000):
    """Process all games in the database and extract features"""
    logger = setup_logging()
    logger.info(f"Starting feature extraction from {db_path}")
    
    conn = sqlite3.connect(db_path)
    extractor = ChessFeatureExtractor()
    
    # Create features table
    create_features_table(conn)
    
    # Get total number of games
    total_games = conn.execute("SELECT COUNT(*) FROM chess_games").fetchone()[0]
    logger.info(f"Total games to process: {total_games}")
    
    try:
        # Process games in batches
        for offset in tqdm(range(0, total_games, batch_size), desc="Processing games"):
            games_df = pd.read_sql_query(f"""
                SELECT moves FROM chess_games 
                LIMIT {batch_size} OFFSET {offset}
            """, conn)
            
            features_list = []
            for _, row in games_df.iterrows():
                features = extractor.extract_features_from_game(row['moves'])
                if features:
                    features_list.append(features)
            
            if features_list:
                features_df = pd.DataFrame(features_list)
                features_df.to_sql('player_features', conn, if_exists='append', index=False)
            
            if offset % 10000 == 0:
                logger.info(f"Processed {offset + len(games_df)} games")
        
        logger.info("Feature extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    DB_PATH = Path("../../data/processed/chess_games.db")
    process_database(DB_PATH)