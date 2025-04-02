"""
PGN Extraction script for Botvinnik, Fischer, Capablanca, and Tal games.
Extracts games from PGN files and saves to CSV with comparison against existing data.
"""

import csv
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

# Project structure setup
PROJECT_ROOT = Path("/Users/samir/Desktop/Uppsala/Thesis/thesis_chess_code")
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
INPUT_DIR = DATA_DIR / "raw" / "pgnmentor"
OUTPUT_DIR = DATA_DIR / "processed"
COMPARISON_FILE = OUTPUT_DIR / "lumbrasgigabase" / "chess_games_clean.csv"

# Create necessary directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize logging
LOG_FILE = LOGS_DIR / "pgn_extraction.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# Player names to match (case-insensitive)
TARGET_PLAYERS = ["Botvinnik", "Fischer", "Capablanca", "Tal"]

@dataclass
class ChessGame:
    """Data structure for chess game information."""
    event: str = field(default="?")
    site: str = field(default="")
    date: str = field(default="")
    round: str = field(default="?")
    white: str = field(default="")
    black: str = field(default="")
    result: str = field(default="")
    white_elo: str = field(default="")
    black_elo: str = field(default="")
    white_title: str = field(default="")
    black_title: str = field(default="")
    eco: str = field(default="")
    opening: str = field(default="")
    time_control: str = field(default="")
    import_date: str = field(default=datetime.now().strftime("%Y-%m-%d"))
    source: str = field(default="PGNMentor")
    moves: str = field(default="")
    eval_info: Optional[str] = field(default=None)
    clock_info: Optional[str] = field(default=None)

    def clean_string(self, s: str) -> str:
        """Clean string values."""
        if not s or s.isspace():
            return ""
        return s.strip().replace('\n', ' ').replace('\r', '')

    def clean_moves(self, moves: str) -> str:
        """Clean moves string."""
        # Remove extra whitespace and normalize
        moves = ' '.join(moves.split())
        # Ensure single space after move numbers
        moves = re.sub(r'(\d+)\.\s+', r'\1. ', moves)
        return moves

    def to_dict(self) -> Dict:
        """Convert to dictionary with cleaned data."""
        return {
            "event": self.clean_string(self.event),
            "site": self.clean_string(self.site),
            "date": self.clean_string(self.date),
            "round": self.clean_string(self.round),
            "white": self.clean_string(self.white),
            "black": self.clean_string(self.black),
            "result": self.clean_string(self.result),
            "white_elo": self.clean_string(self.white_elo),
            "black_elo": self.clean_string(self.black_elo),
            "white_title": self.clean_string(self.white_title),
            "black_title": self.clean_string(self.black_title),
            "eco": self.clean_string(self.eco),
            "opening": self.clean_string(self.opening),
            "time_control": self.clean_string(self.time_control),
            "import_date": self.clean_string(self.import_date),
            "source": self.source,
            "moves": self.clean_moves(self.moves),
            "eval_info": self.eval_info,
            "clock_info": self.clock_info
        }

class PGNParser:
    """Parser for PGN files."""
    def __init__(self):
        # Pre-compile regex patterns
        self.headers_pattern = re.compile(r'\[(.*?) "(.*?)"\]')
        self.target_players_pattern = self.build_player_regex(TARGET_PLAYERS)
    
    def build_player_regex(self, players):
        """Build a regex pattern to match any of the target players."""
        player_patterns = []
        for player in players:
            player_patterns.append(f"{player}")
        pattern = r'\[(?:White|Black) ".*(?:' + '|'.join(player_patterns) + r').*"\]'
        return re.compile(pattern, re.IGNORECASE)
    
    def is_target_game(self, pgn_text):
        """Check if the game contains any of our target players."""
        return bool(self.target_players_pattern.search(pgn_text))
    
    def parse_game(self, pgn_text: str) -> Optional[ChessGame]:
        """Parse a single game from PGN text."""
        try:
            headers = {}
            for line in pgn_text.splitlines():
                header_match = self.headers_pattern.match(line)
                if header_match:
                    key = header_match.group(1)
                    value = header_match.group(2)
                    headers[key] = value
            
            # Find where the moves start (after the last header)
            moves_text = ""
            in_headers = True
            for line in pgn_text.splitlines():
                if not line.strip():
                    in_headers = False
                    continue
                if not in_headers:
                    moves_text += " " + line.strip()
            
            # Clean up moves text
            moves_text = moves_text.strip()
            
            game = ChessGame(
                event=headers.get("Event", ""),
                site=headers.get("Site", ""),
                date=headers.get("Date", ""),
                round=headers.get("Round", ""),
                white=headers.get("White", ""),
                black=headers.get("Black", ""),
                result=headers.get("Result", ""),
                white_elo=headers.get("WhiteElo", ""),
                black_elo=headers.get("BlackElo", ""),
                white_title=headers.get("WhiteTitle", ""),
                black_title=headers.get("BlackTitle", ""),
                eco=headers.get("ECO", ""),
                opening=headers.get("Opening", ""),
                time_control=headers.get("TimeControl", ""),
                moves=moves_text
            )
            return game
        
        except Exception as e:
            logging.error(f"Error parsing game: {e}")
            return None

def process_pgn_file(file_path: Path, parser: PGNParser) -> List[Dict]:
    """Process a PGN file and extract games with target players."""
    games = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Split the file into individual games
        pgn_texts = re.split(r'(?=\[Event ")', content)
        
        for pgn_text in pgn_texts:
            if not pgn_text.strip():
                continue
                
            # Check if this game contains any of our target players
            if parser.is_target_game(pgn_text):
                game = parser.parse_game(pgn_text)
                if game:
                    games.append(game.to_dict())
    
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
    
    return games

def compare_with_existing_data(new_games_df, comparison_file):
    """Compare the extracted games with existing processed data."""
    try:
        # Load the existing data
        existing_df = pd.read_csv(comparison_file)
        
        # Count total games in the new extraction
        total_new_games = len(new_games_df)
        
        # Count how many games are already in the existing dataset
        # We'll compare based on the 'moves' field
        merged_df = new_games_df.merge(
            existing_df, 
            on='moves', 
            how='left', 
            indicator=True
        )
        
        already_present = len(merged_df[merged_df['_merge'] == 'both'])
        unique_games = len(merged_df[merged_df['_merge'] == 'left_only'])
        
        logging.info(f"Total extracted games: {total_new_games}")
        logging.info(f"Games already in existing dataset: {already_present} ({already_present/total_new_games*100:.2f}%)")
        logging.info(f"Unique games not in existing dataset: {unique_games} ({unique_games/total_new_games*100:.2f}%)")
        
        return {
            'total_new_games': total_new_games,
            'already_present': already_present,
            'unique_games': unique_games,
            'percentage_present': already_present/total_new_games*100 if total_new_games > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Error comparing datasets: {e}")
        return None

def main():
    """Main function to extract and process PGN files."""
    start_time = datetime.now()
    logging.info(f"Starting extraction at {start_time}")
    
    # Initialize parser
    parser = PGNParser()
    
    # Find all PGN files
    pgn_files = list(INPUT_DIR.glob('*.pgn'))
    
    if not pgn_files:
        logging.warning(f"No PGN files found in {INPUT_DIR}")
        return
    
    all_games = []
    
    # Process each PGN file
    for pgn_file in tqdm(pgn_files, desc="Processing PGN files"):
        games = process_pgn_file(pgn_file, parser)
        all_games.extend(games)
        logging.info(f"Extracted {len(games)} games from {pgn_file.name}")
    
    # Save to CSV
    output_file = OUTPUT_DIR / "4players_example.csv"
    
    if all_games:
        fieldnames = list(all_games[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_games)
        
        logging.info(f"Saved {len(all_games)} games to {output_file}")
        
        # Compare with existing data
        new_games_df = pd.DataFrame(all_games)
        comparison_results = compare_with_existing_data(new_games_df, COMPARISON_FILE)
        
        # Print final results
        if comparison_results:
            print("\n--- Comparison Results ---")
            print(f"Total extracted games: {comparison_results['total_new_games']}")
            print(f"Games already in existing dataset: {comparison_results['already_present']} "
                  f"({comparison_results['percentage_present']:.2f}%)")
            print(f"Unique games not in existing dataset: {comparison_results['unique_games']} "
                  f"({100-comparison_results['percentage_present']:.2f}%)")
    else:
        logging.warning("No games were extracted")
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    logging.info(f"Processing completed in {processing_time}")

if __name__ == "__main__":
    main()