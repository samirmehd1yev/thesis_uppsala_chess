import chess
import chess.pgn
import io
from features.extractor import FeatureExtractor
from analysis.stockfish_handler import StockfishHandler
from analysis.move_analyzer import MoveAnalyzer
from tqdm import tqdm
from models.enums import Judgment
import pandas as pd
from tabulate import tabulate
import colorama
from colorama import Fore, Style
import multiprocessing
from multiprocessing import Pool, Manager
import os
import time
import logging
import sys
import argparse
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('chess_analyzer')

# Initialize colorama
colorama.init(autoreset=True)

# Dictionary to store debug reasons for moves
move_debug_reasons = {}

# Color mapping for move judgments
JUDGMENT_COLORS = {
    Judgment.BRILLIANT: Fore.MAGENTA,   # Magenta for brilliant moves
    Judgment.GREAT: Fore.CYAN,          # Cyan for great moves
    Judgment.GOOD: Fore.GREEN,          # Green for good moves
    Judgment.INACCURACY: Fore.YELLOW,   # Yellow for inaccuracies
    Judgment.MISTAKE: Fore.RED,         # Red for mistakes
    Judgment.BLUNDER: Fore.RED + Style.BRIGHT  # Bright red for blunders
}

def format_debug_reason(reason):
    """Format the debug reason for better readability"""
    if not reason:
        return "No reason available"
        
    # Remove leading/trailing whitespace
    reason = reason.strip()
    
    # Replace pipe separators with newlines
    formatted = reason.replace(" | ", "\n")
    
    return formatted

# Function to convert eval to numerical value for comparison
def eval_to_numerical(eval_info):
    """Convert evaluation to a numerical value that can be compared"""
    if not eval_info:
        return 0.0
        
    if eval_info.mate is not None:
        # Convert mate scores to centipawns with sign
        if eval_info.mate > 0:
            return 10000 - eval_info.mate * 10  # The smaller positive mate, the better
        else:
            return -10000 - eval_info.mate * 10  # The smaller negative mate, the worse
    elif eval_info.cp is not None:
        return float(eval_info.cp)
    else:
        return 0.0  # Default value

def get_san_move(board, move):
    """Get the move in Standard Algebraic Notation (SAN) format"""
    return board.san(move)

def format_eval(eval_info):
    """Format evaluation to a readable string"""
    if eval_info is None:
        return "?"
        
    if eval_info.mate is not None:
        return f"#{eval_info.mate}"
    elif eval_info.cp is not None:
        return f"{eval_info.cp/100:+.2f}"
    else:
        return "?"

def get_colored_judgment(judgment):
    """Return the judgment with appropriate color based on type"""
    if judgment in JUDGMENT_COLORS:
        return f"{JUDGMENT_COLORS[judgment]}{judgment.value}{Style.RESET_ALL}"
    return str(judgment)

def print_judgment_summary(judgments):
    """Print a summary of judgments to help debug counting issues"""
    white_counts = {
        Judgment.BRILLIANT: 0,
        Judgment.GREAT: 0,
        Judgment.GOOD: 0,
        Judgment.INACCURACY: 0,
        Judgment.MISTAKE: 0,
        Judgment.BLUNDER: 0
    }
    
    black_counts = {
        Judgment.BRILLIANT: 0,
        Judgment.GREAT: 0,
        Judgment.GOOD: 0,
        Judgment.INACCURACY: 0,
        Judgment.MISTAKE: 0,
        Judgment.BLUNDER: 0
    }
    
    # Count the judgments
    for i, judgment in enumerate(judgments):
        is_white = i % 2 == 0  # Even indices are White's moves
        if is_white:
            white_counts[judgment] += 1
        else:
            black_counts[judgment] += 1
    
    # Print the summary
    print("\n" + "="*60)
    print(f"{Fore.BLUE}{Style.BRIGHT}JUDGMENT COUNTS (Direct from analysis){Style.RESET_ALL}")
    print("="*60)
    print(f"\nWhite judgments:")
    for judgment, count in white_counts.items():
        print(f"  {judgment.value}: {count}")
    
    print(f"\nBlack judgments:")
    for judgment, count in black_counts.items():
        print(f"  {judgment.value}: {count}")
    print("="*60)

def calculate_eval_change(prev_eval, curr_eval, is_white_move):
    """Calculate the evaluation change in a way that handles mate scores properly"""
    if not prev_eval or not curr_eval:
        return None
        
    # Safety checks for both evaluations
    if not hasattr(prev_eval, 'cp') or not hasattr(prev_eval, 'mate') or \
       not hasattr(curr_eval, 'cp') or not hasattr(curr_eval, 'mate'):
        return None
    
    # Convert to numerical values for comparison
    prev_num = eval_to_numerical(prev_eval)
    curr_num = eval_to_numerical(curr_eval)
    
    # Calculate change
    change = curr_num - prev_num
    
    # For Black's moves, negate the change to get their perspective
    if not is_white_move:
        change = -change
        
    # Convert to pawn units for display if not dealing with mate scores
    if prev_eval.mate is None and curr_eval.mate is None:
        return change / 100
    
    # Handle mate vs no mate transitions
    if (prev_eval.mate is None and curr_eval.mate is not None) or \
       (prev_eval.mate is not None and curr_eval.mate is None):
        # If we gained a mate advantage, it's a big positive change
        if curr_eval.mate is not None and curr_eval.mate > 0:
            return 5.0  # Big positive (got mate)
        # If we lost a mate advantage or got mated, it's a big negative change
        elif curr_eval.mate is not None and curr_eval.mate < 0:
            return -5.0  # Big negative (got mated)
        # Going from a mate disadvantage to no mate is a positive
        elif prev_eval.mate is not None and prev_eval.mate < 0:
            return 3.0  # Positive (escaped mate)
        # Going from a mate advantage to no mate is a negative
        elif prev_eval.mate is not None and prev_eval.mate > 0:
            return -3.0  # Negative (lost mate)
    
    # Handle mate to mate transitions
    if prev_eval.mate is not None and curr_eval.mate is not None:
        # If same side is mating, smaller positive number is better (faster mate)
        if prev_eval.mate > 0 and curr_eval.mate > 0:
            # Better (faster) mate
            if prev_eval.mate > curr_eval.mate:
                return 0.5
            # Worse (slower) mate
            elif prev_eval.mate < curr_eval.mate:
                return -0.5
            # Same mate
            else:
                return 0.0
        # If same side is getting mated, higher negative number is better (slower mate against)
        elif prev_eval.mate < 0 and curr_eval.mate < 0:
            # Better (slower mate against)
            if abs(prev_eval.mate) < abs(curr_eval.mate):
                return 0.5
            # Worse (faster mate against)
            elif abs(prev_eval.mate) > abs(curr_eval.mate):
                return -0.5
            # Same mate
            else:
                return 0.0
        # Going from mate against to mate for is a huge swing
        elif prev_eval.mate < 0 and curr_eval.mate > 0:
            return 8.0  # Massive positive (turn the tables)
        # Going from mate for to mate against is a disaster
        elif prev_eval.mate > 0 and curr_eval.mate < 0:
            return -8.0  # Massive negative (disaster)
    
    # Default: return normalized change
    return change / 100

def print_analysis_table(game, evals, judgments, top_moves=None, show_debug_reasons=True):
    """Print the game analysis as a formatted table with proper handling of mate scores"""
    moves_list = []
    
    # Get the game mainline moves
    mainline_moves = list(game.mainline_moves())
    board = game.board()
    
    # Initialize variables to track move numbers
    move_num = 1
    is_white_move = True
    
    print("\n" + "="*90)
    print(f"{Fore.BLUE}{Style.BRIGHT}GAME ANALYSIS{Style.RESET_ALL}")
    print("="*90)
    
    # Loop through each move and evaluation
    for i, move in enumerate(mainline_moves):
        # Get move in SAN format
        san_move = get_san_move(board, move)
        
        # Get evaluations before and after the move
        prev_eval = evals[i] if i < len(evals) else None
        curr_eval = evals[i+1] if i+1 < len(evals) else None
        
        # Format evaluations
        prev_eval_str = format_eval(prev_eval) if prev_eval else "?"
        curr_eval_str = format_eval(curr_eval) if curr_eval else "?"
        
        # Get judgment for the move
        judgment = judgments[i] if i < len(judgments) else None
        judgment_str = get_colored_judgment(judgment) if judgment else ""
        
        # Get debug reason directly from logging
        debug_info = ""
        if show_debug_reasons:
            with open('chess_analyzer.log', 'r') as f:
                for line in f:
                    if f"Move {san_move}: " in line and "REASON:" in line:
                        reason_part = line.split("REASON:")[1].strip()
                        formatted_reason = format_debug_reason(reason_part)
                        debug_info = f"\n{Fore.YELLOW}WHY: {formatted_reason}{Style.RESET_ALL}"
                        break
        
        # Get top moves for this position
        top_move_list = []
        if top_moves and i < len(top_moves):
            # Convert UCI moves to SAN for readability
            temp_board = board.copy()
            for uci_move in top_moves[i][:3]:  # Get top 3 moves
                try:
                    chess_move = chess.Move.from_uci(uci_move)
                    san = temp_board.san(chess_move)
                    top_move_list.append(san)
                except ValueError:
                    top_move_list.append(uci_move)
                except Exception as e:
                    logger.warning(f"Error converting move {uci_move}: {e}")
        
        top_moves_str = ", ".join(top_move_list) if top_move_list else ""
        
        # Calculate evaluation change with better handling of mate scores
        eval_change_value = calculate_eval_change(prev_eval, curr_eval, is_white_move)
        
        # Format with +/- and color
        if eval_change_value is not None:
            if eval_change_value > 0:
                eval_change = f"{Fore.GREEN}+{abs(eval_change_value):.2f}{Style.RESET_ALL}"
            elif eval_change_value < 0:
                eval_change = f"{Fore.RED}-{abs(eval_change_value):.2f}{Style.RESET_ALL}"
            else:
                eval_change = f"{eval_change_value:.2f}"
        else:
            eval_change = ""
            
        # Add row to the table
        if is_white_move:
            moves_list.append([
                move_num,
                f"{san_move}{debug_info if is_white_move and show_debug_reasons else ''}",
                prev_eval_str,
                curr_eval_str,
                eval_change,
                judgment_str,
                top_moves_str[:20] + ("..." if len(top_moves_str) > 20 else "")
            ])
        else:
            # Update the last row with black's move
            moves_list[-1].extend([
                f"{san_move}{debug_info if not is_white_move and show_debug_reasons else ''}",
                prev_eval_str,
                curr_eval_str,
                eval_change,
                judgment_str,
                top_moves_str[:20] + ("..." if len(top_moves_str) > 20 else "")
            ])
            move_num += 1
        
        # Push the move and toggle the active player
        board.push(move)
        is_white_move = not is_white_move
    
    # Create headers for the table
    headers = [
        "#", "White", "Eval Before", "Eval After", "Change", "Judgment", "Top Moves",
        "Black", "Eval Before", "Eval After", "Change", "Judgment", "Top Moves"
    ]
    
    # Print the table
    print(tabulate(moves_list, headers=headers, tablefmt="pretty"))
    print("="*90)

def print_feature_summary(features):
    """Print the feature summary with improved formatting"""
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}GAME FEATURE SUMMARY{Style.RESET_ALL}")
    print("="*80)
    
    # Organize features into categories
    categories = {
        "Game Phase": [
            "total_moves", "opening_length", "middlegame_length", "endgame_length"
        ],
        "Material/Position": [
            "material_balance_changes", "piece_mobility_avg", 
            "pawn_structure_changes", "center_control_avg"
        ],
        "White Move Quality": [
            "white_brilliant_count", "white_great_count", "white_good_moves",
            "white_inaccuracy_count", "white_mistake_count", "white_blunder_count",
            "white_avg_eval_change", "white_eval_volatility"
        ],
        "Black Move Quality": [
            "black_brilliant_count", "black_great_count", "black_good_moves",
            "black_inaccuracy_count", "black_mistake_count", "black_blunder_count",
            "black_avg_eval_change", "black_eval_volatility"
        ]
    }
    
    # Print each category of features
    for category, feature_names in categories.items():
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{category}:{Style.RESET_ALL}")
        for name in feature_names:
            if name in features.__dict__:
                value = features.__dict__[name]
                print(f"  {name}: {value:.3f}")
    
    print("="*80)

# Function to evaluate a single position with Stockfish
def evaluate_position(args):
    position, ply, stockfish_path, depth, result_dict = args
    
    # Create a new Stockfish instance for each process
    stockfish = StockfishHandler(path=stockfish_path, depth=depth)
    
    try:
        # Evaluate the position
        result = stockfish.evaluate_position(position, ply)
        
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

# Function to analyze a single move
def analyze_move(args):
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
        
        # Calculate and format evaluation change for logging with proper mate handling
        eval_change_value = calculate_eval_change(prev_eval, curr_eval, player_id == 'white')
        
        if eval_change_value is not None:
            if eval_change_value > 0:
                eval_change = f"+{abs(eval_change_value):.2f}"
            elif eval_change_value < 0:
                eval_change = f"-{abs(eval_change_value):.2f}"
            else:
                eval_change = f"{eval_change_value:.2f}"
        else:
            eval_change = "?"
        
        # Log the analysis result with debug information
        prev_eval_str = format_eval(prev_eval)
        curr_eval_str = format_eval(curr_eval)
        move_san = prev_board.san(move) if prev_board and move else move.uci()
        logger.info(f"Move {move_san}: {prev_eval_str} -> {curr_eval_str} = {judgment} (Δ{eval_change}) | REASON: {debug_reason}")
        
        return judgment
    except Exception as e:
        logger.error(f"Error analyzing move: {e}")
        return Judgment.GOOD  # Default to GOOD on error

def analyze_pgn_game(pgn_content, stockfish_path="stockfish", analysis_depth=20, show_debug=True):
    """Analyze a PGN game and return the analysis results"""
    # Clear log file if it exists
    with open('chess_analyzer.log', 'w') as f:
        pass
        
    logger.info(f"{Fore.GREEN}Analyzing chess game...{Style.RESET_ALL}")
    start_time = time.time()
    
    # Create a file-like object from the PGN content
    pgn_io = io.StringIO(pgn_content)
    
    # Parse the game from the PGN string
    game = chess.pgn.read_game(pgn_io)
    
    if game is None:
        logger.error(f"{Fore.RED}Failed to parse game!{Style.RESET_ALL}")
        return None
    
    logger.info(f"{Fore.GREEN}Game parsed successfully!{Style.RESET_ALL}")
    
    # Initialize the feature extractor
    feature_extractor = FeatureExtractor()
    
    # Get positions from the game
    positions = feature_extractor._get_positions(game)
    
    # Parameters for position evaluation
    depth = analysis_depth  # Depth for analysis
    
    # Determine the number of CPU cores to use
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    logger.info(f"{Fore.YELLOW}Using {num_cores} CPU cores for parallel evaluation{Style.RESET_ALL}")
    
    try:
        # Create a manager for sharing data between processes
        with Manager() as manager:
            # Create a shared dictionary to store results in order
            result_dict = manager.dict()
            
            # PHASE 1: Parallel position evaluation
            logger.info(f"{Fore.YELLOW}PHASE 1: Evaluating positions with Stockfish in parallel...{Style.RESET_ALL}")
            
            # Prepare arguments for parallel evaluation
            eval_args = [(positions[i], i, stockfish_path, depth, result_dict) for i in range(len(positions))]
            
            # Evaluate positions in parallel
            with Pool(processes=num_cores) as pool:
                results = list(tqdm(
                    pool.imap(evaluate_position, eval_args), 
                    total=len(positions),
                    desc="Evaluating positions"
                ))
            
            # Convert shared dictionary to ordered list
            evals = [result_dict.get(i) for i in range(len(positions))]
            
            # Fill in any missing evaluations with neutral values
            for i in range(len(evals)):
                if evals[i] is None:
                    from models.data_classes import Info
                    logger.warning(f"Missing evaluation at ply {i}, using default")
                    evals[i] = Info(ply=i, eval={"type": "cp", "value": 0}, variation=[])
            
            logger.info(f"{Fore.GREEN}Position evaluation completed in {time.time() - start_time:.2f} seconds{Style.RESET_ALL}")
            
            # PHASE 2: Parallel move analysis
            logger.info(f"{Fore.YELLOW}PHASE 2: Analyzing moves in parallel...{Style.RESET_ALL}")
            phase2_start = time.time()
            
            # Get the game mainline moves
            mainline_moves = list(game.mainline_moves())
            
            # Prepare arguments for parallel move analysis
            analysis_args = []
            for i in range(1, len(evals)):
                move_idx = i - 1
                if move_idx < len(mainline_moves):
                    prev_eval = evals[i-1]
                    curr_eval = evals[i]
                    move = mainline_moves[move_idx]
                    prev_board = positions[move_idx]
                    curr_board = positions[move_idx + 1]
                    player_id = 'white' if (i-1) % 2 == 0 else 'black'
                    
                    # Include move_idx in the arguments
                    analysis_args.append((prev_eval, curr_eval, prev_board, curr_board, move, player_id, move_idx))
            
            # Analyze moves in parallel
            with Pool(processes=num_cores) as pool:
                judgments = list(tqdm(
                    pool.imap(analyze_move, analysis_args),
                    total=len(analysis_args),
                    desc="Analyzing moves"
                ))
            
            logger.info(f"{Fore.GREEN}Move analysis completed in {time.time() - phase2_start:.2f} seconds{Style.RESET_ALL}")
            
            # Print judgment summary to help debug counting issues
            print_judgment_summary(judgments)
            
            # Extract features
            logger.info(f"{Fore.YELLOW}Extracting game features...{Style.RESET_ALL}")
            # Pass judgments to extract_features
            features = feature_extractor.extract_features(game, evals, judgments)
            
            # Collect top moves for each position
            top_moves = [eval_info.variation for eval_info in evals if eval_info and eval_info.variation]
            
            # Print the analysis table
            print_analysis_table(game, evals, judgments, top_moves, show_debug_reasons=show_debug)
            
            # Print the feature summary
            print_feature_summary(features)
            
            logger.info(f"{Fore.GREEN}Total analysis completed in {time.time() - start_time:.2f} seconds{Style.RESET_ALL}")
            
            return {
                "game": game,
                "evals": evals,
                "judgments": judgments,
                "features": features,
                "top_moves": top_moves
            }
            
    except Exception as e:
        logger.error(f"{Fore.RED}Error during analysis: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Chess Game Analyzer')
    parser.add_argument('--pgn', type=str, help='Path to PGN file or PGN string')
    parser.add_argument('--stockfish', type=str, default="stockfish", help='Path to Stockfish executable')
    parser.add_argument('--depth', type=int, default=20, help='Analysis depth')
    parser.add_argument('--debug', action='store_true', help='Show debug information for move judgments')
    
    args = parser.parse_args()
    
    # Default PGN if none provided
    pgn_content = """
    1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Be3 Bg7 5. Qd2 c6 6. f3 b5 7. Nge2 Nbd7 8. Bh6 Bxh6 9. Qxh6 Bb7 10. a3 e5 11. O-O-O Qe7 12. Kb1 a6 13. Nc1 O-O-O 14. Nb3 exd4 15. Rxd4 c5 16. Rd1 Nb6 17. g3 Kb8 18. Na5 Ba8 19. Bh3 d5 20. Qf4+ Ka7 21. Rhe1 d4 22. Nd5 Nbxd5 23. exd5 Qd6 24. Rxd4 cxd4 25. Re7+ Kb6 26. Qxd4+ Kxa5 27. b4+ Ka4 28. Qc3 Qxd5 29. Ra7 Bb7 30. Rxb7 Qc4 31. Qxf6 Kxa3 32. Qxa6+ Kxb4 33. c3+ Kxc3 34. Qa1+ Kd2 35. Qb2+ Kd1 36. Bf1 Rd2 37. Rd7 Rxd7 38. Bxc4 bxc4 39. Qxh8 Rd3 40. Qa8 c3 41. Qa4+ Ke1 42. f4 f5 43. Kc1 Rd2 44. Qa7 1-0
    """
    
    # If PGN file path provided, read from file
    if args.pgn:
        if os.path.isfile(args.pgn):
            with open(args.pgn, 'r') as f:
                pgn_content = f.read()
        else:
            # Assume it's a PGN string
            pgn_content = args.pgn
    
    # Run the analysis
    analyze_pgn_game(pgn_content, args.stockfish, args.depth, args.debug)