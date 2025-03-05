#!/usr/bin/env python3
# src/test.py
import argparse
import os
import sys
import logging
import colorama
from colorama import Fore, Style
import chess
import chess.pgn
import io
from tabulate import tabulate
from models.enums import Judgment
from analysis.game_analyzer import GameAnalyzer

# Initialize colorama for colored output
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('chess_analyzer')

# Color mapping for move judgments
JUDGMENT_COLORS = {
    Judgment.BRILLIANT: Fore.MAGENTA,   # Magenta for brilliant moves
    Judgment.GREAT: Fore.CYAN,          # Cyan for great moves
    Judgment.GOOD: Fore.GREEN,          # Green for good moves
    Judgment.INACCURACY: Fore.YELLOW,   # Yellow for inaccuracies
    Judgment.MISTAKE: Fore.RED,         # Red for mistakes
    Judgment.BLUNDER: Fore.RED + Style.BRIGHT  # Bright red for blunders
}

def get_colored_judgment(judgment):
    """Return the judgment with appropriate color based on type"""
    if judgment in JUDGMENT_COLORS:
        return f"{JUDGMENT_COLORS[judgment]}{judgment.value}{Style.RESET_ALL}"
    return str(judgment)

def format_eval(eval_info):
    """Format evaluation to a readable string"""
    if eval_info is None or not hasattr(eval_info, 'eval'):
        return "?"
        
    if eval_info.eval.get('type') == 'mate':
        return f"#{eval_info.eval.get('value')}"
    elif eval_info.eval.get('type') == 'cp':
        return f"{eval_info.eval.get('value')/100:+.2f}"
    else:
        return "?"

def calculate_eval_change(prev_eval, curr_eval, is_white_move):
    """Calculate evaluation change with proper handling of mate scores"""
    if not prev_eval or not curr_eval:
        return None
        
    prev_type = prev_eval.eval.get('type')
    prev_value = prev_eval.eval.get('value', 0)
    curr_type = curr_eval.eval.get('type')
    curr_value = curr_eval.eval.get('value', 0)
    
    # Handle centipawn evaluations
    if prev_type == 'cp' and curr_type == 'cp':
        change = curr_value - prev_value
        # For Black's moves, negate the change to get their perspective
        if not is_white_move:
            change = -change
        return change / 100  # Convert to pawn units
    
    # Handle mate scores - simplified for display purposes
    if curr_type == 'mate' and curr_value > 0:
        return 5.0  # Winning mate
    elif curr_type == 'mate' and curr_value < 0:
        return -5.0  # Losing mate
    elif prev_type == 'mate' and prev_value > 0 and curr_type == 'cp':
        return -3.0  # Lost mate advantage
    elif prev_type == 'mate' and prev_value < 0 and curr_type == 'cp':
        return 3.0  # Escaped mate disadvantage
        
    # Default to a small change
    return 0.0

def print_king_safety_analysis(analysis_result):
    """Print the king safety analysis in a formatted table"""
    if not analysis_result:
        print("No analysis available.")
        return
        
    features = analysis_result.get("features")
    
    if not features:
        return
        
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}KING SAFETY ANALYSIS{Style.RESET_ALL}")
    print("="*80)
    
    # Format safety scores with color indicators
    def format_safety(safety_value):
        if safety_value > 0:
            return f"{Fore.GREEN}{safety_value:.1f}{Style.RESET_ALL}"
        elif safety_value < -200:
            return f"{Fore.RED}{safety_value:.1f}{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{safety_value:.1f}{Style.RESET_ALL}"
    
    # Format vulnerability spikes with color indicators  
    def format_vulnerability(value):
        if value <= 1:
            return f"{Fore.GREEN}{value:.0f}{Style.RESET_ALL}"
        elif value <= 3:
            return f"{Fore.YELLOW}{value:.0f}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{value:.0f}{Style.RESET_ALL}"
    
    # Display the king safety metrics
    print(f"White King Safety (Average): {format_safety(features.white_king_safety)}")
    print(f"Black King Safety (Average): {format_safety(features.black_king_safety)}")
    print(f"White King Safety (Minimum): {format_safety(features.white_king_safety_min)}")
    print(f"Black King Safety (Minimum): {format_safety(features.black_king_safety_min)}")
    print(f"White Vulnerability Spikes: {format_vulnerability(features.white_vulnerability_spikes)}")
    print(f"Black Vulnerability Spikes: {format_vulnerability(features.black_vulnerability_spikes)}")
    
    print("\nInterpreting King Safety Scores:")
    print(f"  {Fore.GREEN}Positive values{Style.RESET_ALL}: King is well protected")
    print(f"  {Fore.YELLOW}Slight negative{Style.RESET_ALL}: Some weaknesses in king position")
    print(f"  {Fore.RED}Strong negative{Style.RESET_ALL}: King is vulnerable to attack")
    
    print("\nVulnerability Spikes indicate sudden drops in king safety,")
    print("which often coincide with tactical opportunities for the opponent.")
    
    print("="*80)

def print_analysis_table(analysis_result):
    """Print the game analysis as a formatted table"""
    if not analysis_result:
        print("No analysis available.")
        return
        
    game = analysis_result.get("game")
    evals = analysis_result.get("evals", [])
    judgments = analysis_result.get("judgments", [])
    top_moves = analysis_result.get("top_moves", [])
    sharpness_scores = analysis_result.get("sharpness", [])
    move_accuracies = analysis_result.get("move_accuracies", [])
    
    # Create a map of move accuracies for easy lookup
    accuracy_map = {(m["move_number"], m["player"]): m["accuracy"] for m in move_accuracies}
    
    moves_list = []
    
    # Get the game mainline moves
    mainline_moves = list(game.mainline_moves())
    board = game.board()
    
    # Initialize variables to track move numbers
    move_num = 1
    is_white_move = True
    
    print("\n" + "="*110)
    print(f"{Fore.BLUE}{Style.BRIGHT}GAME ANALYSIS{Style.RESET_ALL}")
    print("="*110)
    
    # Loop through each move and evaluation
    for i, move in enumerate(mainline_moves):
        # Get move in SAN format
        san_move = board.san(move)
        
        # Get evaluations before and after the move
        prev_eval = evals[i] if i < len(evals) else None
        curr_eval = evals[i+1] if i+1 < len(evals) else None
        
        # Get sharpness for current position
        current_sharpness = sharpness_scores[i].get('sharpness', 0.0) if i < len(sharpness_scores) else 0.0
        
        # Format sharpness with color
        def format_sharpness(sharp_value):
            if sharp_value < 2.0:
                return f"{Fore.GREEN}{sharp_value:.1f}{Style.RESET_ALL}"
            elif sharp_value < 5.0:
                return f"{Fore.YELLOW}{sharp_value:.1f}{Style.RESET_ALL}"
            else:
                return f"{Fore.RED}{sharp_value:.1f}{Style.RESET_ALL}"
        
        # Format evaluations
        prev_eval_str = format_eval(prev_eval)
        curr_eval_str = format_eval(curr_eval)
        
        # Get judgment for the move
        judgment = judgments[i] if i < len(judgments) else None
        judgment_str = get_colored_judgment(judgment) if judgment else ""
        
        # Get top moves for this position
        top_move_list = []
        if i < len(top_moves):
            # Convert UCI moves to SAN for readability
            temp_board = board.copy()
            for uci_move in top_moves[i][:3]:  # Get top 3 moves
                try:
                    chess_move = chess.Move.from_uci(uci_move)
                    san = temp_board.san(chess_move)
                    top_move_list.append(san)
                except Exception:
                    pass
        
        top_moves_str = ", ".join(top_move_list) if top_move_list else ""
        
        # Calculate evaluation change
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
            
        # Get move accuracy
        player = "white" if is_white_move else "black"
        accuracy = accuracy_map.get((move_num, player), 0.0)
        
        # Format accuracy with color
        def format_accuracy(acc_value):
            if acc_value >= 90:
                return f"{Fore.GREEN}{acc_value:.1f}%{Style.RESET_ALL}"
            elif acc_value >= 70:
                return f"{Fore.YELLOW}{acc_value:.1f}%{Style.RESET_ALL}"
            else:
                return f"{Fore.RED}{acc_value:.1f}%{Style.RESET_ALL}"
        
        accuracy_str = format_accuracy(accuracy)
            
        # Add row to the table
        if is_white_move:
            moves_list.append([
                move_num,
                san_move,
                prev_eval_str,
                curr_eval_str,
                eval_change,
                format_sharpness(current_sharpness),
                judgment_str,
                accuracy_str,
                top_moves_str[:20] + ("..." if len(top_moves_str) > 20 else "")
            ])
        else:
            # Update the last row with black's move
            moves_list[-1].extend([
                san_move,
                prev_eval_str,
                curr_eval_str,
                eval_change,
                format_sharpness(current_sharpness),
                judgment_str,
                accuracy_str,
                top_moves_str[:20] + ("..." if len(top_moves_str) > 20 else "")
            ])
            move_num += 1
        
        # Push the move and toggle the active player
        board.push(move)
        is_white_move = not is_white_move
    
    # Create headers for the table
    headers = [
        "#", "White", "Eval Before", "Eval After", "Change", "Sharpness", "Judgment", "Accuracy", "Top Moves",
        "Black", "Eval Before", "Eval After", "Change", "Sharpness", "Judgment", "Accuracy", "Top Moves"
    ]
    
    # Print the table
    print(tabulate(moves_list, headers=headers, tablefmt="pretty"))
    print("="*110)

def print_feature_summary(features):
    """Print the feature summary with formatting"""
    if not features:
        return
        
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
            "white_sacrifice_count", "white_avg_eval_change", "white_eval_volatility",
            "white_accuracy"
        ],
        "Black Move Quality": [
            "black_brilliant_count", "black_great_count", "black_good_moves",
            "black_inaccuracy_count", "black_mistake_count", "black_blunder_count",
            "black_sacrifice_count", "black_avg_eval_change", "black_eval_volatility",
            "black_accuracy"
        ]
    }
    
    # Print each category of features
    for category, feature_names in categories.items():
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{category}:{Style.RESET_ALL}")
        for name in feature_names:
            if hasattr(features, name):
                value = getattr(features, name)
                if name == "white_accuracy" or name == "black_accuracy":
                    # Format accuracy with color
                    if value >= 90:
                        print(f"  {name}: {Fore.GREEN}{value:.1f}%{Style.RESET_ALL}")
                    elif value >= 70:
                        print(f"  {name}: {Fore.YELLOW}{value:.1f}%{Style.RESET_ALL}")
                    else:
                        print(f"  {name}: {Fore.RED}{value:.1f}%{Style.RESET_ALL}")
                elif isinstance(value, (int, float)):
                    print(f"  {name}: {value:.2f}")
                else:
                    print(f"  {name}: {value}")
    
    print("="*80)

def print_judgment_summary(judgments):
    """Print a summary of judgments for debugging"""
    white_counts = {judgment: 0 for judgment in Judgment}
    black_counts = {judgment: 0 for judgment in Judgment}
    
    # Count the judgments
    for i, judgment in enumerate(judgments):
        is_white = i % 2 == 0  # Even indices are White's moves
        if is_white:
            white_counts[judgment] += 1
        else:
            black_counts[judgment] += 1
    
    # Print the summary
    print("\n" + "="*60)
    print(f"{Fore.BLUE}{Style.BRIGHT}JUDGMENT COUNTS{Style.RESET_ALL}")
    print("="*60)
    print(f"\nWhite judgments:")
    for judgment, count in white_counts.items():
        print(f"  {JUDGMENT_COLORS.get(judgment, '')}{judgment.value}{Style.RESET_ALL}: {count}")
    
    print(f"\nBlack judgments:")
    for judgment, count in black_counts.items():
        print(f"  {JUDGMENT_COLORS.get(judgment, '')}{judgment.value}{Style.RESET_ALL}: {count}")
    print("="*60)

def print_sharpness_summary(cumulative_sharpness):
    """Print the sharpness summary with formatting"""
    if not cumulative_sharpness:
        return
        
    print("\n" + "="*80)
    print(f"{Fore.BLUE}{Style.BRIGHT}POSITION SHARPNESS SUMMARY{Style.RESET_ALL}")
    print("="*80)
    
    overall = cumulative_sharpness.get('sharpness', 0.0)
    white = cumulative_sharpness.get('white_sharpness', 0.0)
    black = cumulative_sharpness.get('black_sharpness', 0.0)
    
    # Color coding for sharpness levels
    def sharpness_color(value):
        if value < 2.0:
            return Fore.GREEN  # Low sharpness - calm/solid positions
        elif value < 5.0:
            return Fore.YELLOW  # Medium sharpness
        else:
            return Fore.RED  # High sharpness - tactical/volatile positions
    
    print(f"Overall Cumulative Sharpness: {sharpness_color(overall)}{overall:.2f}{Style.RESET_ALL}")
    print(f"White's Cumulative Sharpness: {sharpness_color(white)}{white:.2f}{Style.RESET_ALL} (positions where White is to move)")
    print(f"Black's Cumulative Sharpness: {sharpness_color(black)}{black:.2f}{Style.RESET_ALL} (positions where Black is to move)")
    
    print("\nSharpness Interpretation:")
    print(f"  {Fore.GREEN}0-2{Style.RESET_ALL}: Calm, positional play")
    print(f"  {Fore.YELLOW}2-5{Style.RESET_ALL}: Moderately complex")
    print(f"  {Fore.RED}5-10{Style.RESET_ALL}: Highly tactical/volatile")
    
    print("="*80)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Chess Game Analyzer')
    parser.add_argument('--pgn', type=str, help='Path to PGN file or PGN string')
    parser.add_argument('--stockfish', type=str, default="stockfish", help='Path to Stockfish executable')
    parser.add_argument('--depth', type=int, default=20, help='Analysis depth')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads per Stockfish instance')
    parser.add_argument('--hash', type=int, default=128, help='Hash size in MB for Stockfish')
    parser.add_argument('--cpus', type=int, help='Number of CPU cores to use (default: cpu_count - 1)')
    parser.add_argument('--output', type=str, help='Path to output file for analysis report (HTML or TXT)')
    parser.add_argument('--quiet', action='store_true', help='Only output errors')
    
    args = parser.parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Default PGN if none provided
    pgn_content = """
    1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Be3 Bg7 5. Qd2 c6 6. f3 b5 7. Nge2 Nbd7 8. Bh6 Bxh6 9. Qxh6 Bb7 10. a3 e5 11. O-O-O Qe7 12. Kb1 a6 13. Nc1 O-O-O 14. Nb3 exd4 15. Rxd4 c5 16. Rd1 Nb6 17. g3 Kb8 18. Na5 Ba8 19. Bh3 d5 20. Qf4+ Ka7 21. Rhe1 d4 22. Nd5 Nbxd5 23. exd5 Qd6 24. Rxd4 cxd4 25. Re7+ Kb6 26. Qxd4+ Kxa5 27. b4+ Ka4 28. Qc3 Qxd5 29. Ra7 Bb7 30. Rxb7 Qc4 31. Qxf6 Kxa3 32. Qxa6+ Kxb4 33. c3+ Kxc3 34. Qa1+ Kd2 35. Qb2+ Kd1 36. Bf1 Rd2 37. Rd7 Rxd7 38. Bxc4 bxc4 39. Qxh8 Rd3 40. Qa8 c3 41. Qa4+ Ke1 42. f4 f5 43. Kc1 Rd2 44. Qa7 1-0
    """
#     pgn_content = """
#         1. e4 c5 2. Nf3 Nc6 3. Bb5 d6 4. O-O Bd7 5. Re1 Nf6 6. c3 a6 7. Ba4 c4 8. d4
# cxd3 9. Bg5 e6 10. Qxd3 Be7 11. Bxf6 gxf6 12. Bxc6 Bxc6 13. c4 O-O 14. Nc3 Kh8
# 15. Rad1 Rg8 16. Qe3 Qf8 17. Nd4 Rc8 18. f4 Bd7 19. b3 Bd8 20. Nf3 b5 21. Qa7
# Bc7 22. Qxa6 bxc4 23. b4 Qg7 24. g3 d5 25. exd5 Bxf4 26. Kf2 f5 27. gxf4 Qxc3
# 28. Qd6 Ba4 29. Rd4 Rg7 30. dxe6 Bc6 31. Ng5 Rxg5 32. Qe5+ Rg7 33. Rd8+ Rxd8 34.
# Qxc3 f6 35. e7 Ra8 36. Qxf6 Be4 37. Rg1 Rxa2+ 38. Ke1 1-0
#     """
    # pgn_content = """
    # 1. d4 d5 2. c4 c6 3. Nc3 Nf6 4. e3 e6 5. Nf3 Nbd7 6. Bd3 dxc4 7. Bxc4 b5 8. Bd3 Bb7 9. O-O a6 10. e4 c5 11. d5 Qc7 12. dxe6 fxe6 13. Bc2 c4 14. Nd4 Nc5 15. Be3 e5 16. Nf3 Be7 17. Ng5 O-O 18. Bxc5 Bxc5 19. Ne6 Qb6 20. Nxf8 Rxf8 21. Nd5 Bxd5 22. exd5 Bxf2+ 23. Kh1 e4 24. Qe2 e3 25. Rfd1 Qd6 26. a4 g6 27. axb5 axb5 28. g3 Nh5 29. Qg4 Bxg3 30. hxg3 Nxg3+ 31. Kg2 Rf2+ 32. Kh3 Nf5 33. Rh1 h5 34. Qxg6+ Qxg6 35. Rhg1 Qxg1 36. Rxg1+ Kf7 { 0-1 Black wins. } 0-1
    # """
    
    # If PGN file path provided, read from file
    if args.pgn:
        if os.path.isfile(args.pgn):
            with open(args.pgn, 'r') as f:
                pgn_content = f.read()
        else:
            # Assume it's a PGN string
            pgn_content = args.pgn
    
    print(f"{Fore.GREEN}Analyzing chess game...{Style.RESET_ALL}")
    
    # Create game analyzer
    analyzer = GameAnalyzer(
        stockfish_path=args.stockfish,
        analysis_depth=args.depth,
        threads=args.threads,
        hash_size=args.hash,
        num_cpus=args.cpus
    )
    
    # Run analysis
    analysis_result = analyzer.analyze_pgn(pgn_content)
    
    if not analysis_result:
        print(f"{Fore.RED}Analysis failed!{Style.RESET_ALL}")
        return
    
    # Print analysis results
    print_judgment_summary(analysis_result.get("judgments", []))
    print_analysis_table(analysis_result)
    print_feature_summary(analysis_result.get("features"))
    print_sharpness_summary(analysis_result.get("cumulative_sharpness"))
    print_king_safety_analysis(analysis_result)  
    
    # Save to file if requested
    if args.output:
        is_html = args.output.lower().endswith('.html')
        report = analyzer.format_analysis_report(analysis_result, html=is_html)
        
        with open(args.output, 'w') as f:
            f.write(report)
            
        print(f"{Fore.GREEN}Analysis report saved to {args.output}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()