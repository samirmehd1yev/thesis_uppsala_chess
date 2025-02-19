"""
Chess Move Analyzer
===================

This module implements a chess move analyzer using the Stockfish engine.
It is inspired by Lichess's move analysis code and provides functionality to:
  - Evaluate positions using Stockfish.
  - Generate advice (e.g., Inaccuracy, Mistake, Blunder) based on evaluation differences.
  - Handle mate evaluations and convert them to advice.
  - Parse a PGN-like moves string into board positions.
  - Determine game phase transitions (opening, middlegame, endgame).
  - Format and print a detailed analysis summary.

Dependencies:
  - python-chess
  - stockfish (python wrapper)
  - tqdm (for progress visualization)
"""

import chess
import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from stockfish import Stockfish
import math
from dataclasses import dataclass
from enum import Enum


class Judgment(Enum):
    """
    Enumeration of move quality judgments.
    """
    INACCURACY = "Inaccuracy"
    MISTAKE = "Mistake"
    BLUNDER = "Blunder"


@dataclass
class Info:
    """
    Holds evaluation information for a given move/position.
    
    Attributes:
        ply (int): The ply number (half-move count) in the game.
        eval (dict): The evaluation dictionary returned by Stockfish.
        variation (List[str], optional): The list of best moves (principal variation).
    """
    ply: int
    eval: dict  # Stockfish evaluation, expected to contain keys "type" and "value"
    variation: List[str] = None

    @property
    def color(self) -> bool:
        """
        Determine the side to move.
        
        Returns:
            bool: True if White is to move (even ply), False if Black.
        """
        return self.ply % 2 == 0

    @property
    def cp(self) -> Optional[int]:
        """
        Get the centipawn evaluation if available.
        
        Returns:
            Optional[int]: The centipawn value, or None if the evaluation is a mate.
        """
        return self.eval["value"] if self.eval["type"] == "cp" else None

    @property
    def mate(self) -> Optional[int]:
        """
        Get the mate evaluation if available.
        
        Returns:
            Optional[int]: The mate value, or None if the evaluation is given in centipawns.
        """
        return self.eval["value"] if self.eval["type"] == "mate" else None

    def eval_comment(self) -> Optional[str]:
        """
        Generate a human-readable evaluation comment.
        
        Returns:
            Optional[str]: A string representing the evaluation (e.g., "#-2" for mate in 2,
                           or "+0.5" for centipawn values), or None if not available.
        """
        if self.mate is not None:
            return f"#{self.mate}"
        elif self.cp is not None:
            return f"{self.cp/100:+.1f}"
        return None


class MateSequence:
    """
    Holds predefined mate sequence descriptions.
    
    Attributes:
        CREATED: Tuple describing a newly created forced mate.
        DELAYED: Tuple describing a delayed mate sequence.
        LOST: Tuple describing a lost forced mate sequence.
    """
    CREATED = ("Checkmate is now unavoidable", "Mate Created")
    DELAYED = ("Not the best checkmate sequence", "Mate Delayed")
    LOST = ("Lost forced checkmate sequence", "Mate Lost")


class Advice:
    """
    Base class for move advice.
    
    Attributes:
        judgment (Judgment): The quality judgment for the move.
        info (Info): Evaluation info after the move.
        prev (Info): Evaluation info before the move.
    """
    def __init__(self, judgment: Judgment, info: Info, prev: Info):
        self.judgment = judgment
        self.info = info
        self.prev = prev

    def make_comment(self, with_eval: bool = True, with_best_move: bool = True) -> str:
        """
        Generate a textual comment summarizing the move quality.
        
        Args:
            with_eval (bool): If True, include evaluation change.
            with_best_move (bool): If True, include best move suggestion.
            
        Returns:
            str: The formatted comment.
        """
        comment = ""
        if with_eval:
            prev_eval = self.prev.eval_comment()
            curr_eval = self.info.eval_comment()
            if prev_eval and curr_eval:
                comment += f"({prev_eval} → {curr_eval}) "
        
        comment += f"{self.judgment.value}."
        
        if with_best_move and self.info.variation and len(self.info.variation) > 0:
            comment += f" {self.info.variation[0]} was best."
            
        return comment


class CpAdvice(Advice):
    """
    Generates advice based on centipawn evaluations.
    """
    @staticmethod
    def winning_chances(cp: int) -> float:
        """
        Convert centipawn evaluation into winning chances using Lichess' formula.
        
        Args:
            cp (int): Centipawn evaluation.
            
        Returns:
            float: Winning chances (range -1 to 1).
        """
        return 2 / (1 + math.exp(-0.004 * cp)) - 1

    @classmethod
    def from_info(cls, prev: Info, info: Info) -> Optional['CpAdvice']:
        """
        Create a CpAdvice instance from two consecutive evaluations.
        
        It handles mate evaluations specially and then compares centipawn values.
        
        Args:
            prev (Info): Evaluation info before the move.
            info (Info): Evaluation info after the move.
            
        Returns:
            Optional[CpAdvice]: An instance of CpAdvice if a mistake is detected, else None.
        """
        # Handle mate scores
        if info.mate is not None:
            # If previous position was winning and now it's a forced mate against
            if prev.cp is not None and prev.cp > 400 and info.mate < 0:
                return cls(Judgment.BLUNDER, info, prev)
            # If position suddenly becomes mate (and wasn't already winning)
            if prev.cp is not None and abs(prev.cp) < 500 and info.mate < 0:
                print(f"prev.cp: {prev.cp}, info.mate: {info.mate}")
                return cls(Judgment.BLUNDER, info, prev)
            return None

        if prev.mate is not None:
            # If we had mate and lost it
            if prev.mate > 0 and info.cp is not None:
                return cls(Judgment.BLUNDER, info, prev)
            return None

        # Normal centipawn evaluation
        if prev.cp is None or info.cp is None:
            return None
                
        prev_wc = cls.winning_chances(prev.cp)
        curr_wc = cls.winning_chances(info.cp)
            
        # Calculate winning chances delta
        delta = curr_wc - prev_wc
        if info.color is False:  # Invert delta for Black moves (since Info.color is True for White)
            delta = -delta
                
        # Determine judgment based on winning chances delta
        judgment = None
        if delta <= -0.3:
            judgment = Judgment.BLUNDER
        elif delta <= -0.2:
            judgment = Judgment.MISTAKE
        elif delta <= -0.1:
            judgment = Judgment.INACCURACY
                
        if judgment is None:
            return None
                
        return cls(judgment, info, prev)


class MateAdvice(Advice):
    """
    Generates advice when a mate sequence is involved.
    """
    def __init__(self, sequence: Tuple[str, str], judgment: Judgment, info: Info, prev: Info):
        """
        Initialize MateAdvice.
        
        Args:
            sequence (Tuple[str, str]): A tuple describing the mate sequence.
            judgment (Judgment): The quality judgment.
            info (Info): Evaluation info after the move.
            prev (Info): Evaluation info before the move.
        """
        super().__init__(judgment, info, prev)
        self.sequence = sequence

    @classmethod
    def from_info(cls, prev: Info, info: Info) -> Optional['MateAdvice']:
        """
        Create MateAdvice from two consecutive evaluation infos.
        
        It converts evaluations from the perspective of the side to move by inverting
        the scores for Black (using `not info.color` since Info.color is True for White).
        
        Args:
            prev (Info): Evaluation info before the move.
            info (Info): Evaluation info after the move.
            
        Returns:
            Optional[MateAdvice]: An instance if mate-related advice applies, else None.
        """
        def invert_if_black(value: Optional[int], is_black: bool) -> Optional[int]:
            """
            Invert evaluation values for Black's perspective.
            
            Args:
                value (Optional[int]): The evaluation value.
                is_black (bool): True if the position is from Black's perspective.
                
            Returns:
                Optional[int]: The inverted value if applicable, else the original.
            """
            return -value if is_black and value is not None else value

        # Invert values if side to move is Black (i.e., not info.color)
        prev_mate = invert_if_black(prev.mate, not info.color)
        curr_mate = invert_if_black(info.mate, not info.color)
        prev_cp   = invert_if_black(prev.cp, not info.color)
        curr_cp   = invert_if_black(info.cp, not info.color)

        sequence = None
        if prev_mate is None and curr_mate is not None and curr_mate < 0:
            sequence = MateSequence.CREATED
        elif prev_mate is not None and curr_mate is None and prev_mate > 0:
            sequence = MateSequence.LOST
        elif (prev_mate is not None and curr_mate is not None and 
              prev_mate > 0 and curr_mate < 0):
            sequence = MateSequence.LOST

        if sequence is None:
            return None

        # Determine judgment based on thresholds
        judgment = None
        if sequence == MateSequence.CREATED:
            if prev_cp is not None and prev_cp < -999:
                judgment = Judgment.INACCURACY
            elif prev_cp is not None and prev_cp < -700:
                judgment = Judgment.MISTAKE
            else:
                judgment = Judgment.BLUNDER
        elif sequence == MateSequence.LOST:
            if curr_cp is not None and curr_cp > 999:
                judgment = Judgment.INACCURACY
            elif curr_cp is not None and curr_cp > 700:
                judgment = Judgment.MISTAKE
            else:
                judgment = Judgment.BLUNDER

        if judgment is None:
            return None

        return cls(sequence, judgment, info, prev)


class GameAnalyzer:
    """
    Analyzes chess games using the Stockfish engine.
    
    This class provides methods to:
      - Parse a PGN-like moves string into board positions.
      - Evaluate positions and moves using Stockfish.
      - Determine game phase transitions.
      - Generate move-by-move advice.
      - Print a detailed analysis summary.
    """
    def __init__(self, stockfish_path="stockfish", depth=16, threads=4, debug=False):
        """
        Initialize the GameAnalyzer with Stockfish engine support.
        
        Args:
            stockfish_path (str): Path to the Stockfish binary.
            depth (int): Engine search depth.
            threads (int): Number of threads for Stockfish.
            debug (bool): If True, print debug information.
        """
        self.debug = debug
        self.stockfish = Stockfish(
            path=stockfish_path, 
            depth=depth,
            parameters={
                "Threads": threads,
                "Hash": 128,
                "MultiPV": 1,
                "Minimum Thinking Time": 20
            }
        )
        self.depth = depth
        # Bit masks for board regions
        self.FIRST_RANK_MASK = 0xFF
        self.LAST_RANK_MASK = 0xFF << 56
        self.SMALL_SQUARE = 0x0303

    def _debug_print(self, *args, **kwargs):
        """
        Helper method for printing debug information.
        """
        if self.debug:
            print(*args, **kwargs)

    def get_positions_from_moves(self, moves_str: str) -> list[chess.Board]:
        """
        Convert a string of chess moves into a list of board positions.
        
        The input string can include move numbers, comments, or results.
        
        Args:
            moves_str (str): The string of chess moves (e.g., PGN format).
            
        Returns:
            list[chess.Board]: A list of board positions starting with the initial board.
        """
        board = chess.Board()
        positions = [board.copy()]
        
        moves = []
        # Split the moves by move numbers (e.g., "1.", "2.", etc.)
        for move_pair in re.split(r'\d+\.', moves_str):
            if move_pair.strip():
                # Remove any comments (within {}) and game results
                clean_pair = ' '.join(word for word in move_pair.split() 
                                      if not (word.startswith('{') or word.endswith('}') 
                                              or word in ['1-0', '0-1', '1/2-1/2']))
                moves.extend(clean_pair.strip().split())

        for move in moves:
            try:
                board.push_san(move)
                positions.append(board.copy())
            except Exception as e:
                self._debug_print(f"Error processing move {move}: {e}")
                continue
                
        return positions

    def count_majors_and_minors(self, board: chess.Board) -> int:
        """
        Count the number of major and minor pieces (excluding kings and pawns).
        
        Args:
            board (chess.Board): The board position.
            
        Returns:
            int: The count of major and minor pieces.
        """
        pieces = board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.KNIGHT, chess.BLACK)
        pieces |= board.pieces(chess.BISHOP, chess.WHITE) | board.pieces(chess.BISHOP, chess.BLACK)
        pieces |= board.pieces(chess.ROOK, chess.WHITE) | board.pieces(chess.ROOK, chess.BLACK)
        pieces |= board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.QUEEN, chess.BLACK)
        return bin(pieces).count('1')

    def is_backrank_sparse(self, board: chess.Board) -> bool:
        """
        Check if the back ranks are sparsely populated (less than 4 pieces).
        
        Args:
            board (chess.Board): The board position.
            
        Returns:
            bool: True if either back rank is sparse, else False.
        """
        white_back = bin(board.occupied_co[chess.WHITE] & self.FIRST_RANK_MASK).count('1')
        black_back = bin(board.occupied_co[chess.BLACK] & self.LAST_RANK_MASK).count('1')
        return white_back < 4 or black_back < 4

    def calculate_mixedness(self, board: chess.Board) -> int:
        """
        Calculate a "mixedness" score based on the distribution of pieces.
        
        The board is divided into overlapping 2x2 regions. Each region is scored based
        on how mixed the pieces are, which can help in determining game phases.
        
        Args:
            board (chess.Board): The board position.
            
        Returns:
            int: The total mixedness score.
        """
        total_score = 0
        for y in range(7):
            for x in range(7):
                region = self.SMALL_SQUARE << (x + 8 * y)
                white_count = bin(board.occupied_co[chess.WHITE] & region).count('1')
                black_count = bin(board.occupied_co[chess.BLACK] & region).count('1')
                total_score += self.score_region(y + 1, white_count, black_count)
        return total_score

    def score_region(self, y: int, white_count: int, black_count: int) -> int:
        """
        Score a 2x2 board region based on piece distribution.
        
        The score depends on the counts of White and Black pieces and the region's rank (y).
        
        Args:
            y (int): The rank (1-indexed) of the region.
            white_count (int): Number of White pieces in the region.
            black_count (int): Number of Black pieces in the region.
            
        Returns:
            int: The score for that region.
        """
        if (white_count, black_count) == (0, 0):
            return 0
        elif (white_count, black_count) == (1, 0):
            return 1 + (8 - y)
        elif (white_count, black_count) == (2, 0):
            return 2 + (y - 2) if y > 2 else 0
        elif (white_count, black_count) == (3, 0):
            return 3 + (y - 1) if y > 1 else 0
        elif (white_count, black_count) == (4, 0):
            return 3 + (y - 1) if y > 1 else 0
        elif (white_count, black_count) == (0, 1):
            return 1 + y
        elif (white_count, black_count) == (1, 1):
            return 5 + abs(3 - y)
        elif (white_count, black_count) == (2, 1):
            return 4 + y
        elif (white_count, black_count) == (3, 1):
            return 5 + y
        elif (white_count, black_count) == (0, 2):
            return 2 + (6 - y) if y < 6 else 0
        elif (white_count, black_count) == (1, 2):
            return 4 + (6 - y)
        elif (white_count, black_count) == (2, 2):
            return 7
        elif (white_count, black_count) == (0, 3):
            return 3 + (7 - y) if y < 7 else 0
        elif (white_count, black_count) == (1, 3):
            return 5 + (6 - y)
        elif (white_count, black_count) == (0, 4):
            return 3 + (7 - y) if y < 7 else 0
        else:
            return 0

    def find_phase_transitions(self, positions: list[chess.Board]) -> tuple[int, int]:
        """
        Determine the game phase transitions between opening, middlegame, and endgame.
        
        Uses heuristics based on:
          - Count of major/minor pieces.
          - Sparsity of back ranks.
          - Mixedness score.
        
        Args:
            positions (list[chess.Board]): List of board positions from the game.
            
        Returns:
            tuple[int, int]: A tuple (middlegame_start, endgame_start) indicating the move numbers
                             where the middlegame and endgame start, respectively.
        """
        middlegame_start, endgame_start = 0, 0
        
        for i, board in enumerate(positions[1:], start=1):
            move_number = (i + 1) // 2
            majors_minors = self.count_majors_and_minors(board)
            backrank_sparse = self.is_backrank_sparse(board)
            mixedness = self.calculate_mixedness(board)
            
            if middlegame_start == 0 and (majors_minors <= 10 or backrank_sparse or mixedness > 150):
                middlegame_start = move_number + 1
            
            if middlegame_start > 0 and endgame_start == 0 and majors_minors <= 6:
                endgame_start = move_number + 1
                
        if endgame_start == 0:
            endgame_start = middlegame_start + 1
            
        return middlegame_start, endgame_start

    def get_best_moves(self, fen: str, num_moves: int = 1) -> List[str]:
        """
        Retrieve the best moves for a given position.
        
        Args:
            fen (str): The FEN representation of the board position.
            num_moves (int): Number of best moves to retrieve.
            
        Returns:
            List[str]: A list of best moves in SAN format.
        """
        self.stockfish.set_fen_position(fen)
        top_moves = self.stockfish.get_top_moves(num_moves)
        return [move["Move"] for move in top_moves]

    def evaluate_position(self, board: chess.Board, ply: int) -> Info:
        """
        Evaluate a board position using Stockfish.
        
        Args:
            board (chess.Board): The board position to evaluate.
            ply (int): The current ply number.
            
        Returns:
            Info: An Info object containing the evaluation and best move variation.
        """
        self.stockfish.set_fen_position(board.fen())
        eval_dict = self.stockfish.get_evaluation()
        variations = self.get_best_moves(board.fen(), 1)
        
        return Info(
            ply=ply,
            eval=eval_dict,
            variation=variations
        )

    def evaluate_moves(self, positions: List[chess.Board]) -> List[Dict]:
        """
        Evaluate each move of the game and generate detailed move analysis.
        
        For each move, the following are determined:
          - Move number and side.
          - Move in SAN.
          - Evaluation before and after the move.
          - Quality label (Good, Inaccuracy, Mistake, or Blunder).
          - Best move suggestion for the position.
        
        Args:
            positions (List[chess.Board]): List of board positions.
            
        Returns:
            List[Dict]: A list of dictionaries with analysis information for each move.
        """
        evaluations = []
        
        if self.debug:
            print("\n" + "=" * 80)
            print("ANALYZING GAME MOVES")
            print("=" * 80)
            print(f"\nEngine Configuration:")
            print(f"- Depth: {self.depth}")
            print(f"- Stockfish Version: {self.stockfish.get_stockfish_major_version()}")
            print("-" * 80 + "\n")
            
            # Create progress bar only in debug mode.
            pbar = tqdm(total=len(positions) - 1, desc="Analyzing moves", ncols=80)
        
        prev_info = None
        for i, board in enumerate(positions[:-1]):
            next_board = positions[i + 1]
            # Determine the move made in SAN by comparing board positions.
            move = board.san(next_board.move_stack[-1])
            
            # Evaluate the position after the move.
            info = self.evaluate_position(next_board, i)
            
            # Generate advice based on the difference between evaluations.
            advice = None
            if prev_info is not None:
                advice = CpAdvice.from_info(prev_info, info) or MateAdvice.from_info(prev_info, info)
            
            # Calculate evaluation change for display purposes.
            eval_change = 0.0
            if prev_info is not None and prev_info.cp is not None and info.cp is not None:
                eval_change = (info.cp - prev_info.cp) / 100
            
            # Retrieve the best move from the board BEFORE the move.
            best_moves = self.get_best_moves(board.fen(), 1)
            best_move = best_moves[0] if best_moves else None

            entry = {
                "move_number": (i // 2) + 1,
                "color": "White" if i % 2 == 0 else "Black",
                "move": move,
                "evaluation": info.eval_comment(),
                "change": f"{eval_change:+.1f}",
                "label": advice.judgment.value if advice else "Good",
                "best_move": best_move
            }
            evaluations.append(entry)
            
            prev_info = info
            
            if self.debug:
                pbar.update(1)
                status = f"Analyzing {entry['color']} move {entry['move']}"
                if entry['label'] != 'Good':
                    status += f" ({entry['label']})"
                pbar.set_description(status)
        
        if self.debug:
            pbar.close()
            
        return evaluations

    def print_analysis(self, evaluations: List[Dict], phases: Tuple[int, int]):
        """
        Print a formatted analysis summary including detailed move analysis,
        player statistics, and game phase transitions.
        
        This method only prints output if debug mode is enabled.
        
        Args:
            evaluations (List[Dict]): List of move analysis dictionaries.
            phases (Tuple[int, int]): Tuple containing the start move of the middlegame and endgame.
        """
        if not self.debug:
            return
            
        print("\n" + "=" * 80)
        print("MOVE ANALYSIS")
        print("=" * 80)
        
        # Print detailed move analysis in table format.
        print("\nDetailed Move Analysis:")
        print("-" * 100)
        header = f"{'Move':^5} | {'Color':^6} | {'Move':^8} | {'Eval':^7} | {'Change':^7} | {'Quality':^10} | {'Best Move':^15}"
        print(header)
        print("-" * 100)
        
        for entry in evaluations:
            best_move = entry.get('best_move', '')
            if best_move and entry['label'] != 'Good':
                best_move = f"→ {best_move}"
            else:
                best_move = ""
                
            print(f"{entry['move_number']:^5} | {entry['color']:^6} | {entry['move']:^8} | "
                  f"{entry['evaluation']:^7} | {entry['change']:^7} | {entry['label']:^10} | {best_move:^15}")
            
        # Calculate player statistics.
        stats = {
            'White': {'Inaccuracy': 0, 'Mistake': 0, 'Blunder': 0, 'Good': 0},
            'Black': {'Inaccuracy': 0, 'Mistake': 0, 'Blunder': 0, 'Good': 0}
        }
        
        for entry in evaluations:
            stats[entry['color']][entry['label']] += 1
        
        # Print summary statistics.
        print("\n" + "=" * 80)
        print("GAME SUMMARY")
        print("=" * 80)
        
        def print_player_stats(color, player_stats):
            total_moves = sum(player_stats.values())
            accuracy = (player_stats['Good'] / total_moves * 100) if total_moves > 0 else 0
            
            print(f"\n{color} player:")
            print("-" * 40)
            print(f"Total Moves   : {total_moves}")
            print(f"Good Moves    : {player_stats['Good']} ({accuracy:.1f}%)")
            print(f"Inaccuracies  : {player_stats['Inaccuracy']}")
            print(f"Mistakes      : {player_stats['Mistake']}")
            print(f"Blunders      : {player_stats['Blunder']}")
            
        for color in ['White', 'Black']:
            print_player_stats(color, stats[color])
        
        # Print game phase transitions.
        middlegame_start, endgame_start = phases
        print("\n" + "=" * 80)
        print("GAME PHASES")
        print("=" * 80)
        total_moves = len(evaluations) // 2
        
        def format_phase(name, start, end=None):
            if end:
                moves = (end - start) + 1  # Include both start and end positions.
                percentage = (moves / total_moves * 100) if total_moves > 0 else 0
                return f"{name:10s}: moves {start:2d}-{end:<2d} ({moves:2d} moves, {percentage:4.1f}%)"
            else:
                moves = total_moves - start + 1  # Include the start position.
                percentage = (moves / total_moves * 100) if total_moves > 0 else 0
                return f"{name:10s}: moves {start:2d}-end ({moves:2d} moves, {percentage:4.1f}%)"

        print(format_phase("Opening", 1, middlegame_start - 1))
        print(format_phase("Middlegame", middlegame_start, endgame_start - 1))
        print(format_phase("Endgame", endgame_start))
        print("=" * 80 + "\n")

    def close_engine(self):
        """
        Closes the Stockfish engine.
        
        If a proper shutdown method is available in the Stockfish wrapper,
        it should be used instead of deleting the instance.
        """
        if self.debug:
            print("\nClosing Stockfish engine...")
        del self.stockfish
        if self.debug:
            print("Engine closed successfully.")
