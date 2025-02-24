# src/analysis/move_analyzer.py
from typing import Optional, Dict
import chess
import math
from models.data_classes import Info
from models.enums import Judgment

class MoveAnalyzer:
    # Use chess library's piece values
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King value is not used in evaluation
    }

    @staticmethod
    def winning_chances(cp: int) -> float:
        """Convert centipawn evaluation into winning chances using Lichess' formula"""
        return 2 / (1 + math.exp(-0.004 * cp)) - 1

    @staticmethod 
    def is_piece_sacrifice(board: chess.Board, move: chess.Move) -> bool:
        """
        More efficient piece sacrifice detection using bitboards
        """
        from_piece = board.piece_type_at(move.from_square)
        if not from_piece or from_piece == chess.PAWN or from_piece == chess.KING:
            return False

        from_value = MoveAnalyzer.PIECE_VALUES[from_piece]
        
        # Get captured piece value
        to_piece = board.piece_type_at(move.to_square)
        to_value = MoveAnalyzer.PIECE_VALUES[to_piece] if to_piece else 0
        
        # Make a copy and make the move
        board_copy = board.copy(stack=False)
        board_copy.push(move)
        
        # Check if the moved piece is under attack
        attackers = board_copy.attackers(not board.turn, move.to_square)
        if attackers:
            # Get value of smallest attacker
            min_attacker_value = min(
                MoveAnalyzer.PIECE_VALUES[board_copy.piece_type_at(sq)]
                for sq in attackers
            )
            # Consider it a sacrifice if we give up more material than we gain
            # and can be captured by a cheaper piece
            return from_value - to_value > 1 and min_attacker_value < from_value

        return False

    @staticmethod
    def position_evaluation_category(cp: Optional[int]) -> str:
        """Categorize position based on evaluation thresholds"""
        if cp is None:
            return "unknown"
        if cp > 200:
            return "winning"
        if cp < -200:
            return "losing"
        return "equal"

    @staticmethod
    def analyze_move(prev: Info, curr: Info, board: chess.Board, move: chess.Move) -> Optional[Judgment]:
        """Analyze a move with efficient position evaluation"""
        # Handle forced mates
        if curr.mate is not None:
            if prev.cp is not None and prev.cp > -200:  # Not already losing
                if curr.mate > 0:
                    return Judgment.GREAT  # Found mate
                if curr.mate < 0:
                    return Judgment.BLUNDER  # Got mated
            return None

        if prev.cp is None or curr.cp is None:
            return None

        # Calculate evaluation change
        eval_change = curr.cp - prev.cp
        # Invert for Black's moves
        if not board.turn:
            eval_change = -eval_change

        # Quick check for major mistakes
        if eval_change <= -300:
            return Judgment.BLUNDER
        elif eval_change <= -200:
            return Judgment.MISTAKE
        elif eval_change <= -100:
            return Judgment.INACCURACY

        # Get position categories
        prev_pos = MoveAnalyzer.position_evaluation_category(prev.cp)
        curr_pos = MoveAnalyzer.position_evaluation_category(curr.cp)

        # Check for Brilliant move - must be a piece sacrifice
        if MoveAnalyzer.is_piece_sacrifice(board, move):
            # Position shouldn't be bad after sacrifice
            if curr.cp > -200:
                # Shouldn't already be completely winning
                if prev.cp < 500:
                    return Judgment.BRILLIANT

        # Check for Great move criteria
        # 1. Turning losing position into equal/winning
        if prev_pos == "losing" and curr_pos in ["equal", "winning"]:
            return Judgment.GREAT
            
        # 2. Turning equal position into winning
        if prev_pos == "equal" and curr_pos == "winning":
            return Judgment.GREAT

        # 3. Only good move among top candidates
        if prev.variation and len(prev.variation) >= 2:
            best_move = prev.variation[0].get("Move")
            second_best_eval = prev.variation[1].get("Centipawn")
            
            if best_move and second_best_eval is not None:
                if move.uci() == best_move:
                    # Significant difference between best and second best
                    if abs(second_best_eval - prev.cp) > 150:
                        return Judgment.GREAT

        # Default to None (Good move) if no other classification applies
        return None