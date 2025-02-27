# src/analysis/move_analyzer.py
import math
from typing import Optional, List, Tuple, Dict
import chess
from models.data_classes import Info
from models.enums import Judgment
import logging

logger = logging.getLogger('chess_analyzer')

class MateSequence:
    """Holds predefined mate sequence descriptions."""
    CREATED = ("Checkmate is now unavoidable", "Mate Created")
    DELAYED = ("Not the best checkmate sequence", "Mate Delayed")
    LOST = ("Lost forced checkmate sequence", "Mate Lost")

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
        try:
            return 2 / (1 + math.exp(-0.004 * cp)) - 1
        except Exception as e:
            logger.warning(f"Error calculating winning chances for cp={cp}: {e}")
            return 0.0  # Neutral value on error
    
    @staticmethod
    def is_piece_sacrifice(prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> bool:
        """
        Determine if a move was a piece sacrifice
        A sacrifice is defined as voluntarily giving up material without immediate recapture
        """
        if not prev_board or not curr_board or not move:
            return False
            
        try:
            # Check what piece was moved
            piece = prev_board.piece_at(move.from_square)
            if piece is None or piece.piece_type == chess.PAWN:
                return False
                
            # Check if a capture happened at the destination
            captured = prev_board.piece_at(move.to_square)
            
            # If piece moved to an empty square or captured a lower value piece, check if it's now under attack
            if captured is None or captured.piece_type < piece.piece_type:
                # Is the piece now under attack?
                attackers = curr_board.attackers(not prev_board.turn, move.to_square)
                if attackers:
                    # Is there adequate defense?
                    defenders = curr_board.attackers(prev_board.turn, move.to_square)
                    if not defenders or len(attackers) > len(defenders):
                        return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking for piece sacrifice: {e}")
            return False
    
    @staticmethod
    def is_only_good_move(prev_board: chess.Board, move: chess.Move, top_moves: List[str], prev_eval: Optional[Info] = None) -> bool:
        """
        Determine if the played move was the only good move in the position
        """
        if not top_moves or len(top_moves) < 2 or not prev_board or not move:
            return False
            
        try:
            # Convert move to UCI format for comparison
            move_uci = move.uci()
            
            # Check if played move is in top moves and is the best move
            if move_uci != top_moves[0]:
                return False
                
            # If we have evaluation data, check for significant eval drop
            if prev_eval and hasattr(prev_eval, 'multipv'):
                multipv_evals = prev_eval.multipv
                if len(multipv_evals) >= 2:
                    best_eval = multipv_evals[0].get('score', {}).get('cp', 0)
                    second_eval = multipv_evals[1].get('score', {}).get('cp', 0)
                    
                    # If best move leads to mate and second doesn't, it's the only good move
                    if multipv_evals[0].get('score', {}).get('mate') is not None and \
                       multipv_evals[1].get('score', {}).get('mate') is None:
                        return True
                    
                    # If both moves lead to mate, check if there's a significant difference
                    if multipv_evals[0].get('score', {}).get('mate') is not None and \
                       multipv_evals[1].get('score', {}).get('mate') is not None:
                        mate1 = multipv_evals[0]['score']['mate']
                        mate2 = multipv_evals[1]['score']['mate']
                        # If first move mates significantly faster
                        if mate1 > 0 and (mate2 < 0 or mate1 < mate2 - 2):
                            return True
                        return False
                    
                    # Check for significant centipawn difference (more than 1 pawn)
                    if abs(best_eval - second_eval) > 100:
                        return True
                    
                    return False
            
            # If we don't have eval data, use move legality and basic position assessment
            if len(top_moves) >= 2:
                # Get board after making the top move
                top_move_board = prev_board.copy()
                try:
                    top_move_board.push(chess.Move.from_uci(top_moves[0]))
                except ValueError:
                    return False  # Invalid move format
                
                # Count legal moves after best move
                top_move_legal_moves = list(top_move_board.legal_moves)
                
                # Get board after making the second best move
                second_move_board = prev_board.copy()
                try:
                    second_move = chess.Move.from_uci(top_moves[1])
                    second_move_board.push(second_move)
                    
                    # Count legal moves after second best move
                    second_move_legal_moves = list(second_move_board.legal_moves)
                    
                    # If there's a big difference in available moves or position characteristics,
                    # the first move might be the only good one
                    if len(top_move_legal_moves) > len(second_move_legal_moves) * 2:
                        return True
                        
                    # Check if second move leads to immediate tactical problems
                    if second_move_board.is_check() or \
                       second_move_board.is_checkmate() or \
                       second_move_board.is_stalemate():
                        return True
                        
                except Exception:
                    # If second move is invalid, the top move is definitely the only good one
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if move is the only good move: {e}")
            return False

    @staticmethod
    def move_complexity(prev_board: chess.Board, move: chess.Move) -> float:
        """
        Calculate the complexity/difficulty of finding a move
        
        Returns:
            Complexity score (0-1), where higher means more complex
        """
        if not prev_board or not move:
            return 0.0
            
        try:
            complexity = 0.0
            
            # Factor 1: Is it a capture move?
            if prev_board.is_capture(move):
                complexity += 0.1
                
            # Factor 2: Is it a check?
            after_board = prev_board.copy()
            after_board.push(move)
            if after_board.is_check():
                complexity += 0.2
                
            # Factor 3: Piece value (more valuable pieces are usually moved with more care)
            piece = prev_board.piece_at(move.from_square)
            if piece:
                if piece.piece_type == chess.QUEEN:
                    complexity += 0.3
                elif piece.piece_type in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    complexity += 0.2
                    
            # Factor 4: Is it a non-obvious move? (not to the edge, not to the center)
            destination_rank = chess.square_rank(move.to_square)
            destination_file = chess.square_file(move.to_square)
            if 2 <= destination_rank <= 5 and 2 <= destination_file <= 5:
                pass  # Center squares, no complexity added
            elif destination_rank in [0, 7] or destination_file in [0, 7]:
                complexity += 0.1  # Edge squares
            else:
                complexity += 0.15  # In-between
                
            # Factor 5: Is the piece already under attack?
            if prev_board.is_attacked_by(not prev_board.turn, move.from_square):
                complexity -= 0.2  # Reduce complexity as it's more obvious to move threatened pieces
                
            # Ensure complexity is within bounds
            return max(0.0, min(1.0, complexity))
        except Exception as e:
            logger.warning(f"Error calculating move complexity: {e}")
            return 0.0
        
    @staticmethod
    def _check_brilliant_conditions(
        prev: Info, 
        curr: Info, 
        prev_board: chess.Board,
        curr_board: chess.Board,
        move: chess.Move
    ) -> Tuple[bool, str]:
        """Helper method to check brilliant move conditions"""
        # Check if this creates a mate sequence with a piece sacrifice
        if curr.mate is not None and curr.mate > 0 and \
           MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
            return True, " | BRILLIANT: Creates mate sequence with piece sacrifice"
            
        # Check for BRILLIANT move (good piece sacrifice)
        if prev.cp is not None and curr.cp is not None and \
           MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
            # Ensure position is not already totally winning and not losing after
            if (prev.color and prev.cp < 300 and curr.cp > -100) or \
               (not prev.color and prev.cp > -300 and curr.cp < 100):
                return True, " | BRILLIANT: Good piece sacrifice from non-winning position"
                
        return False, ""
        
    @staticmethod
    def _check_great_conditions(
        prev: Info,
        curr: Info,
        prev_board: chess.Board,
        move: chess.Move,
        top_moves: List[str],
        move_difficulty: float = 0.0
    ) -> Tuple[bool, str]:
        """Helper method to check great move conditions"""
        if prev.cp is None or curr.cp is None:
            return False, ""
            
        is_only_good = MoveAnalyzer.is_only_good_move(prev_board, move, top_moves, prev)
        
        # Case 1: Difficult and only good move
        if move_difficulty > 0.3 and is_only_good:
            return True, f" | GREAT: Only good move in a complex position (complexity={move_difficulty:.2f})"
        
        # Case 2: Turns losing into equal/winning
        if (prev.color and prev.cp < -150 and curr.cp > -50) or \
           (not prev.color and prev.cp > 150 and curr.cp < 50):
            return True, f" | GREAT: Turns losing position into equal/winning (prev_cp={prev.cp}, curr_cp={curr.cp})"
        
        # Case 3: Turns equal into clearly winning
        if (prev.color and abs(prev.cp) < 50 and curr.cp > 200) or \
           (not prev.color and abs(prev.cp) < 50 and curr.cp < -200):
            return True, f" | GREAT: Turns equal position into clearly winning (prev_cp={prev.cp}, curr_cp={curr.cp})"
        
        # Case 4: Maintains winning advantage in complex position
        if move_difficulty > 0.4 and \
           ((prev.color and prev.cp > 200 and curr.cp > 180) or \
            (not prev.color and prev.cp < -200 and curr.cp < -180)):
            return True, f" | GREAT: Maintains winning advantage in complex position (complexity={move_difficulty:.2f})"
            
        return False, ""
        
    @staticmethod
    def analyze_move_with_top_moves(
        prev: Info, 
        curr: Info, 
        prev_board: chess.Board = None, 
        curr_board: chess.Board = None, 
        move: chess.Move = None,
        top_moves: List[str] = None,
        debug: bool = True
    ) -> Tuple[Judgment, str]:
        """
        Enhanced move analysis taking into account top engine moves
        
        Returns:
            Tuple of (Judgment, debug_reason) where debug_reason explains the judgment
        """
        # Check for missing inputs - return GOOD as a safe default
        if not prev or not curr:
            logger.warning("Missing evaluation info, defaulting to GOOD")
            return (Judgment.GOOD, "Missing evaluation info")
            
        try:
            # Check if move was forced (only one legal move)
            is_forced = False
            reason = ""
            
            if prev_board:
                legal_moves = list(prev_board.legal_moves)
                is_forced = len(legal_moves) == 1
                if is_forced:
                    reason = "FORCED MOVE: Only one legal move available"
            
            # Check if move is in top engine moves
            is_top_move = False
            if move and top_moves and move.uci() == top_moves[0]:
                is_top_move = True
                reason += " | TOP MOVE: Matches engine's first choice"
                
                # Check for brilliant/great moves if we have board information
                if prev_board and curr_board and move:
                    # Check for brilliant move conditions
                    is_brilliant, brilliant_reason = MoveAnalyzer._check_brilliant_conditions(
                        prev, curr, prev_board, curr_board, move
                    )
                    if is_brilliant:
                        return (Judgment.BRILLIANT, reason + brilliant_reason)
                        
                    # Check for great move conditions
                    move_difficulty = MoveAnalyzer.move_complexity(prev_board, move)
                    is_great, great_reason = MoveAnalyzer._check_great_conditions(
                        prev, curr, prev_board, move, top_moves, move_difficulty
                    )
                    if is_great:
                        return (Judgment.GREAT, reason + great_reason)
            else:
                if move and top_moves:
                    reason += f" | NOT TOP MOVE: Played {move.uci()}, top was {top_moves[0]}"
            
            # For forced moves that aren't brilliant or great, return GOOD
            if is_forced:
                return (Judgment.GOOD, reason)
            
            # Handle mate scores
            if curr.mate is not None and prev.cp is not None:
                if prev.cp > 400 and curr.mate < 0:
                    reason += f" | BLUNDER: Went from winning position to getting mated (prev_cp={prev.cp}, curr_mate={curr.mate})"
                    return (Judgment.BLUNDER, reason)
                if abs(prev.cp) < 500 and curr.mate < 0:
                    reason += f" | BLUNDER: Went from roughly equal to getting mated (prev_cp={prev.cp}, curr_mate={curr.mate})"
                    return (Judgment.BLUNDER, reason)
                    
            if prev.mate is not None and prev.mate > 0 and curr.cp is not None:
                reason += f" | BLUNDER: Lost a mate in {prev.mate} (now cp={curr.cp})"
                return (Judgment.BLUNDER, reason)
                
            # Handle missing cp values
            if prev.cp is None or curr.cp is None:
                logger.warning("Missing evaluation cp value(s), defaulting to GOOD")
                reason += " | GOOD: Missing evaluation data"
                return (Judgment.GOOD, reason)
                
            # Calculate winning chances difference - Lichess style
            prev_wc = MoveAnalyzer.winning_chances(prev.cp)
            curr_wc = MoveAnalyzer.winning_chances(curr.cp)
            delta = curr_wc - prev_wc
            
            # Invert delta if the move maker was Black
            if not prev.color:  # prev.color is True for White moves, False for Black moves
                delta = -delta
            
            reason += f" | EVAL: prev_cp={prev.cp}, curr_cp={curr.cp}, delta={delta:.2f}"
            
            # Original Lichess-style classification
            if delta <= -0.3:
                reason += " | BLUNDER: Major decrease in winning chances"
                return (Judgment.BLUNDER, reason)
            elif delta <= -0.2:
                reason += " | MISTAKE: Significant decrease in winning chances"
                return (Judgment.MISTAKE, reason)
            elif delta <= -0.1:
                reason += " | INACCURACY: Minor decrease in winning chances"
                return (Judgment.INACCURACY, reason)
            
            # For good moves, check for special cases again
            if is_top_move and prev_board and curr_board and move:
                # Check for BRILLIANT move (good piece sacrifice) again
                if MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
                    reason += " | BRILLIANT: Good piece sacrifice"
                    return (Judgment.BRILLIANT, reason)
                
                # Check for GREAT move again
                if top_moves:
                    is_only_good = MoveAnalyzer.is_only_good_move(prev_board, move, top_moves, prev)
                    move_difficulty = MoveAnalyzer.move_complexity(prev_board, move)
                    
                    reason += f" | ANALYSIS: only_good={is_only_good}, difficulty={move_difficulty:.2f}"
                    
                    # If it's a difficult move and the only good option, it's a GREAT move
                    if is_only_good and move_difficulty > 0.3:
                        reason += " | GREAT: Only good and difficult move"
                        return (Judgment.GREAT, reason)
            
            reason += " | GOOD: Solid move that maintains evaluation"
            return (Judgment.GOOD, reason)
            
        except Exception as e:
            logger.error(f"Error in move analysis: {e}")
            return (Judgment.GOOD, f"Error during analysis: {e}")  # Default to GOOD on error

    @staticmethod
    def analyze_move(prev: Info, curr: Info, prev_board: chess.Board = None, 
                     curr_board: chess.Board = None, move: chess.Move = None) -> Judgment:
        """
        Standard move analysis without top moves data (for backwards compatibility)
        """
        judgment, _ = MoveAnalyzer.analyze_move_with_top_moves(
            prev, curr, prev_board, curr_board, move, None
        )
        return judgment