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
    
    # Lichess exact multiplier
    WINNING_CHANCES_MULTIPLIER = -0.00368208
    
    @staticmethod
    def raw_winning_chances(cp: int) -> float:
        """
        Raw winning chances calculation using Lichess' exact formula.
        https://github.com/lichess-org/lila/pull/11148
        """
        try:
            return 2 / (1 + math.exp(MoveAnalyzer.WINNING_CHANCES_MULTIPLIER * cp)) - 1
        except Exception as e:
            logger.warning(f"Error calculating raw winning chances for cp={cp}: {e}")
            return 0.0  # Neutral value on error
    
    @staticmethod
    def cp_winning_chances(cp: int) -> float:
        """
        Winning chances for centipawn evaluation with capped values.
        """
        capped_cp = min(max(-1000, cp), 1000)
        return MoveAnalyzer.raw_winning_chances(capped_cp)
    
    @staticmethod
    def mate_winning_chances(mate: int) -> float:
        """
        Calculate winning chances for a mate score.
        
        Args:
            mate: Number of moves to mate (positive if winning, negative if losing)
        
        Returns:
            Float between -1 and 1 representing winning chances
        """
        cp = (21 - min(10, abs(mate))) * 100
        signed_cp = cp * (1 if mate > 0 else -1)
        return MoveAnalyzer.raw_winning_chances(signed_cp)
    
    @staticmethod
    def eval_winning_chances(score: Dict[str, Optional[int]]) -> float:
        """
        Calculate winning chances from an evaluation score that could be either cp or mate.
        
        Args:
            score: Dictionary with 'cp' or 'mate' key
        
        Returns:
            Float between -1 and 1 representing winning chances
        """
        if 'mate' in score and score['mate'] is not None:
            return MoveAnalyzer.mate_winning_chances(score['mate'])
        elif 'cp' in score and score['cp'] is not None:
            return MoveAnalyzer.cp_winning_chances(score['cp'])
        
        logger.warning(f"Invalid score format: {score}")
        return 0.0
    
    @staticmethod
    def winning_chances(cp: int) -> float:
        """
        Convert centipawn evaluation into winning chances using Lichess' formula.
        For backward compatibility - delegates to cp_winning_chances.
        """
        return MoveAnalyzer.cp_winning_chances(cp)
    
    @staticmethod
    def to_pov(color: str, diff: float) -> float:
        """
        Adjust a value based on the player's point of view
        
        Args:
            color: 'white' or 'black'
            diff: The value to adjust
            
        Returns:
            Adjusted value based on color's perspective
        """
        return diff if color == 'white' else -diff
    
    @staticmethod
    def pov_chances(color: str, eval_score: Dict[str, Optional[int]]) -> float:
        """
        Calculate winning chances from a player's point of view
        
        Args:
            color: 'white' or 'black'
            eval_score: Evaluation score with 'cp' or 'mate' key
            
        Returns:
            Winning chances from the perspective of the given color
        """
        return MoveAnalyzer.to_pov(color, MoveAnalyzer.eval_winning_chances(eval_score))
    
    @staticmethod
    def pov_diff(color: str, eval1: Dict[str, Optional[int]], eval2: Dict[str, Optional[int]]) -> float:
        """
        Compute the difference in winning chances between two evaluations from a player's perspective
        
        Args:
            color: 'white' or 'black'
            eval1: First evaluation
            eval2: Second evaluation
            
        Returns:
            Difference in winning chances, normalized to -1 to 1 range
        """
        return (MoveAnalyzer.pov_chances(color, eval1) - MoveAnalyzer.pov_chances(color, eval2)) / 2
    
    @staticmethod
    def are_similar_evals(color: str, best_eval: Dict[str, Optional[int]], second_best_eval: Dict[str, Optional[int]]) -> bool:
        """
        Check if two evaluations are similar enough (used for puzzle validation)
        
        Args:
            color: 'white' or 'black'
            best_eval: Best move evaluation
            second_best_eval: Second best move evaluation
            
        Returns:
            True if evaluations are similar enough
        """
        return MoveAnalyzer.pov_diff(color, best_eval, second_best_eval) < 0.14
    
    @staticmethod
    def has_multiple_solutions(color: str, best_eval: Dict[str, Optional[int]], second_best_eval: Dict[str, Optional[int]]) -> bool:
        """
        Check if a position has multiple good solutions (used for puzzle validation)
        
        Args:
            color: 'white' or 'black'
            best_eval: Best move evaluation
            second_best_eval: Second best move evaluation
            
        Returns:
            True if multiple good solutions exist
        """
        # if second best eval equivalent of cp is >= 200
        return (MoveAnalyzer.pov_chances(color, second_best_eval) >= 0.3524 or
                MoveAnalyzer.are_similar_evals(color, best_eval, second_best_eval))
    
    @staticmethod
    def is_piece_sacrifice(prev_board: chess.Board, curr_board: chess.Board, move: chess.Move) -> bool:
        """
        Determine if a move was a piece sacrifice
        A sacrifice is defined as voluntarily giving up material without immediate equal or better compensation.
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
            piece_value = MoveAnalyzer.PIECE_VALUES.get(piece.piece_type, 0)
            captured_value = MoveAnalyzer.PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
            
            # If we captured a piece of equal or higher value, it's not a sacrifice
            if captured and captured_value >= piece_value:
                return False
            
            # Check if the piece is now under attack
            attackers = curr_board.attackers(not prev_board.turn, move.to_square)
            if not attackers:
                return False  # No attackers means no sacrifice
            
            # Check for legal attackers
            legal_attackers = []
            for attacker_square in attackers:
                attack_move = chess.Move(attacker_square, move.to_square)
                if curr_board.is_legal(attack_move):
                    legal_attackers.append(attacker_square)
            
            if not legal_attackers:
                return False  # No legal attackers means no sacrifice
            
            # Check for defenders
            defenders = curr_board.attackers(prev_board.turn, move.to_square)
            
            # Basic sacrifice check: piece is undefended or underdefended
            if not defenders or len(defenders) < len(legal_attackers):
                return True
            
            # Check the material value exchange
            min_attacker_value = float('inf')
            for attacker_square in legal_attackers:
                attacker_piece = curr_board.piece_at(attacker_square)
                attacker_value = MoveAnalyzer.PIECE_VALUES.get(attacker_piece.piece_type, 0)
                min_attacker_value = min(min_attacker_value, attacker_value)
            
            # If our piece is more valuable than the cheapest attacker, it's potentially a sacrifice
            if piece_value > min_attacker_value:
                # Check if we have a defender of equal or lesser value than the cheapest attacker
                has_efficient_defender = False
                for defender_square in defenders:
                    defender_piece = curr_board.piece_at(defender_square)
                    defender_value = MoveAnalyzer.PIECE_VALUES.get(defender_piece.piece_type, 0)
                    if defender_value <= min_attacker_value:
                        has_efficient_defender = True
                        break
                
                # If we don't have an efficient defender, it's a sacrifice
                if not has_efficient_defender:
                    return True
            
            return False  # Not a sacrifice
            
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
            # print(f"DEBUG: came here 1 {move_uci} = {top_moves} prev_eval: {prev_eval}")
            
            # If we have evaluation data, check for significant eval drop
            if prev_eval and hasattr(prev_eval, 'multipv') and prev_eval.multipv and len(prev_eval.multipv) >= 2:
                # print(f"DEBUG: came here 2")
                # Get evaluations for the best and second-best moves
                best_move_data = prev_eval.multipv[0]
                second_move_data = prev_eval.multipv[1]
                
                # If best move leads to mate and second doesn't, it's the only good move
                if 'mate' in best_move_data['score'] and 'cp' in second_move_data['score']:
                    return True
                
                # If second move leads to mate against us, first move is the only good one
                if 'cp' in best_move_data['score'] and 'mate' in second_move_data['score']:
                    if second_move_data['score']['mate'] < 0:
                        return True
                
                # Check for significant centipawn difference (more than 1 pawn)
                if 'cp' in best_move_data['score'] and 'cp' in second_move_data['score']:
                    best_eval = best_move_data['score']['cp']
                    second_eval = second_move_data['score']['cp']
                    
                    # If the difference is more than 1.5 pawns, it's the only good move
                    if abs(best_eval - second_eval) > 150:
                        return True
                    
                    # If position is critical (close to 0), even a smaller difference matters
                    if abs(best_eval) < 100 and abs(best_eval - second_eval) > 80:
                        return True
            
            # Even with top moves, we can check if they're all similar
            if len(top_moves) >= 3:
                # If only one of top three moves leads to clearly better position, it's the only good move
                return False
            
            return False
        except Exception as e:
            print(f"Error checking if only good move: {e}")
            return False

    @staticmethod
    def _check_great_conditions(
        prev: Info,
        curr: Info,
        prev_board: chess.Board,
        move: chess.Move,
        top_moves: List[str],
        move_difficulty: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Helper method to check if a move should be classified as 'great'.
        
        A move is considered great if it meets any of these conditions:
        1. It's the only good move in a complex position
        2. It turns a losing position into equal or winning
        3. It turns an equal position into clearly winning
        4. It maintains winning advantage in a complex position
        
        Args:
            prev: Position info before the move
            curr: Position info after the move
            prev_board: Board state before the move
            move: The chess move to analyze
            top_moves: List of top engine moves
            move_difficulty: Complexity score of the move (0-1)
            
        Returns:
            Tuple[bool, str]: (is_great, reason_message)
        """
        # Guard against None values
        if prev.cp is None or curr.cp is None:
            return False, ""
        
        # Check if this is the only good move
        is_only_good = MoveAnalyzer.is_only_good_move(prev_board, move, top_moves, prev)
        
        # Determine if this is a simple capture (capturing an undefended piece)
        is_simple_capture = False
        if prev_board:
            captured_piece = prev_board.piece_at(move.to_square)
            if captured_piece and not prev_board.is_attacked_by(not prev_board.turn, move.to_square):
                is_simple_capture = True
        
        # Case 1: Difficult and only good move
        if move_difficulty > 0.3 and is_only_good and not is_simple_capture:
            return True, f" | GREAT: Only good move in a complex position (complexity={move_difficulty:.2f})"
        
        # Case 2: Turns losing into equal/winning
        # For white (prev.color = True): Turn < -150 CP into > -50 CP
        # For black (prev.color = False): Turn > +150 CP into < +50 CP
        if ((prev.color and prev.cp < -150 and curr.cp > -50) or 
            (not prev.color and prev.cp > 150 and curr.cp < 50)):
            return True, f" | GREAT: Turns losing position into equal/winning (prev_cp={prev.cp}, curr_cp={curr.cp})"
        
        # Case 3: Turns equal into clearly winning
        # For white (prev.color = True): Turn ±50 CP into > +200 CP
        # For black (prev.color = False): Turn ±50 CP into < -200 CP
        if ((prev.color and abs(prev.cp) < 50 and curr.cp > 200) or 
            (not prev.color and abs(prev.cp) < 50 and curr.cp < -200)):
            return True, f" | GREAT: Turns equal position into clearly winning (prev_cp={prev.cp}, curr_cp={curr.cp})"
        
        # Case 4: Maintains winning advantage in complex position
        # Only counts for difficult positions that aren't simple captures
        if (move_difficulty > 0.4 and not is_simple_capture and 
            ((prev.color and prev.cp > 200 and curr.cp > 180) or 
            (not prev.color and prev.cp < -200 and curr.cp < -180))):
            return True, f" | GREAT: Maintains winning advantage in complex position (complexity={move_difficulty:.2f})"
        
        return False, ""

    @staticmethod
    def move_complexity(prev_board: chess.Board, move: chess.Move) -> float:
        """
        Calculate the complexity/difficulty of finding a move.
        
        Higher scores indicate more complex moves that are harder to find.
        
        Args:
            prev_board: The board state before the move
            move: The chess move to analyze
            
        Returns:
            float: Complexity score (0-1), where higher means more complex
        """
        if not prev_board or not move:
            return 0.0
            
        try:
            complexity = 0.0
            
            # Create the resulting board after the move
            after_board = prev_board.copy()
            after_board.push(move)
            
            # Get piece information up front to avoid repeated lookups
            moving_piece = prev_board.piece_at(move.from_square)
            captured_piece = prev_board.piece_at(move.to_square)
            
            if not moving_piece:
                return 0.0  # Can't analyze a move without a piece
            
            piece_type = moving_piece.piece_type
            distance = chess.square_distance(move.from_square, move.to_square)
            destination_rank = chess.square_rank(move.to_square)
            destination_file = chess.square_file(move.to_square)
            game_stage = len(prev_board.move_stack)
            
            # === TACTICAL FACTORS ===
            
            # Sacrifices (greatly increases complexity)
            if MoveAnalyzer.is_piece_sacrifice(prev_board, after_board, move):
                complexity += 0.4
            
            # Captures with evaluation
            if captured_piece:
                moving_value = MoveAnalyzer.PIECE_VALUES[piece_type]
                captured_value = MoveAnalyzer.PIECE_VALUES[captured_piece.piece_type]
                
                # Capturing with less/equal valuable piece (usually obvious)
                if moving_value <= captured_value:
                    complexity -= 0.15
                    
                # Capturing undefended pieces (obvious)
                if not prev_board.is_attacked_by(not prev_board.turn, move.to_square):
                    complexity -= 0.25
                    
                # Capturing with more valuable piece (sometimes complex)
                elif moving_value > captured_value:
                    complexity += 0.1
            
            # Check detection
            if after_board.is_check():
                # Discovered checks are complex and impressive
                piece_gives_check = False
                # Create a board where only the moved piece exists
                check_test_board = chess.Board(None)
                check_test_board.set_piece_at(move.to_square, moving_piece)
                check_test_board.set_piece_at(after_board.king(not prev_board.turn), chess.Piece(chess.KING, not prev_board.turn))
                
                # If the moved piece is directly checking, it's a direct check
                if check_test_board.is_check():
                    piece_gives_check = True
                    
                if not piece_gives_check:
                    # It's a discovered check (more complex)
                    complexity += 0.3
                else:
                    # Direct checks are generally more obvious
                    complexity -= 0.1
            
            # === PIECE-SPECIFIC CONSIDERATIONS ===
            
            # Knight moves (less intuitive patterns)
            if piece_type == chess.KNIGHT:
                complexity += 0.15
            
            # Long-range piece moves
            if piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP] and distance > 3:
                complexity += 0.2
            
            # Pawn moves evaluation
            if piece_type == chess.PAWN:
                # Promotions are usually obvious
                if destination_rank in [0, 7]:
                    complexity -= 0.2
                    
                # Quiet pawn moves in middlegame can be strategic
                elif not captured_piece and 2 <= destination_rank <= 5:
                    complexity += 0.1
                    
                # Pawn captures are usually obvious
                elif captured_piece:
                    complexity -= 0.05
            
            # === STRATEGIC FACTORS ===
            
            # Moving to attacked squares (without capturing) is complex
            if after_board.is_attacked_by(not prev_board.turn, move.to_square) and not captured_piece:
                complexity += 0.25
            
            # Moving away from attacked squares is reactive and obvious
            if prev_board.is_attacked_by(not prev_board.turn, move.from_square):
                complexity -= 0.2
            
            # Board geography considerations
            if game_stage < 20 and 2 <= destination_rank <= 5 and 2 <= destination_file <= 5:
                # Center control in early game is standard
                complexity -= 0.1
            elif destination_rank in [0, 7] or destination_file in [0, 7]:
                # Edge/corner maneuvers can be complex
                complexity += 0.1
            
            # King safety considerations
            if piece_type == chess.KING:
                # Castling is generally obvious
                if prev_board.is_castling(move):
                    complexity -= 0.2
                # King moves in middlegame are often complex
                elif 8 < game_stage < 40:
                    complexity += 0.3
            
            # === TEMPO AND DEVELOPMENT ===
            
            # Development in opening
            if game_stage < 15:
                # Standard development moves are obvious
                if (piece_type in [chess.KNIGHT, chess.BISHOP] and 
                    chess.square_rank(move.from_square) in [0, 1, 6, 7] and
                    chess.square_rank(move.to_square) not in [0, 1, 6, 7]):
                    complexity -= 0.2
                    
                # Early queen/rook development is unusual
                if piece_type in [chess.QUEEN, chess.ROOK] and game_stage < 10:
                    complexity += 0.15
            
            # === ENDGAME SPECIFIC ===
            
            # Endgame king activity
            if game_stage > 40 and piece_type == chess.KING:
                # Active king in endgame is standard
                if 2 <= destination_rank <= 5 and 2 <= destination_file <= 5:
                    complexity -= 0.15
            
            # Ensure complexity is within bounds [0.0, 1.0]
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
                    return (Judgment.GOOD, reason)
            
            # Check if this is a simple capture of an undefended piece
            is_simple_capture = False
            if prev_board and move:
                captured_piece = prev_board.piece_at(move.to_square)
                if captured_piece and not prev_board.is_attacked_by(not prev_board.turn, move.to_square):
                    # If capturing an undefended piece, it's a simple capture
                    is_simple_capture = True
                    reason += " | SIMPLE CAPTURE: Taking an undefended piece"
            
            # Check if move is in top engine moves
            is_top_move = False
            if move and top_moves and len(top_moves) > 0:
                move_uci = move.uci()
                
                # Try direct UCI comparison
                if move_uci in top_moves:
                    is_top_move = True
                    if move_uci == top_moves[0]:
                        logger.debug(f"TOP MOVE (UCI): {move_uci} matches engine's first choice: {top_moves[:3]}")
                        reason += " | TOP MOVE: Matches engine's first choice"
                    else:
                        move_index = top_moves.index(move_uci) + 1  # 1-based index for display
                        logger.debug(f"GOOD MOVE (UCI): {move_uci} found in top moves at position {move_index}: {top_moves[:3]}")
                        reason += f" | GOOD MOVE: Within engine's top {move_index} choices"
                # Try SAN format if we have the board
                elif prev_board:
                    try:
                        # Try to get SAN format of the move
                        move_san = prev_board.san(move)
                        if move_san in top_moves:
                            is_top_move = True
                            if move_san == top_moves[0]:
                                logger.debug(f"TOP MOVE (SAN): {move_san} matches engine's first choice: {top_moves[:3]}")
                                reason += " | TOP MOVE: Matches engine's first choice"
                            else:
                                move_index = top_moves.index(move_san) + 1  # 1-based index for display
                                logger.debug(f"GOOD MOVE (SAN): {move_san} found in top moves at position {move_index}: {top_moves[:3]}")
                                reason += f" | GOOD MOVE: Within engine's top {move_index} choices"
                        else:
                            logger.debug(f"MATCH (SAN): Move {move_san} not found in top moves: {top_moves}")
                    except Exception as e:
                        logger.warning(f"Error converting move to SAN: {e}")
                
                # If it's a top move, check for brilliant/great moves
                if is_top_move and prev_board and curr_board and move:
                    # Check for brilliant move conditions
                    is_brilliant, brilliant_reason = MoveAnalyzer._check_brilliant_conditions(
                        prev, curr, prev_board, curr_board, move
                    )
                    if is_brilliant:
                        return (Judgment.BRILLIANT, reason + brilliant_reason)
                        
                    # Check for great move conditions - but not for simple captures
                    if not is_simple_capture:
                        move_difficulty = MoveAnalyzer.move_complexity(prev_board, move)
                        is_great, great_reason = MoveAnalyzer._check_great_conditions(
                            prev, curr, prev_board, move, top_moves, move_difficulty
                        )
                        if is_great:
                            return (Judgment.GREAT, reason + great_reason)
            else:
                if move and top_moves:
                    reason += f" | NOT TOP MOVE: Played {move.uci()}, top was {top_moves[0] if top_moves else 'unknown'}"
            
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
                # logger.warning(f"Missing evaluation cp value(s), defaulting to GOOD: {prev.cp}, {curr.cp}")
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
                # if top move skip blunder
                if is_top_move:
                    return (Judgment.GOOD, reason)
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
                
                # Check for GREAT move again - but not for simple captures
                if top_moves and not is_simple_capture:
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

    @staticmethod
    def calculate_move_accuracy(win_percent_before: float, win_percent_after: float, is_top_move: bool = False) -> float:
        """
        Calculate move accuracy percentage based on the difference in winning chances.
        
        Uses the exact Lichess formula from their codebase.
        
        Args:
            win_percent_before: Winning percentage before the move (0.0 to 1.0)
            win_percent_after: Winning percentage after the move (0.0 to 1.0)
            is_top_move: Whether the played move is the engine's top choice
            
        Returns:
            Accuracy percentage (0 to 100)
        """
        try:
            # Handle None values
            if win_percent_before is None or win_percent_after is None:
                return 100.0
            
            # If it's the top move, return 100% accuracy directly
            if is_top_move:
                return 100.0
            
            # If the position improved or stayed the same, return 100%
            if win_percent_after >= win_percent_before:
                return 100.0
                
            # Calculate the drop in winning percentage (0-100 range)
            win_diff = (win_percent_before - win_percent_after) * 100
            
            # Exact Lichess formula
            raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) + -3.166924740191411
            
            # Add +1 uncertainty bonus (due to imperfect analysis)
            accuracy = raw + 1.0
            
            # Clamp the result between 0 and 100
            return max(0, min(100, accuracy))
        except Exception as e:
            logger.warning(f"Error calculating move accuracy: {e}")
            return 0.0  # Return 0 accuracy on error