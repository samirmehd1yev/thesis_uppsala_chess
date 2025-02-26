# src/features/extractor.py
import chess
import chess.pgn
from typing import Dict, List
import numpy as np
from models.data_classes import FeatureVector, Info
from models.enums import Judgment
from analysis.phase_detector import GamePhaseDetector
from analysis.move_analyzer import MoveAnalyzer

class FeatureExtractor:
    def __init__(self):
        self.phase_detector = GamePhaseDetector()
        
    # CHANGE THIS METHOD - Added judgments parameter
    def extract_features(self, game: chess.pgn.Game, evals: List[Info] = None, judgments: List[Judgment] = None) -> FeatureVector:
        """Extract all features from a game"""
        if game is None:
            raise ValueError("Game cannot be None")
            
        if evals is not None and not isinstance(evals, list):
            raise TypeError("evals must be a list of Info objects")
            
        features = FeatureVector()
        
        # Get positions and phases
        positions = self._get_positions(game)
        mg_start, eg_start = self.phase_detector.find_phase_transitions(positions)
        total_moves = len(positions) // 2
        features.total_moves = total_moves
        
        opening_length, middlegame_length, endgame_length = self._phase_length(mg_start, eg_start, total_moves)
        features.opening_length = opening_length
        features.middlegame_length = middlegame_length
        features.endgame_length = endgame_length
        
        # Material and position features
        features.material_balance_changes = self._calculate_material_changes(positions)
        features.piece_mobility_avg = self._calculate_mobility(positions)
        features.pawn_structure_changes = self._calculate_pawn_changes(positions)
        features.center_control_avg = self._calculate_center_control(positions)
        
        # Calculate quality metrics if evaluations available
        if evals and len(evals) > 1:
            # Get the actual moves played
            moves = list(game.mainline_moves())
            
            # CHANGE HERE - Use pre-calculated judgments if provided
            if judgments and len(judgments) > 0:
                # Use pre-calculated judgments
                self._count_judgment_metrics(judgments, features)
                # Still count sacrifices if we have positions and moves
                if positions and moves:
                    self.count_sacrifices(positions, moves, features)
            else:
                # Otherwise calculate move qualities with board information for brilliant/great detection
                self._calculate_quality_metrics(evals, features, positions, moves)
        
        return features
    
    def _phase_length(self, mg_start: int, eg_start: int, total_moves: int) -> tuple[float, float, float]:
        """Calculate length of each game phase"""
        # Case analysis:
        # 1. No phases detected (mg_start = 0, eg_start = 0) -> All moves are opening
        # 2. Only middlegame detected (mg_start > 0, eg_start = 0) -> Opening + Middlegame
        # 3. Both phases detected (mg_start > 0, eg_start > 0) -> All three phases
        
        if mg_start == 0:
            # Case 1: Game never left opening phase
            opening_length = float(total_moves)
            middlegame_length = 0.0
            endgame_length = 0.0
        elif eg_start == 0:
            # Case 2: Game only reached middlegame
            opening_length = float(mg_start - 1)
            middlegame_length = float(total_moves - (mg_start - 1))
            endgame_length = 0.0
        else:
            # Case 3: Game reached all phases
            opening_length = float(mg_start - 1)
            middlegame_length = float(eg_start - mg_start)
            endgame_length = float(total_moves - eg_start + 1)
        
        return opening_length, middlegame_length, endgame_length
        
    # ADD THIS NEW METHOD
    def _count_judgment_metrics(self, judgments: List[Judgment], features: FeatureVector) -> None:
        """
        Count judgment metrics from pre-calculated judgments
        
        Args:
            judgments: List of pre-calculated move judgments
            features: FeatureVector object to update
        """
        # Initialize counters
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
        
        # Update feature vector with counts
        features.white_brilliant_count = white_counts[Judgment.BRILLIANT]
        features.white_great_count = white_counts[Judgment.GREAT]
        features.white_good_moves = white_counts[Judgment.GOOD]
        features.white_inaccuracy_count = white_counts[Judgment.INACCURACY]
        features.white_mistake_count = white_counts[Judgment.MISTAKE]
        features.white_blunder_count = white_counts[Judgment.BLUNDER]
        
        features.black_brilliant_count = black_counts[Judgment.BRILLIANT]
        features.black_great_count = black_counts[Judgment.GREAT]
        features.black_good_moves = black_counts[Judgment.GOOD]
        features.black_inaccuracy_count = black_counts[Judgment.INACCURACY]
        features.black_mistake_count = black_counts[Judgment.MISTAKE]
        features.black_blunder_count = black_counts[Judgment.BLUNDER]
        
        # Note: We can't count sacrifices here since we don't have board positions
        # This method is used when judgments are pre-calculated
        # Sacrifice counts will be 0 in this case
    
    def _get_positions(self, game: chess.pgn.Game) -> List[chess.Board]:
        """Get list of positions from game"""
        positions = []
        board = game.board()
        
        for move in game.mainline_moves():
            positions.append(board.copy())
            board.push(move)
            
        positions.append(board.copy())
        return positions
        
    def _calculate_material_changes(self, positions: List[chess.Board]) -> float:
        """Calculate rate of material balance changes"""
        if not positions:
            return 0.0
            
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        changes = 0
        for i in range(1, len(positions)):
            prev_mat = self._get_material_value(positions[i-1], material_values)
            curr_mat = self._get_material_value(positions[i], material_values)
            if prev_mat != curr_mat:
                changes += 1
                
        return changes / (len(positions) - 1) if len(positions) > 1 else 0
    
    def _get_material_value(self, board: chess.Board, values: Dict) -> int:
        """Get total material value on board"""
        total = 0
        for piece_type, value in values.items():
            total += len(board.pieces(piece_type, chess.WHITE)) * value
            total -= len(board.pieces(piece_type, chess.BLACK)) * value
        return total
    
    def _calculate_mobility(self, positions: List[chess.Board]) -> float:
        """Calculate average piece mobility"""
        total_mobility = 0
        for board in positions:
            mobility = len(list(board.legal_moves))
            total_mobility += mobility
            
        return total_mobility / len(positions) if positions else 0
    
    def _calculate_pawn_changes(self, positions: List[chess.Board]) -> float:
        """Calculate rate of pawn structure changes"""
        changes = 0
        for i in range(1, len(positions)):
            prev_pawns = self._get_pawn_structure(positions[i-1])
            curr_pawns = self._get_pawn_structure(positions[i])
            if prev_pawns != curr_pawns:
                changes += 1
                
        return changes / (len(positions) - 1) if len(positions) > 1 else 0
    
    def _get_pawn_structure(self, board: chess.Board) -> int:
        """Get pawn structure hash"""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        return white_pawns | (black_pawns << 32)
    
    def _calculate_center_control(self, positions: List[chess.Board]) -> float:
        """Calculate average center square control"""
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        total_control = 0
        
        for board in positions:
            control = 0
            for square in center_squares:
                if board.piece_at(square):
                    control += 1
            total_control += control / len(center_squares)
            
        return total_control / len(positions) if positions else 0
        
    # Keep this method for backwards compatibility
    def _calculate_quality_metrics(self, evals: List[Info], features: FeatureVector, 
                                  positions: List[chess.Board] = None, 
                                  moves: List[chess.Move] = None) -> None:
        """Calculate move quality related features and statistics for both players"""
        if len(positions) == 0:
            return
            
        if len(evals) < 2:
            return
            
        moves = list(game.mainline_moves())
        if len(moves) == 0:
            return
            
        judgments = {'White': [], 'Black': []}
        eval_changes = {'White': [], 'Black': []}
        
        # Initialize counters
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
        
        # Initialize sacrifice counters
        white_sacrifices = 0
        black_sacrifices = 0
        
        # Skip first evaluation as we need pairs to analyze moves
        for i in range(1, len(evals)):
            if i-1 >= len(moves):
                break
                
            prev, curr = evals[i-1], evals[i]
            is_white = (i - 1) % 2 == 0  # Even indices are White's moves
            color = 'White' if is_white else 'Black'
            
            # Get move quality with additional board information for brilliant/great moves
            if positions and moves and i-1 < len(moves) and i-1 < len(positions) and i < len(positions):
                move = moves[i-1]
                prev_board = positions[i-1]
                curr_board = positions[i]
                
                # Check for sacrifice
                if MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
                    if is_white:
                        white_sacrifices += 1
                    else:
                        black_sacrifices += 1
                
                judgment, _ = MoveAnalyzer.analyze_move_with_top_moves(
                    prev, curr, 
                    prev_board=prev_board, 
                    curr_board=curr_board, 
                    move=move
                )
            else:
                # Use basic analysis if board/move info not available or indices are out of bounds
                judgment = MoveAnalyzer.analyze_move(prev, curr)
            
            # Always append a judgment
            judgments[color].append(judgment)
            
            # Increment the appropriate counter
            if is_white:
                white_counts[judgment] += 1
            else:
                black_counts[judgment] += 1
            
            # Track eval changes
            if prev.cp is not None and curr.cp is not None:
                # Get raw eval change from current player's perspective
                eval_change = curr.cp - prev.cp
                # For Black's moves, negate the change to get their perspective
                if not is_white:
                    eval_change = -eval_change
                eval_changes[color].append(eval_change)
        
        # Directly set counts from the counters we've maintained
        features.white_brilliant_count = white_counts[Judgment.BRILLIANT]
        features.white_great_count = white_counts[Judgment.GREAT]
        features.white_good_moves = white_counts[Judgment.GOOD]
        features.white_inaccuracy_count = white_counts[Judgment.INACCURACY]
        features.white_mistake_count = white_counts[Judgment.MISTAKE]
        features.white_blunder_count = white_counts[Judgment.BLUNDER]
        features.white_sacrifice_count = white_sacrifices  # Set sacrifice count
        
        features.black_brilliant_count = black_counts[Judgment.BRILLIANT]
        features.black_great_count = black_counts[Judgment.GREAT]
        features.black_good_moves = black_counts[Judgment.GOOD]
        features.black_inaccuracy_count = black_counts[Judgment.INACCURACY]
        features.black_mistake_count = black_counts[Judgment.MISTAKE]
        features.black_blunder_count = black_counts[Judgment.BLUNDER]
        features.black_sacrifice_count = black_sacrifices  # Set sacrifice count
        
        # Calculate eval metrics with proper perspective normalization
        if eval_changes['White']:
            features.white_avg_eval_change = abs(np.mean(eval_changes['White']))
            features.white_eval_volatility = np.std(eval_changes['White'])
        
        if eval_changes['Black']:
            features.black_avg_eval_change = np.mean(eval_changes['Black'])
            features.black_eval_volatility = np.std(eval_changes['Black'])

    def count_sacrifices(self, positions: List[chess.Board], moves: List[chess.Move], features: FeatureVector) -> None:
        """
        Count sacrifices for both players
        
        Args:
            positions: List of board positions
            moves: List of moves played
            features: FeatureVector object to update
        """
        white_sacrifices = 0
        black_sacrifices = 0
        
        for i in range(len(moves)):
            if i < len(positions) and i+1 < len(positions):
                prev_board = positions[i]
                curr_board = positions[i+1]
                move = moves[i]
                
                if MoveAnalyzer.is_piece_sacrifice(prev_board, curr_board, move):
                    is_white = i % 2 == 0  # Even indices are White's moves
                    if is_white:
                        white_sacrifices += 1
                    else:
                        black_sacrifices += 1
        
        features.white_sacrifice_count = white_sacrifices
        features.black_sacrifice_count = black_sacrifices