import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple
import numpy as np
from models.data_classes import FeatureVector, Info
from models.enums import Judgment
from analysis.phase_detector import GamePhaseDetector
from analysis.move_analyzer import MoveAnalyzer
from analysis.king_safety import KingSafetyEvaluator

class FeatureExtractor:
    def __init__(self):
        self.phase_detector = GamePhaseDetector()
        self.king_safety_evaluator = KingSafetyEvaluator()
        
    def extract_features(self, game: chess.pgn.Game, evals: Optional[List[Info]] = None, judgments: Optional[List[Judgment]] = None) -> FeatureVector:
        """
        Extract all features from a game
        
        Args:
            game: Chess game to analyze
            evals: Optional list of position evaluations
            judgments: Optional pre-calculated move judgments
            
        Returns:
            FeatureVector containing extracted features
        """
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
        # Normalize phase lengths by total moves
        features.opening_length = opening_length / total_moves if total_moves > 0 else 0.0
        features.middlegame_length = middlegame_length / total_moves if total_moves > 0 else 0.0
        features.endgame_length = endgame_length / total_moves if total_moves > 0 else 0.0
        
        # Material and position features - separate for white and black
        white_material, black_material = self._calculate_material_changes_by_color(positions)
        features.white_material_changes = white_material
        features.black_material_changes = black_material
        
        white_mobility, black_mobility = self._calculate_mobility_by_color(positions)
        features.white_piece_mobility_avg = white_mobility
        features.black_piece_mobility_avg = black_mobility
        
        white_pawn_changes, black_pawn_changes = self._calculate_pawn_changes_by_color(positions)
        features.white_pawn_structure_changes = white_pawn_changes
        features.black_pawn_structure_changes = black_pawn_changes
        
        white_center, black_center = self._calculate_center_control_by_color(positions)
        features.white_center_control_avg = white_center
        features.black_center_control_avg = black_center
        
        # Calculate development metrics - new
        moves_list = list(game.mainline_moves())
        minor_piece_development_white, minor_piece_development_black, \
        queen_development_white, queen_development_black = self._calculate_development_metrics(positions, moves_list)
        
        # Normalize development metrics by total moves
        features.minor_piece_development_white = minor_piece_development_white / total_moves if total_moves > 0 else 0.0
        features.minor_piece_development_black = minor_piece_development_black / total_moves if total_moves > 0 else 0.0
        features.queen_development_white = queen_development_white / total_moves if total_moves > 0 else 0.0
        features.queen_development_black = queen_development_black / total_moves if total_moves > 0 else 0.0
        
        # Calculate engine move alignment - new
        if evals and len(evals) > 1:
            top_move_alignment_white, top_move_alignment_black, top2_3_move_alignment_white, top2_3_move_alignment_black = self._calculate_engine_move_alignment(moves_list, evals)
            
            features.top_move_alignment_white = top_move_alignment_white
            features.top_move_alignment_black = top_move_alignment_black
            features.top2_3_move_alignment_white = top2_3_move_alignment_white
            features.top2_3_move_alignment_black = top2_3_move_alignment_black
        
        # Calculate material features - new addition
        material_volatility_white, material_volatility_black, material_balance_std, piece_exchange_rate_white, piece_exchange_rate_black, pawn_exchange_rate_white, pawn_exchange_rate_black = self._calculate_material_features(positions)
        # print(f"Material volatility white: {material_volatility_white}, Material volatility black: {material_volatility_black}, Material balance std: {material_balance_std}, Piece exchange rate white: {piece_exchange_rate_white}, Piece exchange rate black: {piece_exchange_rate_black}, Pawn exchange rate white: {pawn_exchange_rate_white}, Pawn exchange rate black: {pawn_exchange_rate_black}")
        features.material_volatility_white = material_volatility_white
        features.material_volatility_black = material_volatility_black
        features.material_balance_std = material_balance_std
        features.piece_exchange_rate_white = piece_exchange_rate_white
        features.piece_exchange_rate_black = piece_exchange_rate_black
        features.pawn_exchange_rate_white = pawn_exchange_rate_white
        features.pawn_exchange_rate_black = pawn_exchange_rate_black
        
        # Calculate position control features - new addition
        space_advantage_white, space_advantage_black, \
        pawn_control_white, pawn_control_black = self._calculate_position_control_features(positions)
        
        features.space_advantage_white = space_advantage_white
        features.space_advantage_black = space_advantage_black
        features.pawn_control_white = pawn_control_white
        features.pawn_control_black = pawn_control_black
        
        # King safety features - new addition
        king_safety_metrics = self._calculate_king_safety(positions)
        features.white_king_safety = king_safety_metrics['white']['avg_safety'] 
        features.black_king_safety = king_safety_metrics['black']['avg_safety']
        features.white_king_safety_min = king_safety_metrics['white']['min_safety']
        features.black_king_safety_min = king_safety_metrics['black']['min_safety']
        
        # Normalize vulnerability spikes by total moves
        features.white_vulnerability_spikes = king_safety_metrics['white']['vulnerability_spikes'] / total_moves if total_moves > 0 else 0.0
        features.black_vulnerability_spikes = king_safety_metrics['black']['vulnerability_spikes'] / total_moves if total_moves > 0 else 0.0
        
        # Move statistics
        # Convert mainline_moves to a list before passing to _calculate_move_statistics
        moves_list = list(game.mainline_moves())
        capture_frequency_white, capture_frequency_black, check_frequency_white, check_frequency_black, castle_move_white, castle_move_black = self._calculate_move_statistics(positions, moves_list)
        features.capture_frequency_white = capture_frequency_white
        features.capture_frequency_black = capture_frequency_black
        features.check_frequency_white = check_frequency_white
        features.check_frequency_black = check_frequency_black
        
        # Normalize castling move numbers by total moves
        features.castle_move_white = castle_move_white / total_moves if total_moves > 0 else 0.0
        features.castle_move_black = castle_move_black / total_moves if total_moves > 0 else 0.0
        
        # Calculate quality metrics if evaluations available
        if evals and len(evals) > 1:
            # Get the actual moves played
            moves = list(game.mainline_moves())
            
            # Use pre-calculated judgments if provided
            if judgments and len(judgments) > 0:
                # print("CAMEEEEEE TO COUNT JUDGMENTS")
                self._count_judgment_metrics(judgments, features)
                # Still count sacrifices if we have positions and moves
                if positions and moves:
                    self.count_sacrifices(positions, moves, features)
            else:
                # Otherwise calculate move qualities with board information for brilliant/great detection
                self._calculate_quality_metrics(evals, features, positions, moves)
                # print("CAMEEEEEE TO COUNT QUALITY METRICS")

            self._calculate_eval_changes(evals, features, moves)
            # Normalize move quality counts by total moves
            if total_moves > 0:
                features.white_brilliant_count /= total_moves
                features.white_great_count /= total_moves
                features.white_good_moves /= total_moves
                features.white_inaccuracy_count /= total_moves
                features.white_mistake_count /= total_moves
                features.white_blunder_count /= total_moves
                features.white_sacrifice_count /= total_moves
                
                features.black_brilliant_count /= total_moves
                features.black_great_count /= total_moves
                features.black_good_moves /= total_moves
                features.black_inaccuracy_count /= total_moves
                features.black_mistake_count /= total_moves
                features.black_blunder_count /= total_moves
                features.black_sacrifice_count /= total_moves
        
        return features
    
    def _calculate_king_safety(self, positions: List[chess.Board]) -> Dict[str, Dict[str, float]]:
        """
        Calculate king safety metrics for both players throughout the game.
        
        Args:
            positions: List of board positions
            
        Returns:
            Dictionary with king safety metrics for both players
        """
        if not positions:
            return {
                "white": {
                    "avg_safety": 0,
                    "min_safety": 0,
                    "safety_drop": 0,
                    "vulnerability_spikes": 0
                },
                "black": {
                    "avg_safety": 0,
                    "min_safety": 0,
                    "safety_drop": 0,
                    "vulnerability_spikes": 0
                }
            }
        
        # Track safety scores for both colors
        white_scores = []
        black_scores = []
        
        # Skip first few positions to avoid early game anomalies
        start_idx = min(10, len(positions) // 4)
        
        # Evaluate king safety for each position
        for position in positions[start_idx:]:
            # Skip invalid positions
            if position.king(chess.WHITE) is None or position.king(chess.BLACK) is None:
                continue
                
            # Get safety scores for both sides
            white_safety = self.king_safety_evaluator.evaluate_king_safety(position, chess.WHITE)
            black_safety = self.king_safety_evaluator.evaluate_king_safety(position, chess.BLACK)
            
            white_scores.append(white_safety)
            black_scores.append(black_safety)
        
        # Calculate derived metrics
        white_features = self._calculate_safety_metrics(white_scores)
        black_features = self._calculate_safety_metrics(black_scores)
        
        return {
            "white": white_features,
            "black": black_features
        }
    
    def _calculate_safety_metrics(self, safety_scores: List[int]) -> Dict[str, float]:
        """
        Calculate derived metrics from raw safety scores.
        
        Args:
            safety_scores: List of safety scores for a player
            
        Returns:
            Dictionary of derived metrics
        """
        if not safety_scores:
            return {
                "avg_safety": 0,
                "min_safety": 0,
                "safety_drop": 0,
                "vulnerability_spikes": 0
            }
        
        # Average safety score
        avg_safety = sum(safety_scores) / len(safety_scores)
        
        # Minimum safety score
        min_safety = min(safety_scores)
        
        # Calculate maximum safety drop between consecutive positions
        max_drop = 0
        for i in range(1, len(safety_scores)):
            drop = safety_scores[i-1] - safety_scores[i]
            max_drop = max(max_drop, drop)
        
        # Count vulnerability spikes (sudden drops in safety)
        threshold = min(100, abs(avg_safety * 0.3))  # 30% of average as threshold
        vulnerability_spikes = 0
        for i in range(1, len(safety_scores)):
            if safety_scores[i-1] - safety_scores[i] > threshold:
                vulnerability_spikes += 1
        
        return {
            "avg_safety": avg_safety,
            "min_safety": min_safety,
            "safety_drop": max_drop,
            "vulnerability_spikes": vulnerability_spikes
        }
    
    def _phase_length(self, mg_start: int, eg_start: int, total_moves: int) -> Tuple[float, float, float]:
        """
        Calculate length of each game phase
        
        Args:
            mg_start: Move number where middlegame starts
            eg_start: Move number where endgame starts
            total_moves: Total number of moves in the game
            
        Returns:
            Tuple of (opening_length, middlegame_length, endgame_length)
        """
        if mg_start == 0:
            # Game never left opening phase
            opening_length = float(total_moves)
            middlegame_length = 0.0
            endgame_length = 0.0
        elif eg_start == 0:
            # Game only reached middlegame
            opening_length = float(mg_start - 1)
            middlegame_length = float(total_moves - (mg_start - 1))
            endgame_length = 0.0
        else:
            # Game reached all phases
            opening_length = float(mg_start - 1)
            middlegame_length = float(eg_start - mg_start)
            endgame_length = float(total_moves - eg_start + 1)
        
        return opening_length, middlegame_length, endgame_length
        
    def _count_judgment_metrics(self, judgments: List[Judgment], features: FeatureVector) -> None:
        """
        Count judgment metrics from pre-calculated judgments
        
        Args:
            judgments: List of pre-calculated move judgments
            features: FeatureVector object to update
        """
        # Initialize counters
        white_counts = {judgment: 0 for judgment in Judgment}
        black_counts = {judgment: 0 for judgment in Judgment}
        
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
    
    def _get_positions(self, game: chess.pgn.Game) -> List[chess.Board]:
        """
        Get list of positions from game
        
        Args:
            game: Chess game to get positions from
            
        Returns:
            List of chess board positions
        """
        positions = []
        board = game.board()
        
        for move in game.mainline_moves():
            positions.append(board.copy())
            board.push(move)
            
        positions.append(board.copy())  # Final position
        return positions
        
    def _calculate_material_changes_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate rate of material changes separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_changes_rate, black_changes_rate)
        """
        if not positions:
            return 0.0, 0.0
            
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_changes = 0
        black_changes = 0
        
        # Track material values for debugging
        white_material_values = []
        black_material_values = []
        
        # Calculate initial material values
        prev_white_mat = self._get_material_value_for_color(positions[0], material_values, chess.WHITE)
        prev_black_mat = self._get_material_value_for_color(positions[0], material_values, chess.BLACK)
        
        white_material_values.append(prev_white_mat)
        black_material_values.append(prev_black_mat)
        
        import logging
        logging.debug(f"Initial material values in _calculate_material_changes_by_color - White: {prev_white_mat}, Black: {prev_black_mat}")
        
        for i in range(1, len(positions)):
            curr_white_mat = self._get_material_value_for_color(positions[i], material_values, chess.WHITE)
            curr_black_mat = self._get_material_value_for_color(positions[i], material_values, chess.BLACK)
            
            white_material_values.append(curr_white_mat)
            black_material_values.append(curr_black_mat)
            
            # Count any change in material value
            if prev_white_mat != curr_white_mat:
                white_changes += 1
                logging.debug(f"Move {i//2 + 1}: White material change: {prev_white_mat} -> {curr_white_mat}")
                
            if prev_black_mat != curr_black_mat:
                black_changes += 1
                logging.debug(f"Move {i//2 + 1}: Black material change: {prev_black_mat} -> {curr_black_mat}")
            
            # Update previous values for next iteration
            prev_white_mat = curr_white_mat
            prev_black_mat = curr_black_mat
                
        white_change_rate = white_changes / (len(positions) - 1) if len(positions) > 1 else 0
        black_change_rate = black_changes / (len(positions) - 1) if len(positions) > 1 else 0
        
        # Add debug logging
        logging.debug(f"Material changes count in _calculate_material_changes_by_color - White: {white_changes}, Black: {black_changes}")
        logging.debug(f"White material values: {white_material_values}")
        logging.debug(f"Black material values: {black_material_values}")
        logging.debug(f"Material change rates - White: {white_change_rate}, Black: {black_change_rate}")
        
        return white_change_rate, black_change_rate
    
    def _get_material_value_for_color(self, board: chess.Board, values: Dict[chess.PieceType, int], color: chess.Color) -> int:
        """
        Get total material value for a specific color
        
        Args:
            board: Chess board to evaluate
            values: Dictionary mapping piece types to values
            color: Chess color to calculate value for
            
        Returns:
            Material value for the specified color
        """
        total = 0
        for piece_type, value in values.items():
            total += len(board.pieces(piece_type, color)) * value
        return total
    
    def _calculate_mobility_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate average piece mobility separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_mobility_avg, black_mobility_avg)
        """
        if not positions:
            return 0.0, 0.0
            
        white_mobility_total = 0
        black_mobility_total = 0
        white_positions = 0
        black_positions = 0
        
        for board in positions:
            if board.turn == chess.WHITE:
                white_mobility_total += len(list(board.legal_moves))
                white_positions += 1
            else:
                black_mobility_total += len(list(board.legal_moves))
                black_positions += 1
                
        white_avg = white_mobility_total / white_positions if white_positions > 0 else 0
        black_avg = black_mobility_total / black_positions if black_positions > 0 else 0
        
        return white_avg, black_avg
    
    def _calculate_pawn_changes_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate rate of pawn structure changes separately for white and black
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_pawn_changes_rate, black_pawn_changes_rate)
        """
        if not positions or len(positions) <= 1:
            return 0.0, 0.0
            
        white_changes = 0
        black_changes = 0
        
        for i in range(1, len(positions)):
            prev_white_pawns = self._get_pawn_structure_for_color(positions[i-1], chess.WHITE)
            curr_white_pawns = self._get_pawn_structure_for_color(positions[i], chess.WHITE)
            
            prev_black_pawns = self._get_pawn_structure_for_color(positions[i-1], chess.BLACK)
            curr_black_pawns = self._get_pawn_structure_for_color(positions[i], chess.BLACK)
            
            if prev_white_pawns != curr_white_pawns:
                white_changes += 1
                
            if prev_black_pawns != curr_black_pawns:
                black_changes += 1
                
        white_change_rate = white_changes / (len(positions) - 1)
        black_change_rate = black_changes / (len(positions) - 1)
        
        return white_change_rate, black_change_rate
    
    def _get_pawn_structure_for_color(self, board: chess.Board, color: chess.Color) -> int:
        """
        Get pawn structure hash for a specific color
        
        Args:
            board: Chess board to evaluate
            color: Chess color to calculate pawn structure for
            
        Returns:
            Hash value representing pawn structure for the specified color
        """
        return board.pieces(chess.PAWN, color)
    
    def _calculate_center_control_by_color(self, positions: List[chess.Board]) -> tuple:
        """
        Calculate average center control for each color
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of (white_center_control, black_center_control)
        """
        if not positions:
            return 0.0, 0.0
            
        # Define center squares
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        white_control_sum = 0
        black_control_sum = 0
        
        for board in positions:
            white_control = 0
            black_control = 0
            
            for square in center_squares:
                # Count attackers for each square
                white_attackers = len(board.attackers(chess.WHITE, square))
                black_attackers = len(board.attackers(chess.BLACK, square))
                
                white_control += white_attackers
                black_control += black_attackers
            
            # Normalize by number of center squares
            white_control_sum += white_control / len(center_squares)
            black_control_sum += black_control / len(center_squares)
        
        # Calculate averages
        white_center_control = white_control_sum / len(positions)
        black_center_control = black_control_sum / len(positions)
        
        return white_center_control, black_center_control
        
    def _calculate_material_features(self, positions: List[chess.Board]) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculate material-related features including volatility, balance std, and exchange rates.
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of material features
        """
        if not positions or len(positions) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King's value is not included in material count
        }
        
        # Track material values throughout the game
        white_material_values = []
        black_material_values = []
        material_balances = []
        
        white_material_changes = []
        black_material_changes = []
        
        white_piece_exchanges = 0
        black_piece_exchanges = 0
        white_pawn_exchanges = 0
        black_pawn_exchanges = 0
        
        white_move_count = 0
        black_move_count = 0
        
        # Calculate material values for each position
        for board in positions:
            white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                                for piece_type, value in material_values.items())
            black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                                for piece_type, value in material_values.items())
            
            white_material_values.append(white_material)
            black_material_values.append(black_material)
            material_balances.append(white_material - black_material)
        
        # Log initial material values
        import logging
        logging.debug(f"Initial material values - White: {white_material_values[0]}, Black: {black_material_values[0]}")
        
        # Calculate material changes and detect exchanges
        for i in range(1, len(positions)):
            prev_board = positions[i-1]
            curr_board = positions[i]
            
            prev_white_material = white_material_values[i-1]
            curr_white_material = white_material_values[i]
            prev_black_material = black_material_values[i-1]
            curr_black_material = black_material_values[i]
            
            # Calculate material changes for each player (regardless of whose turn it is)
            white_material_change = abs(curr_white_material - prev_white_material)
            black_material_change = abs(curr_black_material - prev_black_material)
            
            # Always track material changes for both players
            white_material_changes.append(white_material_change)
            black_material_changes.append(black_material_change)
            
            # Log material changes for debugging
            if white_material_change > 0 or black_material_change > 0:
                logging.debug(f"Move {i//2 + 1}: Material change - White: {white_material_change}, Black: {black_material_change}")
                logging.debug(f"  White material: {prev_white_material} -> {curr_white_material}")
                logging.debug(f"  Black material: {prev_black_material} -> {curr_black_material}")
            
            # Determine whose move it was
            is_white_move = prev_board.turn == chess.WHITE
            
            if is_white_move:
                white_move_count += 1
                
                # Check for captures by White (Black lost material)
                if curr_black_material < prev_black_material:
                    # Count what kind of pieces were captured
                    black_pieces_before = sum(len(prev_board.pieces(piece_type, chess.BLACK)) 
                                            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                    black_pieces_after = sum(len(curr_board.pieces(piece_type, chess.BLACK)) 
                                            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                    
                    black_pawns_before = len(prev_board.pieces(chess.PAWN, chess.BLACK))
                    black_pawns_after = len(curr_board.pieces(chess.PAWN, chess.BLACK))
                    
                    # Count piece captures
                    if black_pieces_before > black_pieces_after:
                        white_piece_exchanges += 1
                        logging.debug(f"Move {i//2 + 1}: White captured Black piece")
                    
                    # Count pawn captures
                    if black_pawns_before > black_pawns_after:
                        white_pawn_exchanges += 1
                        logging.debug(f"Move {i//2 + 1}: White captured Black pawn")
            else:
                # Black's move
                black_move_count += 1
                
                # Check for captures by Black (White lost material)
                if curr_white_material < prev_white_material:
                    # Count what kind of pieces were captured
                    white_pieces_before = sum(len(prev_board.pieces(piece_type, chess.WHITE)) 
                                            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                    white_pieces_after = sum(len(curr_board.pieces(piece_type, chess.WHITE)) 
                                            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                    
                    white_pawns_before = len(prev_board.pieces(chess.PAWN, chess.WHITE))
                    white_pawns_after = len(curr_board.pieces(chess.PAWN, chess.WHITE))
                    
                    # Count piece captures
                    if white_pieces_before > white_pieces_after:
                        black_piece_exchanges += 1
                        logging.debug(f"Move {i//2 + 1}: Black captured White piece")
                    
                    # Count pawn captures
                    if white_pawns_before > white_pawns_after:
                        black_pawn_exchanges += 1
                        logging.debug(f"Move {i//2 + 1}: Black captured White pawn")
        
        # Calculate final metrics
        # Use max(1, len()) to avoid division by zero
        material_volatility_white = sum(white_material_changes) / max(1, len(white_material_changes))
        material_volatility_black = sum(black_material_changes) / max(1, len(black_material_changes))
        
        material_balance_std = float(np.std(material_balances)) if len(material_balances) > 1 else 0.0
        
        piece_exchange_rate_white = white_piece_exchanges / max(1, white_move_count)
        piece_exchange_rate_black = black_piece_exchanges / max(1, black_move_count)
        
        pawn_exchange_rate_white = white_pawn_exchanges / max(1, white_move_count)
        pawn_exchange_rate_black = black_pawn_exchanges / max(1, black_move_count)
        
        # Add summary logging
        logging.debug(f"Material changes - White: {sum(white_material_changes)}, Black: {sum(black_material_changes)}")
        logging.debug(f"Material volatility - White: {material_volatility_white}, Black: {material_volatility_black}")
        logging.debug(f"Piece exchanges - White: {white_piece_exchanges}, Black: {black_piece_exchanges}")
        logging.debug(f"Pawn exchanges - White: {white_pawn_exchanges}, Black: {black_pawn_exchanges}")
        logging.debug(f"Material balance std: {material_balance_std}")
        
        return (material_volatility_white, material_volatility_black, material_balance_std,
                piece_exchange_rate_white, piece_exchange_rate_black,
                pawn_exchange_rate_white, pawn_exchange_rate_black)
        
    def _calculate_position_control_features(self, positions: List[chess.Board]) -> Tuple[float, float, float, float]:
        """
        Calculate position control features including space advantage and pawn control.
        
        Args:
            positions: List of board positions
            
        Returns:
            Tuple of position control features
        """
        if not positions:
            return 0.0, 0.0, 0.0, 0.0
        
        # Define board halves
        white_half = set()
        black_half = set()
        
        for rank in range(0, 4):  # Ranks 1-4 (0-3 in zero-based indexing)
            for file in range(0, 8):
                white_half.add(chess.square(file, rank))
        
        for rank in range(4, 8):  # Ranks 5-8 (4-7 in zero-based indexing)
            for file in range(0, 8):
                black_half.add(chess.square(file, rank))
        
        # Initialize accumulators for each metric
        space_advantage_white_total = 0
        space_advantage_black_total = 0
        pawn_control_white_total = 0
        pawn_control_black_total = 0
        
        # Process each position
        for board in positions:
            # Space advantage - count controlled squares in opponent's half
            white_space = sum(1 for sq in black_half if board.is_attacked_by(chess.WHITE, sq))
            black_space = sum(1 for sq in white_half if board.is_attacked_by(chess.BLACK, sq))
            
            # Pawn control - count squares controlled by pawns
            white_pawn_control = self._calculate_pawn_control(board, chess.WHITE)
            black_pawn_control = self._calculate_pawn_control(board, chess.BLACK)
            
            # Accumulate metrics
            space_advantage_white_total += white_space / len(black_half)
            space_advantage_black_total += black_space / len(white_half)
            pawn_control_white_total += white_pawn_control
            pawn_control_black_total += black_pawn_control
        
        # Calculate averages
        num_positions = len(positions)
        space_advantage_white = space_advantage_white_total / num_positions
        space_advantage_black = space_advantage_black_total / num_positions
        pawn_control_white = pawn_control_white_total / num_positions
        pawn_control_black = pawn_control_black_total / num_positions
        
        return (space_advantage_white, space_advantage_black,
                pawn_control_white, pawn_control_black)

    def _calculate_piece_mobility_for_board(self, board: chess.Board) -> Tuple[float, float]:
        """
        Calculate average piece mobility for both players.
        
        Args:
            board: Chess board position
            
        Returns:
            Tuple of (white_mobility, black_mobility)
        """
        # Create copies of the board for each side's turn
        white_board = board.copy()
        black_board = board.copy()
        
        # Ensure it's White's turn on the white board
        if white_board.turn != chess.WHITE:
            white_board.push(chess.Move.null())
        
        # Ensure it's Black's turn on the black board
        if black_board.turn != chess.BLACK:
            black_board.push(chess.Move.null())
        
        # Count pieces for each player (excluding kings)
        white_piece_count = sum(1 for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
                              for _ in board.pieces(piece_type, chess.WHITE))
        
        black_piece_count = sum(1 for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
                              for _ in board.pieces(piece_type, chess.BLACK))
        
        # Count legal moves for each side
        white_legal_moves = len(list(white_board.legal_moves))
        black_legal_moves = len(list(black_board.legal_moves))
        
        # Calculate average mobility (moves per piece)
        white_mobility = white_legal_moves / white_piece_count if white_piece_count > 0 else 0.0
        black_mobility = black_legal_moves / black_piece_count if black_piece_count > 0 else 0.0
        
        return white_mobility, black_mobility

    def _calculate_pawn_control(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate the number of squares controlled by pawns.
        
        Args:
            board: Chess board position
            color: Player color to check pawn control for
            
        Returns:
            Average number of squares controlled per pawn
        """
        pawn_squares = board.pieces(chess.PAWN, color)
        pawn_count = len(pawn_squares)
        
        if pawn_count == 0:
            return 0.0
        
        controlled_squares = set()
        
        for pawn_square in pawn_squares:
            # Get attack squares for this pawn
            rank = chess.square_rank(pawn_square)
            file = chess.square_file(pawn_square)
            
            # Pawns attack diagonally forward
            if color == chess.WHITE:
                # White pawns attack up-left and up-right
                attack_squares = []
                if rank < 7:  # Not on the last rank
                    if file > 0:  # Not on a-file
                        attack_squares.append(chess.square(file - 1, rank + 1))
                    if file < 7:  # Not on h-file
                        attack_squares.append(chess.square(file + 1, rank + 1))
            else:
                # Black pawns attack down-left and down-right
                attack_squares = []
                if rank > 0:  # Not on the first rank
                    if file > 0:  # Not on a-file
                        attack_squares.append(chess.square(file - 1, rank - 1))
                    if file < 7:  # Not on h-file
                        attack_squares.append(chess.square(file + 1, rank - 1))
            
            # Add attack squares to the set
            controlled_squares.update(attack_squares)
        
        # Return average number of squares controlled per pawn
        return len(controlled_squares) / pawn_count
        
    def _calculate_development_metrics(self, positions: List[chess.Board], moves: List[chess.Move]) -> Tuple[float, float, float, float]:
        """
        Calculate development metrics for both players.
        
        Args:
            positions: List of board positions
            moves: List of moves played in the game
            
        Returns:
            Tuple of development metrics for white and black
        """
        if not positions or len(positions) < 2 or not moves:
            return 0.0, 0.0, 0.0, 0.0
        
        # Initial position
        initial_board = chess.Board()
        
        # Track development of pieces
        white_pieces_developed = {
            'knight_kingside': False,
            'knight_queenside': False,
            'bishop_kingside': False,
            'bishop_queenside': False,
            'queen': False
        }
        
        black_pieces_developed = {
            'knight_kingside': False,
            'knight_queenside': False,
            'bishop_kingside': False,
            'bishop_queenside': False,
            'queen': False
        }
        
        # Track when all minor pieces are developed
        white_all_minor_pieces_move = 0
        black_all_minor_pieces_move = 0
        
        # Track first queen move
        white_queen_move = 0
        black_queen_move = 0
        
        # Process each move and position
        for move_idx, (move, board) in enumerate(zip(moves, positions[1:])):  # Skip initial position
            move_number = move_idx // 2 + 1  # Convert to move number (1-indexed)
            is_white_move = move_idx % 2 == 0
            
            # Get piece type and from/to squares
            piece = board.piece_at(move.to_square)
            if not piece:
                continue
                
            piece_type = piece.piece_type
            from_square = move.from_square
            to_square = move.to_square
            
            # Process development based on piece type and color
            if is_white_move:
                # White's move
                if piece_type == chess.KNIGHT:
                    # Determine kingside/queenside knight
                    from_file = chess.square_file(from_square)
                    
                    # Knight development is moving from starting rank to a new position
                    if chess.square_rank(from_square) == 0:
                        if from_file == 1:  # b1 (queenside knight)
                            white_pieces_developed['knight_queenside'] = True
                        elif from_file == 6:  # g1 (kingside knight)
                            white_pieces_developed['knight_kingside'] = True
                
                elif piece_type == chess.BISHOP:
                    # Determine kingside/queenside bishop
                    from_file = chess.square_file(from_square)
                    
                    # Bishop development is moving from starting rank
                    if chess.square_rank(from_square) == 0:
                        if from_file == 2:  # c1 (queenside bishop)
                            white_pieces_developed['bishop_queenside'] = True
                        elif from_file == 5:  # f1 (kingside bishop)
                            white_pieces_developed['bishop_kingside'] = True
                
                elif piece_type == chess.QUEEN:
                    # Track first queen move
                    if not white_pieces_developed['queen']:
                        white_pieces_developed['queen'] = True
                        white_queen_move = move_number
            else:
                # Black's move
                if piece_type == chess.KNIGHT:
                    # Determine kingside/queenside knight
                    from_file = chess.square_file(from_square)
                    
                    # Knight development is moving from starting rank to a new position
                    if chess.square_rank(from_square) == 7:
                        if from_file == 1:  # b8 (queenside knight)
                            black_pieces_developed['knight_queenside'] = True
                        elif from_file == 6:  # g8 (kingside knight)
                            black_pieces_developed['knight_kingside'] = True
                
                elif piece_type == chess.BISHOP:
                    # Determine kingside/queenside bishop
                    from_file = chess.square_file(from_square)
                    
                    # Bishop development is moving from starting rank
                    if chess.square_rank(from_square) == 7:
                        if from_file == 2:  # c8 (queenside bishop)
                            black_pieces_developed['bishop_queenside'] = True
                        elif from_file == 5:  # f8 (kingside bishop)
                            black_pieces_developed['bishop_kingside'] = True
                
                elif piece_type == chess.QUEEN:
                    # Track first queen move
                    if not black_pieces_developed['queen']:
                        black_pieces_developed['queen'] = True
                        black_queen_move = move_number
            
            # Check if all minor pieces are developed
            white_minor_pieces = [
                white_pieces_developed['knight_kingside'],
                white_pieces_developed['knight_queenside'],
                white_pieces_developed['bishop_kingside'],
                white_pieces_developed['bishop_queenside']
            ]
            
            black_minor_pieces = [
                black_pieces_developed['knight_kingside'],
                black_pieces_developed['knight_queenside'],
                black_pieces_developed['bishop_kingside'],
                black_pieces_developed['bishop_queenside']
            ]
            
            if all(white_minor_pieces) and white_all_minor_pieces_move == 0:
                white_all_minor_pieces_move = move_number
            
            if all(black_minor_pieces) and black_all_minor_pieces_move == 0:
                black_all_minor_pieces_move = move_number
        
        # If pieces were never developed, use a high value (e.g., total moves + 10)
        total_moves = len(moves) // 2
        if white_all_minor_pieces_move == 0:
            white_all_minor_pieces_move = total_moves + 10
        if black_all_minor_pieces_move == 0:
            black_all_minor_pieces_move = total_moves + 10
        
        # If queen was never moved, set to a high value
        if white_queen_move == 0:
            white_queen_move = total_moves + 10
        if black_queen_move == 0:
            black_queen_move = total_moves + 10
        
        return (
            white_all_minor_pieces_move, black_all_minor_pieces_move,
            white_queen_move, black_queen_move
        )
        
    def _calculate_engine_move_alignment(self, moves: List[chess.Move], evals: List[Info]) -> Tuple[float, float, float, float]:
        """
        Calculate how often each player follows the top engine moves.
        
        Args:
            moves: List of moves played in the game
            evals: List of position evaluations with engine recommendations
            
        Returns:
            Tuple of (top_move_alignment_white, top_move_alignment_black, 
                     top2_3_move_alignment_white, top2_3_move_alignment_black)
        """
        if not moves or not evals or len(evals) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        # Initialize counters
        white_top_move_count = 0
        black_top_move_count = 0
        white_top2_3_move_count = 0
        black_top2_3_move_count = 0
        
        white_move_count = 0
        black_move_count = 0
        
        # Skip first evaluation as we need the previous position's evaluation
        for i in range(len(moves)):
            # Skip if we don't have evaluation data for this position
            if i >= len(evals) - 1:
                break
                
            move = moves[i]
            eval_info = evals[i]  # Evaluation before the move
            
            # Skip if no variation data (engine recommendations)
            if not eval_info.variation or not eval_info.multipv:
                continue
            
            is_white = i % 2 == 0  # Even indices are White's moves
            
            # Get the move in UCI format for comparison
            move_uci = move.uci()
            
            # Get top engine moves from multipv
            top_moves = []
            for mv in eval_info.multipv[:min(3, len(eval_info.multipv))]:
                if "move" in mv:
                    top_moves.append(mv["move"])
            
            # If we don't have enough top moves, use the variation
            if len(top_moves) < 3 and len(eval_info.variation) >= 3:
                top_moves = eval_info.variation[:3]
            
            # If we still don't have enough top moves, skip
            if not top_moves:
                continue
            
            # Check if the played move matches the top engine move(s)
            if is_white:
                white_move_count += 1
                
                if len(top_moves) > 0 and move_uci == top_moves[0]:
                    white_top_move_count += 1
                elif len(top_moves) > 1 and move_uci == top_moves[1] or len(top_moves) > 2 and move_uci == top_moves[2]:
                    white_top2_3_move_count += 1
            else:
                black_move_count += 1
                
                if len(top_moves) > 0 and move_uci == top_moves[0]:
                    black_top_move_count += 1
                elif len(top_moves) > 1 and move_uci == top_moves[1] or len(top_moves) > 2 and move_uci == top_moves[2]:
                    black_top2_3_move_count += 1
        
        # Calculate percentages
        top_move_alignment_white = white_top_move_count / white_move_count if white_move_count > 0 else 0.0
        top_move_alignment_black = black_top_move_count / black_move_count if black_move_count > 0 else 0.0
        top2_3_move_alignment_white = white_top2_3_move_count / white_move_count if white_move_count > 0 else 0.0
        top2_3_move_alignment_black = black_top2_3_move_count / black_move_count if black_move_count > 0 else 0.0
        
        return (
            top_move_alignment_white, top_move_alignment_black,
            top2_3_move_alignment_white, top2_3_move_alignment_black
        )
        
    def _calculate_eval_changes(self, evals: List[Info], features: FeatureVector, moves: List[chess.Move]) -> None:
        """Calculate evaluation changes for both players"""
        white_eval_changes = []
        black_eval_changes = []
        
        # Skip first evaluation as we need pairs
        for i in range(1, len(evals)):
            if i-1 >= len(moves):
                break
                
            prev, curr = evals[i-1], evals[i]
            is_white = (i - 1) % 2 == 0  # Even indices are White's moves
            
            # Handle both CP and mate scores
            if prev.cp is not None and curr.cp is not None:
                # Both positions have CP scores
                eval_change = curr.cp - prev.cp
            elif prev.mate is not None or curr.mate is not None:
                # At least one position has a mate score - convert to CP
                prev_cp = 10000 if prev.mate and prev.mate > 0 else -10000 if prev.mate else (prev.cp or 0)
                curr_cp = 10000 if curr.mate and curr.mate > 0 else -10000 if curr.mate else (curr.cp or 0)
                eval_change = curr_cp - prev_cp
            else:
                # Skip if no score available
                continue
                
            # For Black's moves, negate the change to get their perspective
            if not is_white:
                eval_change = -eval_change
                black_eval_changes.append(eval_change)
            else:
                white_eval_changes.append(eval_change)
        
        # Calculate evaluation metrics
        if white_eval_changes:
            features.white_avg_eval_change = float(np.mean(np.abs(white_eval_changes)))
            features.white_eval_volatility = float(np.std(white_eval_changes))
        
        if black_eval_changes:
            features.black_avg_eval_change = float(np.mean(np.abs(black_eval_changes)))
            features.black_eval_volatility = float(np.std(black_eval_changes))
    
    def _calculate_quality_metrics(self, evals: List[Info], features: FeatureVector, 
                                  positions: List[chess.Board] = None, 
                                  moves: List[chess.Move] = None) -> None:
        """
        Calculate move quality related features and statistics for both players
        
        Args:
            evals: List of position evaluations
            features: FeatureVector to update
            positions: List of board positions
            moves: List of moves played
        """
        if not positions or not moves or len(evals) < 2:
            return
            
        white_counts = {judgment: 0 for judgment in Judgment}
        black_counts = {judgment: 0 for judgment in Judgment}
        
        # Initialize sacrifice counters
        white_sacrifices = 0
        black_sacrifices = 0
        
        # Skip first evaluation as we need pairs to analyze moves
        for i in range(1, len(evals)):
            if i-1 >= len(moves):
                break
                
            prev, curr = evals[i-1], evals[i]
            is_white = (i - 1) % 2 == 0  # Even indices are White's moves
            
            # Get move quality with additional board information
            if i-1 < len(moves) and i-1 < len(positions) and i < len(positions):
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
                # Use basic analysis if board/move info not available
                judgment = MoveAnalyzer.analyze_move(prev, curr)
            
            # Increment the appropriate counter
            if is_white:
                white_counts[judgment] += 1
            else:
                black_counts[judgment] += 1
            
        
        # Set judgment counts
        features.white_brilliant_count = white_counts[Judgment.BRILLIANT]
        features.white_great_count = white_counts[Judgment.GREAT]
        features.white_good_moves = white_counts[Judgment.GOOD]
        features.white_inaccuracy_count = white_counts[Judgment.INACCURACY]
        features.white_mistake_count = white_counts[Judgment.MISTAKE]
        features.white_blunder_count = white_counts[Judgment.BLUNDER]
        features.white_sacrifice_count = white_sacrifices
        
        features.black_brilliant_count = black_counts[Judgment.BRILLIANT]
        features.black_great_count = black_counts[Judgment.GREAT]
        features.black_good_moves = black_counts[Judgment.GOOD]
        features.black_inaccuracy_count = black_counts[Judgment.INACCURACY]
        features.black_mistake_count = black_counts[Judgment.MISTAKE]
        features.black_blunder_count = black_counts[Judgment.BLUNDER]
        features.black_sacrifice_count = black_sacrifices

    def _calculate_move_statistics(self, positions: List[chess.Board], moves: List[chess.Move]) -> Tuple[float, float, float, float, int, int]:
        """
        Calculate move-related statistics including capture frequency, check frequency,
        and castle timing for both players separately.
        
        Args:
            positions: List of board positions (positions[i] is the board before moves[i])
            moves: List of moves played
            
        Returns:
            Tuple of (capture_frequency_white, capture_frequency_black, 
                    check_frequency_white, check_frequency_black, 
                    castle_move_white, castle_move_black)
        """
        if not positions or not moves:
            return 0.0, 0.0, 0.0, 0.0, 0, 0
        
        # Convert moves to a list if it's not already one
        if not isinstance(moves, list):
            moves = list(moves)
                
        total_white_moves = 0
        total_black_moves = 0
        white_capture_count = 0
        black_capture_count = 0
        white_check_count = 0
        black_check_count = 0
        white_castle_move = 0  # Changed to 0 for never castled
        black_castle_move = 0  # Changed to 0 for never castled
        
        for i, move in enumerate(moves):
            if i >= len(positions):
                break
                    
            board = positions[i]
            is_white = board.turn == chess.WHITE
            
            # Count total moves by color
            if is_white:
                total_white_moves += 1
            else:
                total_black_moves += 1
                
            # Check for captures (including en passant)
            is_capture = board.piece_at(move.to_square) is not None
            is_en_passant = False
            
            # Check for en passant if it's a pawn move
            if board.piece_type_at(move.from_square) == chess.PAWN:
                if board.ep_square == move.to_square:
                    is_en_passant = True
            
            if is_capture or is_en_passant:
                if is_white:
                    white_capture_count += 1
                else:
                    black_capture_count += 1
            
            # Check for castling
            if board.piece_type_at(move.from_square) == chess.KING:
                # Check if it's a castling move (king moves two squares)
                file_diff = abs(chess.square_file(move.from_square) - chess.square_file(move.to_square))
                if file_diff > 1:  # King moved more than one file, must be castling
                    move_number = (i // 2) + 1  # Convert to move number (1-based)
                    if is_white:
                        white_castle_move = move_number
                    else:
                        black_castle_move = move_number
            
            # Make a copy of the board and apply the move to check for check
            next_board = board.copy()
            next_board.push(move)
            
            if next_board.is_check():
                if is_white:
                    white_check_count += 1
                else:
                    black_check_count += 1
        
        # Calculate frequencies
        capture_frequency_white = white_capture_count / total_white_moves if total_white_moves > 0 else 0.0
        capture_frequency_black = black_capture_count / total_black_moves if total_black_moves > 0 else 0.0
        check_frequency_white = white_check_count / total_white_moves if total_white_moves > 0 else 0.0
        check_frequency_black = black_check_count / total_black_moves if total_black_moves > 0 else 0.0
        
        # Return actual move numbers for castling (not normalized)
        return (capture_frequency_white, capture_frequency_black, 
                check_frequency_white, check_frequency_black, 
                white_castle_move, black_castle_move)
    
    def count_sacrifices(self, positions: List[chess.Board], moves: List[chess.Move], features: FeatureVector) -> None:
        """
        Count sacrifices for both players
        
        Args:
            positions: List of board positions
            moves: List of moves played
            features: FeatureVector object to update
        """
        # Convert moves to a list if it's not already one
        if not isinstance(moves, list):
            moves = list(moves)
            
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