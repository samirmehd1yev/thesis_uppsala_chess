import chess
from typing import Dict, List, Tuple, Set, Optional
import math

class KingSafetyEvaluator:
    """
    Evaluates king safety using multiple methods described in chess programming literature.
    Implements concepts from Chess Programming Wiki:
    - Pawn Shield evaluation
    - Pawn Storm detection
    - King Tropism (distance of attacking pieces)
    - King Zone attacks (counting attacks on squares around the king)
    - Attack Units with scaling
    """
    
    # Constants for king safety evaluation
    PAWN_SHIELD_VALUES = {
        # Values for pawns on the same rank and files around the king
        # Format: (file_distance, rank_distance): value
        (0, 0): 0,      # Same square as king (impossible)
        (0, 1): 40,     # Pawn directly in front of king
        (1, 1): 30,     # Pawn diagonally in front
        (-1, 1): 30,    # Pawn diagonally in front (other side)
        (1, 0): 10,     # Pawn beside king
        (-1, 0): 10,    # Pawn beside king (other side)
        (0, 2): 15,     # Pawn two squares in front
        (1, 2): 10,     # Pawn two squares in front, one to the side
        (-1, 2): 10,    # Pawn two squares in front, one to the side (other)
    }
    
    # Penalties for missing shield pawns based on king position
    MISSING_SHIELD_PENALTY = {
        'kingside': {
            'f': 15,     # Missing f pawn
            'g': 30,     # Missing g pawn
            'h': 20,     # Missing h pawn
        },
        'queenside': {
            'a': 20,     # Missing a pawn
            'b': 30,     # Missing b pawn
            'c': 15,     # Missing c pawn
        }
    }
    
    # Penalties for open files near the king
    OPEN_FILE_PENALTIES = {
        0: 50,   # Open file where king is
        1: 25,   # Open file 1 square away
        2: 10,   # Open file 2 squares away
    }
    
    # Piece tropism weights (for calculating distance-based threats)
    TROPISM_WEIGHTS = {
        chess.PAWN: 0,      # Pawns not considered for tropism
        chess.KNIGHT: 3,    # Knights are dangerous up close
        chess.BISHOP: 2,    # Bishops can be dangerous from a distance
        chess.ROOK: 4,      # Rooks are very dangerous on open files
        chess.QUEEN: 6,     # Queens are most dangerous
        chess.KING: 0,      # Kings not considered for tropism
    }
    
    # Attack unit values for different pieces
    ATTACK_UNIT_VALUES = {
        chess.PAWN: 0,      # Pawns don't count for attack units
        chess.KNIGHT: 2,    # Knights contribute 2 attack units
        chess.BISHOP: 2,    # Bishops contribute 2 attack units
        chess.ROOK: 3,      # Rooks contribute 3 attack units
        chess.QUEEN: 5,     # Queens contribute 5 attack units
    }
    
    # Safety table for converting attack units to score
    # This is similar to the table from Glaurung/Stockfish
    SAFETY_TABLE = [
        0,   0,   1,   2,   3,   5,   7,   9,  12,  15,
        18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
        68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
        140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
        260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
        377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
        494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500
    ]
    
    def __init__(self):
        """Initialize the KingSafetyEvaluator with pre-calculated data"""
        # Cache of king zone squares (squares around king)
        self.king_zone_cache = {}
        
    def evaluate_king_safety(self, board: chess.Board, color: chess.Color) -> int:
        """
        Main function to evaluate king safety for a given side.
        Returns a positive score for good king safety, negative for poor safety.
        
        Args:
            board: Chess board position
            color: Color to evaluate king safety for
        
        Returns:
            Score representing king safety (higher is safer)
        """
        # Get the king square
        king_square = board.king(color)
        if king_square is None:
            return 0  # No king found (shouldn't happen in normal chess)
        
        # Get the opponent color
        opponent = not color
        
        # Get king zone (squares around the king)
        king_zone = self.get_king_zone(king_square)
        
        # Calculate different king safety factors
        shield_score = self.evaluate_pawn_shield(board, king_square, color)
        storm_score = self.evaluate_pawn_storm(board, king_square, color)
        tropism_score = self.evaluate_king_tropism(board, king_square, opponent)
        attack_score = self.evaluate_king_zone_attacks(board, king_zone, opponent)
        open_file_score = self.evaluate_open_files_near_king(board, king_square, color)
        
        # Combine scores with appropriate weights
        # The weights can be adjusted based on testing and engine preferences
        safety_score = (
            shield_score * 1.0 +       # Pawn shield is important
            storm_score * 0.8 +        # Pawn storm is a potential threat
            open_file_score * 1.2 +    # Open files are very dangerous
            tropism_score * 0.5 +      # Piece distance is a moderate factor
            attack_score * 1.5         # Attacks are highly dangerous
        )
        
        # Scale based on material balance - fewer opponent pieces means less danger
        material_scale = self.get_material_scale(board, opponent)
        safety_score = int(safety_score * material_scale)
        
        return safety_score
        
    def get_king_zone(self, king_square: chess.Square) -> Set[chess.Square]:
        """
        Get the 'king zone' - squares that are within 2 squares of the king.
        This includes squares the king can move to and additional squares
        in front of the king (facing his starting position).
        
        Args:
            king_square: Square where the king is located
            
        Returns:
            Set of squares in the king zone
        """
        # Check cache first
        if king_square in self.king_zone_cache:
            return self.king_zone_cache[king_square]
            
        # Calculate king zone
        king_zone = set()
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Add squares in a 3x3 box around the king
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            for r in range(max(0, king_rank - 1), min(8, king_rank + 2)):
                king_zone.add(chess.square(f, r))
        
        # Add additional forward squares based on king's rank
        if king_rank < 6:  # Not near the last rank
            # Add forward squares (assuming white, reversed for black)
            forward_rank = king_rank + 2
            if 0 <= forward_rank < 8:
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    king_zone.add(chess.square(f, forward_rank))
        
        # Store in cache and return
        self.king_zone_cache[king_square] = king_zone
        return king_zone
    
    def evaluate_pawn_shield(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        """
        Evaluate the pawn shield in front of the king.
        
        Args:
            board: Chess board position
            king_square: Square where the king is located
            color: Color of the king
            
        Returns:
            Score for pawn shield (higher is better)
        """
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        shield_score = 0
        
        # Determine if king has castled and what side
        castling_side = 'none'
        if color == chess.WHITE:
            if king_file >= 5:
                castling_side = 'kingside'
            elif king_file <= 3:
                castling_side = 'queenside'
        else:  # BLACK
            if king_file >= 5:
                castling_side = 'kingside'
            elif king_file <= 3:
                castling_side = 'queenside'
        
        # If king hasn't castled, less shield penalty
        if castling_side == 'none':
            # For non-castled kings, evaluate nearby pawns
            for file_offset in range(-1, 2):
                for rank_offset in range(0, 3):
                    # Adjust rank direction based on color
                    actual_rank_offset = rank_offset if color == chess.WHITE else -rank_offset
                    
                    # Check if square is valid
                    if 0 <= king_file + file_offset < 8 and 0 <= king_rank + actual_rank_offset < 8:
                        check_square = chess.square(king_file + file_offset, king_rank + actual_rank_offset)
                        if board.piece_at(check_square) == chess.Piece(chess.PAWN, color):
                            shield_value = self.PAWN_SHIELD_VALUES.get((file_offset, rank_offset), 0)
                            shield_score += shield_value
            return shield_score
        
        # Evaluate castled king's pawn shield
        if castling_side == 'kingside':
            # Check kingside castling (f, g, h pawns)
            files_to_check = ['f', 'g', 'h']
            for file_str in files_to_check:
                file_idx = ord(file_str) - ord('a')
                
                # For white, check rank 2 and 3; for black, check rank 7 and 6
                if color == chess.WHITE:
                    shield_ranks = [1, 2]  # 0-indexed
                else:
                    shield_ranks = [6, 5]  # 0-indexed
                
                # Check if pawn exists in shield positions
                pawn_found = False
                for rank_idx in shield_ranks:
                    square = chess.square(file_idx, rank_idx)
                    if board.piece_at(square) == chess.Piece(chess.PAWN, color):
                        pawn_found = True
                        # Give bonus based on file and rank
                        file_distance = abs(file_idx - king_file)
                        rank_distance = abs(rank_idx - king_rank)
                        shield_value = self.PAWN_SHIELD_VALUES.get((file_distance, rank_distance), 0)
                        shield_score += shield_value
                        break
                
                # Penalty for missing shield pawn
                if not pawn_found:
                    shield_score -= self.MISSING_SHIELD_PENALTY['kingside'][file_str]
        
        elif castling_side == 'queenside':
            # Check queenside castling (a, b, c pawns)
            files_to_check = ['a', 'b', 'c']
            for file_str in files_to_check:
                file_idx = ord(file_str) - ord('a')
                
                # For white, check rank 2 and 3; for black, check rank 7 and 6
                if color == chess.WHITE:
                    shield_ranks = [1, 2]  # 0-indexed
                else:
                    shield_ranks = [6, 5]  # 0-indexed
                
                # Check if pawn exists in shield positions
                pawn_found = False
                for rank_idx in shield_ranks:
                    square = chess.square(file_idx, rank_idx)
                    if board.piece_at(square) == chess.Piece(chess.PAWN, color):
                        pawn_found = True
                        # Give bonus based on file and rank
                        file_distance = abs(file_idx - king_file)
                        rank_distance = abs(rank_idx - king_rank)
                        shield_value = self.PAWN_SHIELD_VALUES.get((file_distance, rank_distance), 0)
                        shield_score += shield_value
                        break
                
                # Penalty for missing shield pawn
                if not pawn_found:
                    shield_score -= self.MISSING_SHIELD_PENALTY['queenside'][file_str]
        
        return shield_score
    
    def evaluate_pawn_storm(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        """
        Evaluate pawn storms (enemy pawns advancing toward the king).
        
        Args:
            board: Chess board position
            king_square: Square where the king is located
            color: Color of the king
            
        Returns:
            Score for pawn storm threat (negative means danger)
        """
        storm_score = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        enemy_color = not color
        
        # Files to check around the king
        files_to_check = [
            max(0, king_file - 1),
            king_file,
            min(7, king_file + 1)
        ]
        
        # Check for enemy pawns in these files
        for file_idx in files_to_check:
            # Get distance of closest enemy pawn in this file
            enemy_pawn_distance = 8  # Default to maximum
            
            # Check from king's position toward enemy side
            if color == chess.WHITE:
                # White king: check upward
                for rank_idx in range(king_rank + 1, 8):
                    square = chess.square(file_idx, rank_idx)
                    if board.piece_at(square) == chess.Piece(chess.PAWN, enemy_color):
                        enemy_pawn_distance = rank_idx - king_rank
                        break
            else:
                # Black king: check downward
                for rank_idx in range(king_rank - 1, -1, -1):
                    square = chess.square(file_idx, rank_idx)
                    if board.piece_at(square) == chess.Piece(chess.PAWN, enemy_color):
                        enemy_pawn_distance = king_rank - rank_idx
                        break
            
            # Score based on distance (closer pawns are more threatening)
            if enemy_pawn_distance <= 3:  # Only consider pawns within 3 squares
                storm_penalty = max(0, 30 - enemy_pawn_distance * 10)  # Scale based on distance
                
                # Increase penalty if it's on the same file as the king
                if file_idx == king_file:
                    storm_penalty *= 1.5
                
                storm_score -= storm_penalty
        
        return storm_score
    
    def evaluate_open_files_near_king(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        """
        Evaluate the danger from open files near the king.
        
        Args:
            board: Chess board position
            king_square: Square where the king is located
            color: Color of the king
            
        Returns:
            Score for open file threats (negative means danger)
        """
        king_file = chess.square_file(king_square)
        open_file_score = 0
        
        # Check files from king_file-2 to king_file+2
        for file_offset in range(-2, 3):
            file_idx = king_file + file_offset
            
            # Skip if file is out of bounds
            if file_idx < 0 or file_idx > 7:
                continue
            
            # Check if file is open (no pawns)
            file_is_open = True
            file_is_semi_open = True
            
            # Check for any pawns on this file
            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)
                
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == color:
                        file_is_open = False
                    else:
                        file_is_semi_open = False
            
            # Calculate penalty based on file status and distance from king
            file_distance = abs(file_offset)
            
            if file_is_open:
                open_file_score -= self.OPEN_FILE_PENALTIES.get(file_distance, 0)
            elif file_is_semi_open:
                # Semi-open files are less dangerous
                open_file_score -= self.OPEN_FILE_PENALTIES.get(file_distance, 0) // 2
        
        return open_file_score
    
    def evaluate_king_tropism(self, board: chess.Board, king_square: chess.Square, opponent_color: chess.Color) -> int:
        """
        Evaluate king tropism (distance of enemy pieces to the king).
        
        Args:
            board: Chess board position
            king_square: Square where the king is located
            opponent_color: Color of the attacking pieces
            
        Returns:
            Score for king tropism (negative means danger)
        """
        tropism_score = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Calculate threat value from each enemy piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            # Skip empty squares and opponent's pieces
            if not piece or piece.color != opponent_color:
                continue
            
            # Skip pawns for tropism calculation
            if piece.piece_type == chess.PAWN:
                continue
            
            # Calculate Manhattan distance to the king
            file_distance = abs(chess.square_file(square) - king_file)
            rank_distance = abs(chess.square_rank(square) - king_rank)
            manhattan_distance = file_distance + rank_distance
            
            # Calculate threat based on piece type and distance
            piece_weight = self.TROPISM_WEIGHTS.get(piece.piece_type, 0)
            
            # Closer pieces are more threatening - use inverse distance
            if manhattan_distance > 0:
                threat_value = piece_weight * (10 / manhattan_distance)
                tropism_score -= threat_value
        
        return int(tropism_score)
    
    def evaluate_king_zone_attacks(self, board: chess.Board, king_zone: Set[chess.Square], opponent_color: chess.Color) -> int:
        """
        Evaluate attacks on the king zone (squares around the king).
        Uses the attack units and safety table approach from Stockfish/Glaurung.
        
        Args:
            board: Chess board position
            king_zone: Set of squares in the king zone
            opponent_color: Color of the attacking pieces
            
        Returns:
            Score for king zone attacks (negative means danger)
        """
        attack_units = 0
        num_attackers = 0
        checked = False
        
        # Check if king is in check
        if board.is_check():
            checked = True
            attack_units += 10  # Significant penalty for being in check
        
        # Count attacks on king zone squares
        for zone_square in king_zone:
            attackers = board.attackers(opponent_color, zone_square)
            
            for attacker_square in attackers:
                piece = board.piece_at(attacker_square)
                if piece:
                    # Count this piece as an attacker
                    num_attackers += 1
                    
                    # Add attack units based on piece type
                    attack_units += self.ATTACK_UNIT_VALUES.get(piece.piece_type, 0)
                    
                    # Check for additional threats
                    
                    # Knight check threat (knight is one move away from checking)
                    if piece.piece_type == chess.KNIGHT and zone_square == board.king(not opponent_color):
                        attack_units += 5
                        
                    # Check for safe checks (attacker is not defended)
                    if (piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP] and 
                            not board.is_attacked_by(not opponent_color, attacker_square)):
                        attack_units += 3
        
        # No real danger if fewer than 2 attackers or very low attack units
        if num_attackers < 2 or attack_units < 5:
            return 0
        
        # Use the safety table to get the final score
        safety_index = min(99, attack_units)
        attack_score = -self.SAFETY_TABLE[safety_index]
        
        # Increase penalty if in check
        if checked:
            attack_score = int(attack_score * 1.5)
        
        return attack_score
    
    def get_material_scale(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate a scaling factor based on remaining attacking material.
        Fewer attacking pieces means less danger.
        
        Args:
            board: Chess board position
            color: Color of attacking pieces
            
        Returns:
            Scaling factor (0.0 to 1.0)
        """
        # Calculate total attacking potential
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type == chess.QUEEN:
                    total_material += 9
                elif piece.piece_type == chess.ROOK:
                    total_material += 5
                elif piece.piece_type == chess.BISHOP or piece.piece_type == chess.KNIGHT:
                    total_material += 3
        
        # Scale between 0.1 (minimum) and 1.0 (maximum)
        # Full scaling at 15 points of material (Q + R + minor)
        material_scale = min(1.0, max(0.1, total_material / 15))
        
        return material_scale

    def visualize_king_safety(self, board: chess.Board) -> Dict[str, Dict[str, float]]:
        """
        Provide a detailed breakdown of king safety for both sides.
        Useful for debugging and understanding king safety factors.
        
        Args:
            board: Chess board position
            
        Returns:
            Dictionary with detailed safety metrics for both colors
        """
        result = {}
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
                
            color_name = "white" if color == chess.WHITE else "black"
            king_zone = self.get_king_zone(king_square)
            
            # Get individual components
            shield_score = self.evaluate_pawn_shield(board, king_square, color)
            storm_score = self.evaluate_pawn_storm(board, king_square, color)
            tropism_score = self.evaluate_king_tropism(board, king_square, not color)
            open_file_score = self.evaluate_open_files_near_king(board, king_square, color)
            attack_score = self.evaluate_king_zone_attacks(board, king_zone, not color)
            
            # Calculate total
            material_scale = self.get_material_scale(board, not color)
            total_score = int((shield_score + storm_score + open_file_score + tropism_score + attack_score) * material_scale)
            
            result[color_name] = {
                "shield_score": shield_score,
                "storm_score": storm_score,
                "open_file_score": open_file_score,
                "tropism_score": tropism_score,
                "attack_score": attack_score,
                "material_scale": material_scale,
                "total": total_score
            }
            
        return result