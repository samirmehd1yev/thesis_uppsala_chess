import chess
from typing import List, Tuple

class PieceMobility:
    """
    A class that calculates piece mobility using a method inspired by Stockfish.
    """
    
    def __init__(self):
        # Mobility bonus values from Stockfish (middlegame values)
        self.mobility_bonuses = {
            chess.KNIGHT: [-62, -53, -12, -3, 3, 12, 21, 28, 37],
            chess.BISHOP: [-47, -20, 14, 29, 39, 53, 53, 60, 62, 69, 78, 83, 91, 96],
            chess.ROOK: [-60, -24, 0, 3, 4, 14, 20, 30, 41, 41, 41, 45, 57, 58, 67],
            chess.QUEEN: [-29, -16, -8, -8, 18, 25, 23, 37, 41, 54, 65, 68, 69, 70, 70, 70, 71, 72, 
                         74, 76, 90, 104, 105, 106, 112, 114, 114, 119]
        }
    
    def calculate_mobility_by_color(self, positions: List[chess.Board]) -> Tuple[float, float]:
        """
        Calculate average piece mobility separately for white and black.
        
        Args:
            positions: List of board positions
        Returns:
            Tuple of (white_mobility_avg, black_mobility_avg)
        """
        if not positions:
            return 0.0, 0.0
            
        white_mobility_total = 0.0
        black_mobility_total = 0.0
        white_positions = 0
        black_positions = 0
        
        for board in positions:
            white_score = self.calculate_mobility_score(board, chess.WHITE)
            black_score = self.calculate_mobility_score(board, chess.BLACK)
            
            if board.turn == chess.WHITE:
                white_mobility_total += white_score
                white_positions += 1
            else:
                black_mobility_total += black_score
                black_positions += 1
        
        white_avg = white_mobility_total / white_positions if white_positions > 0 else 0
        black_avg = black_mobility_total / black_positions if black_positions > 0 else 0
        
        return white_avg, black_avg
    
    def calculate_mobility_score(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate mobility score for a specific color on a given board position.
        
        Args:
            board: Chess board
            color: Color to calculate mobility for
        Returns:
            Mobility score
        """
        enemy_color = not color
        mobility_score = 0
        
        # Calculate mobility area - squares where pieces can move safely
        mobility_area = self._calculate_mobility_area(board, color, enemy_color)
        
        # Calculate mobility for each piece type
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                # Get attacks for this piece considering x-ray for sliding pieces
                attacks = self._get_piece_attacks(board, square, piece_type, color)
                
                # Count attacks that fall within mobility area
                mobility_count = bin(attacks & mobility_area).count('1')
                
                # Cap mobility count to the maximum index in our bonus array
                mobility_index = min(mobility_count, len(self.mobility_bonuses[piece_type]) - 1)
                
                # Add the bonus for this piece
                mobility_score += self.mobility_bonuses[piece_type][mobility_index]
        
        return mobility_score
    
    def _calculate_mobility_area(self, board: chess.Board, color: chess.Color, enemy_color: chess.Color) -> int:
        """
        Calculate the mobility area (squares where pieces can move safely).
        
        Args:
            board: Chess board
            color: Side to calculate mobility area for
            enemy_color: Opposing side
        Returns:
            Bitboard representing the mobility area
        """
        # Start with all squares
        mobility_area = chess.BB_ALL
        
        # 1. Find blocked pawns and pawns on the first two ranks
        blocked_pawns = 0
        low_ranks = chess.BB_RANK_1 | chess.BB_RANK_2 if color == chess.WHITE else chess.BB_RANK_7 | chess.BB_RANK_8
        
        for pawn_square in board.pieces(chess.PAWN, color):
            # Check if pawn is on the first two ranks
            if (1 << pawn_square) & low_ranks:
                blocked_pawns |= 1 << pawn_square
                continue
                
            # Check if pawn is blocked
            front_square = pawn_square + 8 if color == chess.WHITE else pawn_square - 8
            if 0 <= front_square < 64 and board.piece_at(front_square):
                blocked_pawns |= 1 << pawn_square
        
        # 2. Exclude squares with king and queen
        king_queen_squares = board.pieces_mask(chess.KING, color) | board.pieces_mask(chess.QUEEN, color)
        
        # 3. Exclude squares with pinned pieces
        pinned_pieces = 0
        king_square = board.king(color)
        if king_square is not None:
            for square in range(64):
                if board.piece_at(square) and board.piece_at(square).color == color and board.is_pinned(color, square):
                    pinned_pieces |= 1 << square
        
        # 4. Exclude squares attacked by enemy pawns
        enemy_pawn_attacks = 0
        for pawn_square in board.pieces(chess.PAWN, enemy_color):
            if enemy_color == chess.WHITE:
                # White pawn attacks
                if pawn_square % 8 > 0:  # Not on a-file
                    enemy_pawn_attacks |= 1 << (pawn_square + 7)
                if pawn_square % 8 < 7:  # Not on h-file
                    enemy_pawn_attacks |= 1 << (pawn_square + 9)
            else:
                # Black pawn attacks
                if pawn_square % 8 > 0:  # Not on a-file
                    enemy_pawn_attacks |= 1 << (pawn_square - 9)
                if pawn_square % 8 < 7:  # Not on h-file
                    enemy_pawn_attacks |= 1 << (pawn_square - 7)
        
        # Remove all excluded squares from the mobility area
        mobility_area &= ~blocked_pawns
        mobility_area &= ~king_queen_squares
        mobility_area &= ~pinned_pieces
        mobility_area &= ~enemy_pawn_attacks
        
        return mobility_area
    
    def _get_piece_attacks(self, board: chess.Board, square: int, piece_type: int, color: chess.Color) -> int:
        """
        Get attacks for a specific piece with x-ray capability for sliding pieces.
        
        Args:
            board: Chess board
            square: Square the piece is on
            piece_type: Type of piece
            color: Color of the piece
        Returns:
            Bitboard of attacked squares
        """
        if piece_type == chess.KNIGHT:
            return board.attacks_mask(square)
        
        elif piece_type == chess.BISHOP:
            # Handle bishop x-ray attacks through queens
            attacks = 0
            for direction in [7, 9, -7, -9]:  # NW, NE, SW, SE
                current = square
                while True:
                    current += direction
                    if not 0 <= current < 64:  # Off the board
                        break
                    if abs(chess.square_file(current) - chess.square_file(current - direction)) != 1:
                        break  # Not a diagonal move
                    
                    attacks |= 1 << current
                    
                    piece = board.piece_at(current)
                    if piece:
                        if piece.piece_type == chess.QUEEN:
                            continue  # X-ray through queens
                        break  # Stop at other pieces
                        
            return attacks
                
        elif piece_type == chess.ROOK:
            # Handle rook x-ray attacks through queens and own rooks
            attacks = 0
            for direction in [8, -8, 1, -1]:  # N, S, E, W
                current = square
                while True:
                    current += direction
                    if not 0 <= current < 64:  # Off the board
                        break
                    if direction in [1, -1] and chess.square_rank(current) != chess.square_rank(current - direction):
                        break  # Not on same rank for horizontal moves
                    
                    attacks |= 1 << current
                    
                    piece = board.piece_at(current)
                    if piece:
                        if piece.piece_type == chess.QUEEN or (piece.piece_type == chess.ROOK and piece.color == color):
                            continue  # X-ray through queens and own rooks
                        break  # Stop at other pieces
                        
            return attacks
                
        elif piece_type == chess.QUEEN:
            # Combine bishop and rook attacks
            bishop_attacks = self._get_piece_attacks(board, square, chess.BISHOP, color)
            rook_attacks = self._get_piece_attacks(board, square, chess.ROOK, color)
            return bishop_attacks | rook_attacks
        
        # Restrict moves of pinned pieces to the ray between king and attacker
        king_square = board.king(color)
        if king_square is not None and board.is_pinned(color, square):
            attacks = board.attacks_mask(square)
            # Get the ray between king and piece
            ray = 0
            
            # Check if king and piece are aligned
            kf, kr = chess.square_file(king_square), chess.square_rank(king_square)
            pf, pr = chess.square_file(square), chess.square_rank(square)
            
            if kf == pf:  # Same file
                ray = self._get_file_ray(king_square, square)
            elif kr == pr:  # Same rank
                ray = self._get_rank_ray(king_square, square)
            elif abs(kf - pf) == abs(kr - pr):  # Same diagonal
                ray = self._get_diagonal_ray(king_square, square)
                
            # Restrict attacks to the ray
            return attacks & ray
            
        return board.attacks_mask(square)
    
    def _get_file_ray(self, sq1: int, sq2: int) -> int:
        """Get all squares on the same file between and beyond sq1 and sq2"""
        ray = 0
        file = chess.square_file(sq1)
        
        for rank in range(8):
            ray |= 1 << chess.square(file, rank)
            
        return ray
    
    def _get_rank_ray(self, sq1: int, sq2: int) -> int:
        """Get all squares on the same rank between and beyond sq1 and sq2"""
        ray = 0
        rank = chess.square_rank(sq1)
        
        for file in range(8):
            ray |= 1 << chess.square(file, rank)
            
        return ray
    
    def _get_diagonal_ray(self, sq1: int, sq2: int) -> int:
        """Get all squares on the same diagonal between and beyond sq1 and sq2"""
        ray = 0
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
        
        # Determine diagonal direction
        f_dir = 1 if f2 > f1 else -1
        r_dir = 1 if r2 > r1 else -1
        
        # Add all squares along the diagonal
        f, r = 0, 0
        while 0 <= f < 8 and 0 <= r < 8:
            ray |= 1 << chess.square(f, r)
            f += f_dir
            r += r_dir
            
        # Start from the other end and go in the opposite direction
        f, r = 7, 7
        if f_dir == 1 and r_dir == -1:
            f, r = 7, 0
        elif f_dir == -1 and r_dir == 1:
            f, r = 0, 7
        elif f_dir == -1 and r_dir == -1:
            f, r = 0, 0
            
        while 0 <= f < 8 and 0 <= r < 8:
            ray |= 1 << chess.square(f, r)
            f -= f_dir
            r -= r_dir
            
        return ray