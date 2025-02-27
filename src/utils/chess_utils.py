import chess
from typing import Set, List, Tuple

def get_piece_squares(board: chess.Board, piece_type: chess.PieceType, 
                     color: chess.Color) -> Set[chess.Square]:
    """
    Get squares occupied by specific piece type
    
    Args:
        board: Chess board to examine
        piece_type: Type of piece to find (e.g. chess.PAWN)
        color: Color of pieces to find (chess.WHITE or chess.BLACK)
        
    Returns:
        Set of squares containing pieces of specified type and color
    """
    return board.pieces(piece_type, color)

def calculate_piece_mobility(board: chess.Board, square: chess.Square) -> int:
    """
    Calculate number of legal moves for piece on a specific square
    
    Args:
        board: Chess board to analyze
        square: Square to check for piece mobility
        
    Returns:
        Number of legal moves for the piece
    """
    piece = board.piece_at(square)
    if not piece:
        return 0
        
    moves = 0
    for move in board.legal_moves:
        if move.from_square == square:
            moves += 1
    return moves

def is_center_square(square: chess.Square) -> bool:
    """
    Check if square is in center (d4,d5,e4,e5)
    
    Args:
        square: Chess square to check
        
    Returns:
        True if square is in the center, False otherwise
    """
    center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
    return square in center_squares

def get_game_phases(moves: int, mg_start: int, eg_start: int) -> List[float]:
    """
    Calculate phase lengths as percentages
    
    Args:
        moves: Total number of moves in the game
        mg_start: Move number where middlegame starts
        eg_start: Move number where endgame starts
        
    Returns:
        List of [opening_pct, middlegame_pct, endgame_pct]
    """
    if moves == 0:
        return [0, 0, 0]
    
    opening = mg_start / moves
    middlegame = (eg_start - mg_start) / moves
    endgame = (moves - eg_start) / moves
    
    return [opening, middlegame, endgame]

def count_piece_types(board: chess.Board) -> Tuple[int, int]:
    """
    Count the number of pieces of each type on the board
    
    Args:
        board: Chess board to analyze
        
    Returns:
        Tuple of (white_piece_count, black_piece_count)
    """
    white_pieces = 0
    black_pieces = 0
    
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        white_pieces += len(board.pieces(piece_type, chess.WHITE))
        black_pieces += len(board.pieces(piece_type, chess.BLACK))
    
    return white_pieces, black_pieces

def is_developed(board: chess.Board, color: chess.Color) -> bool:
    """
    Check if pieces are developed (minor pieces moved, castled, etc.)
    
    Args:
        board: Chess board to analyze
        color: Color to check development for
        
    Returns:
        True if pieces are developed, False otherwise
    """
    if color == chess.WHITE:
        # Check if knights and bishops moved from starting squares
        knight_developed = (not board.piece_at(chess.B1) or board.piece_at(chess.B1).piece_type != chess.KNIGHT) and \
                          (not board.piece_at(chess.G1) or board.piece_at(chess.G1).piece_type != chess.KNIGHT)
        bishop_developed = (not board.piece_at(chess.C1) or board.piece_at(chess.C1).piece_type != chess.BISHOP) and \
                          (not board.piece_at(chess.F1) or board.piece_at(chess.F1).piece_type != chess.BISHOP)
        # Check if king has castled
        king_castled = board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE) or \
                      board.piece_at(chess.C1) == chess.Piece(chess.KING, chess.WHITE)
        
        return knight_developed and bishop_developed and king_castled
    else:
        # Check if knights and bishops moved from starting squares
        knight_developed = (not board.piece_at(chess.B8) or board.piece_at(chess.B8).piece_type != chess.KNIGHT) and \
                          (not board.piece_at(chess.G8) or board.piece_at(chess.G8).piece_type != chess.KNIGHT)
        bishop_developed = (not board.piece_at(chess.C8) or board.piece_at(chess.C8).piece_type != chess.BISHOP) and \
                          (not board.piece_at(chess.F8) or board.piece_at(chess.F8).piece_type != chess.BISHOP)
        # Check if king has castled
        king_castled = board.piece_at(chess.G8) == chess.Piece(chess.KING, chess.BLACK) or \
                      board.piece_at(chess.C8) == chess.Piece(chess.KING, chess.BLACK)
        
        return knight_developed and bishop_developed and king_castled
