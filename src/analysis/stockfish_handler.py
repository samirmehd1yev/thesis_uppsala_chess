from typing import List, Dict, Optional
import chess
from stockfish import Stockfish
from models.data_classes import Info

class StockfishHandler:
    def __init__(self, path: str = "stockfish", depth: int = 16):
        self.stockfish = Stockfish(
            path=path,
            depth=depth,
            parameters={
                "Threads": 7,
                "Hash": 128,
                "MultiPV": 1,
                "Minimum Thinking Time": 20
            }
        )
        self.depth = depth

    def get_best_moves(self, fen: str, num_moves: int = 1) -> List[str]:
        """
        Retrieve the best moves for a given position.
        
        Args:
            fen: The FEN representation of the board position
            num_moves: Number of best moves to retrieve
            
        Returns:
            List of best moves in UCI format
        """
        self.stockfish.set_fen_position(fen)
        top_moves = self.stockfish.get_top_moves(num_moves)
        return [move["Move"] for move in top_moves]

    def evaluate_position(self, board: chess.Board, ply: int) -> Info:
        """
        Evaluate position and return Info object with corrected perspective.
        
        Args:
            board: The board position to evaluate
            ply: The current ply number
            
        Returns:
            Info object containing evaluation and best move variation
        """
        self.stockfish.set_fen_position(board.fen())
        eval_dict = self.stockfish.get_evaluation()
        variations = self.get_best_moves(board.fen(), 1)
        
        # Correct evaluation perspective based on side to move
        # if not board.turn:  # If it's Black's move
        #     if eval_dict["type"] == "cp":
        #         eval_dict["value"] = -eval_dict["value"]
        #     elif eval_dict["type"] == "mate":
        #         eval_dict["value"] = -eval_dict["value"]
            
        return Info(
            ply=ply,
            eval=eval_dict,
            variation=variations
        )

    def close(self):
        """Clean up Stockfish engine resources"""
        del self.stockfish
