import chess
import chess.pgn
import io
import numpy as np
from collections import defaultdict
import re
import pandas as pd
from typing import Tuple, List, Dict

class MaterialFeatures:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Not counted in material balance
        }
        
    def get_material_balance(self, board):
        """Calculate material balance from white's perspective"""
        balance = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_pieces = len(board.pieces(piece_type, chess.WHITE))
            black_pieces = len(board.pieces(piece_type, chess.BLACK))
            balance += (white_pieces - black_pieces) * self.piece_values[piece_type]
        return balance
    
    def detect_sacrifice(self, board, move):
        """Detect if a move is a genuine sacrifice (losing material without compensation)"""
        if not board.is_capture(move):
            return False
            
        # Get the pieces involved
        from_piece = board.piece_at(move.from_square)
        to_piece = board.piece_at(move.to_square)
        
        if from_piece is None or to_piece is None:
            return False
            
        # Check piece values
        attacking_value = self.piece_values[from_piece.piece_type]
        defending_value = self.piece_values[to_piece.piece_type]
        
        # Check if captured piece is defended
        is_defended = len(board.attackers(not board.turn, move.to_square)) > 0
        
        # Check for immediate recapture possibility
        board_copy = board.copy()
        board_copy.push(move)
        can_recapture = len(board_copy.attackers(board_copy.turn, move.to_square)) > 0
        
        # It's a sacrifice if:
        # 1. We give up more valuable piece
        # 2. Captured piece is defended OR we can't immediately recapture
        return (attacking_value > defending_value and 
                (is_defended or not can_recapture))

    def get_positions_from_moves(self, moves_str: str) -> list[chess.Board]:
        """Converts a string of chess moves into a list of board positions."""
        board = chess.Board()
        positions = [board.copy()]
        
        # Remove move numbers and results
        moves = []
        for move_pair in re.split(r'\d+\.', moves_str):
            if move_pair.strip():
                # Remove any {...} comments and results
                clean_pair = ' '.join(word for word in move_pair.split() 
                                    if not (word.startswith('{') or word.endswith('}') 
                                          or word in ['1-0', '0-1', '1/2-1/2']))
                moves.extend(clean_pair.strip().split())

        for move in moves:
            try:
                board.push_san(move)
                positions.append(board.copy())
            except Exception as e:
                if self.debug:
                    print(f"Error processing move {move}: {e}")
                continue
                
        return positions, moves
    
    def analyze_game(self, moves_str):
        """Analyze a game and return material-related features"""
        positions, moves = self.get_positions_from_moves(moves_str)
        
        if len(positions) <= 1:  # No valid moves parsed
            return {
                'material_volatility': 0,
                'sac_count': 0,
                'exchange_sac_count': 0,
                'avg_material_imbalance': 0,
                'max_material_deficit': 0,
                'moves_analyzed': [],
                'sacrifices': [],
                'exchange_sacrifices': [],
                'material_balance_history': []
            }
        
        # Track material balance throughout the game
        material_balances = []
        sacrifice_moves = []
        exchange_sacrifice_moves = []
        moves_info = []
        
        # Analyze each position
        for i in range(len(positions)-1):
            curr_pos = positions[i]
            next_pos = positions[i+1]
            move = next_pos.move_stack[-1]
            
            move_number = (i // 2) + 1
            is_white = i % 2 == 0
            san_move = moves[i]
            
            # Store move info
            moves_info.append({
                'move_number': move_number,
                'color': 'White' if is_white else 'Black',
                'move': san_move
            })
            
            # Check for sacrifices before making the move
            if self.detect_sacrifice(curr_pos, move):
                moving_piece = curr_pos.piece_at(move.from_square)
                captured_piece = curr_pos.piece_at(move.to_square)
                
                sacrifice_info = {
                    'move_number': move_number,
                    'color': 'White' if is_white else 'Black',
                    'move': san_move,
                    'sacrificed_piece': moving_piece.symbol(),
                    'captured_piece': captured_piece.symbol()
                }
                sacrifice_moves.append(sacrifice_info)
                
                # Check specifically for exchange sacrifices (Rook for minor piece)
                if (moving_piece.piece_type == chess.ROOK and 
                    captured_piece.piece_type in [chess.KNIGHT, chess.BISHOP]):
                    exchange_sacrifice_moves.append(sacrifice_info)
            
            # Calculate material balance after the move
            balance = self.get_material_balance(next_pos)
            material_balances.append(balance)
        
        # Calculate features
        material_balances = np.array(material_balances)
        if len(material_balances) > 0:
            features = {
                'material_volatility': float(np.std(material_balances)),
                'sac_count': len(sacrifice_moves),
                'exchange_sac_count': len(exchange_sacrifice_moves),
                'avg_material_imbalance': float(np.mean(np.abs(material_balances))),
                'max_material_deficit': float(min(material_balances)),
                'moves_analyzed': moves_info,
                'sacrifices': sacrifice_moves,
                'exchange_sacrifices': exchange_sacrifice_moves,
                'material_balance_history': list(zip(range(1, len(material_balances) + 1), 
                                                   material_balances.tolist()))
            }
        else:
            features = {
                'material_volatility': 0,
                'sac_count': 0,
                'exchange_sac_count': 0,
                'avg_material_imbalance': 0,
                'max_material_deficit': 0,
                'moves_analyzed': [],
                'sacrifices': [],
                'exchange_sacrifices': [],
                'material_balance_history': []
            }
        
        return features

    def analyze_games(self, games_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Analyze multiple games and return features for each.
        
        Args:
            games_df: DataFrame containing chess games
            
        Returns:
            Tuple containing:
            - DataFrame with numeric features
            - List of detailed game analysis
        """
        all_features = []
        game_details = []
        
        if self.debug:
            print("\n" + "="*80)
            print("ANALYZING CHESS GAMES")
            print("="*80)
        
        for _, game in games_df.iterrows():
            features = self.analyze_game(game['moves'])
            print(f"MOVES: {game['moves']}")
            if features:
                # Separate numeric features
                numeric_features = {
                    'game_id': game['id'],
                    'white': game['white'],
                    'black': game['black'],
                    'material_volatility': features['material_volatility'],
                    'sac_count': features['sac_count'],
                    'exchange_sac_count': features['exchange_sac_count'],
                    'avg_material_imbalance': features['avg_material_imbalance'],
                    'max_material_deficit': features['max_material_deficit']
                }
                all_features.append(numeric_features)
                
                # Store detailed game analysis
                game_details.append({
                    'game_id': game['id'],
                    'white': game['white'],
                    'black': game['black'],
                    'white_elo': game.get('white_elo', '?'),
                    'black_elo': game.get('black_elo', '?'),
                    'moves': features['moves_analyzed'],
                    'sacrifices': features['sacrifices'],
                    'exchange_sacrifices': features['exchange_sacrifices'],
                    'material_history': features['material_balance_history']
                })
                
                if self.debug:
                    self._print_game_analysis(game_details[-1])
        
        if self.debug:
            print("\nAGGREGATED FEATURES:")
            print("-" * 80)
            
        return pd.DataFrame(all_features), game_details

    def _print_game_analysis(self, game: Dict) -> None:
        """Print detailed analysis of a single game."""
        print(f"\n{'='*80}")
        print(f"GAME {game['game_id']}")
        print(f"White: {game['white']} ({game['white_elo']})")
        print(f"Black: {game['black']} ({game['black_elo']})")
        print(f"{'-'*80}")
        
        print("\nOPENING SEQUENCE:")
        print("-" * 40)
        for move in game['moves'][:10]:
            print(f"Move {move['move_number']:2d} | {move['color']:5s} | {move['move']:<15s}")
        
        if game['sacrifices']:
            print("\nSACRIFICES DETECTED:")
            print("-" * 40)
            for sac in game['sacrifices']:
                print(f"Move {sac['move_number']:2d} | {sac['color']:5s} | "
                      f"{sac['sacrificed_piece']} → {sac['captured_piece']} | {sac['move']:<15s}")
        
        if game['exchange_sacrifices']:
            print("\nEXCHANGE SACRIFICES:")
            print("-" * 40)
            for sac in game['exchange_sacrifices']:
                print(f"Move {sac['move_number']:2d} | {sac['color']:5s} | "
                      f"R → {sac['captured_piece']} | {sac['move']:<15s}")
        
        print("\nMATERIAL BALANCE:")
        print("-" * 40)
        for move_num, balance in game['material_history']:
            print(f"Move {move_num:2d} | {balance:+.1f}")
        
        print("="*80)


def main():
    # Connect to your database
    conn = sqlite3.connect('/Users/samir/Desktop/Uppsala/Thesis/thesis_chess_code/data/processed/chess_games.db')
    query = """
    SELECT id, white, black, white_elo, black_elo, moves
    FROM chess_games
    LIMIT 2
    """
    games_df = pd.read_sql(query, conn)
    
    # Initialize analyzer with debug=True
    analyzer = MaterialFeatures(debug=True)
    
    # Analyze games
    features_df, _ = analyzer.analyze_games(games_df)
    
    if analyzer.debug:
        print("\nAGGREGATED FEATURES:")
        print("-" * 80)
        print(features_df)
    
    conn.close()

if __name__ == "__main__":
    import sqlite3
    main()