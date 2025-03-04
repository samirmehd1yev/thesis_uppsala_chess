# src/analysis/wdl_sharpness.py
import math
import chess
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from models.data_classes import Info

class WdlSharpnessAnalyzer:
    """
    Analyzer for measuring position sharpness in chess based on WDL (Win/Draw/Loss) statistics.
    This follows the approach described in the Chess Engine Lab articles, using the LC0's WDL model
    to calculate the sharpness of a position.
    """
    
    def __init__(self):
        pass
    import numpy as np

    
    def calculate_sharpness(self, 
                           board: chess.Board, 
                           wdl: Tuple[float, float, float]) -> float:
        """
        This function calculates the sharpness score based on a formula posted by the
        LC0 team on Twitter.
        wdl: list
            The WDL as a list of integers ranging from 0 to 1000
        return -> float
            The shaprness score based on the WDL
        """
        # max() to avoid a division by 0, min() to avoid log(0)
        # print(f"Debug: WDL: {wdl}")
        W = min(max(wdl[0], 0.0001), 0.9999)
        L = min(max(wdl[2], 0.0001), 0.9999)

        # max() in order to prevent negative values
        # I added the *min(W, L) to reduce the sharpness of completely winning positions
        # The *4 is just a scaling factor
        
        try:
            return (max(2/(np.log((1/W)-1) + np.log((1/L)-1)), 0))**2
        except Exception as e:
            print(f"Error calculating sharpness: {e}")
            return 0.0

    
    def calculate_position_sharpness(self,
                                    board: chess.Board,
                                    eval_info: Info) -> Dict[str, float]:
        """
        Calculate sharpness metrics for a given position based on evaluation info.
        
        Args:
            board: Current board position
            eval_info: Evaluation information from the engine
            
        Returns:
            Dictionary with sharpness scores for each position
        """
        result = {
            'sharpness': 0.0,
            'white_sharpness': 0.0,
            'black_sharpness': 0.0
        }
        
        # If we have WDL information directly
        if eval_info.wdl:
            # Extract WDL values from the dictionary
            wins = eval_info.wdl.get('wins', 0)
            draws = eval_info.wdl.get('draws', 0)
            losses = eval_info.wdl.get('losses', 0)
            
            # Already in probability form (0-1)
            sharpness = self.calculate_sharpness(board, (wins, draws, losses))
            # move number
            ply = board.ply()
            move_number = ply // 2 + 1
            # print(f"Debug: Move number: {move_number}")
            
            # print(f"Debug: Sharpness: {sharpness}")
            
            # Both sides experience the same objective sharpness in the position
            result['sharpness'] = sharpness
            result['white_sharpness'] = sharpness
            result['black_sharpness'] = sharpness
            
        
        return result
    
    def calculate_cumulative_sharpness(self, sharpness_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate cumulative sharpness over a list of sharpness scores.
        
        Args:
            sharpness_scores: List of sharpness score dictionaries
            
        Returns:
            Dictionary with cumulative sharpness scores
        """
        if not sharpness_scores:
            return {'sharpness': 0.0, 'white_sharpness': 0.0, 'black_sharpness': 0.0}
        
        # Separate scores for white and black positions
        white_positions = []
        black_positions = []
        
        for i, score in enumerate(sharpness_scores):
            # Even ply numbers (0, 2, 4...) are positions where it's White's turn to move
            # Odd ply numbers (1, 3, 5...) are positions where it's Black's turn to move
            if i % 2 == 0:
                white_positions.append(score.get('sharpness', 0.0))
            else:
                black_positions.append(score.get('sharpness', 0.0))
        
        # Calculate cumulative values
        white_cumulative = sum(white_positions)
        black_cumulative = sum(black_positions)
        
        # Overall cumulative is the sum of white and black cumulative values
        overall_cumulative = white_cumulative + black_cumulative
        
        return {
            'sharpness': overall_cumulative,
            'white_sharpness': white_cumulative,
            'black_sharpness': black_cumulative
        }
    
    def calculate_sharpness_change(self, 
                                  prev_sharpness: float, 
                                  curr_sharpness: float) -> float:
        """
        Calculate the change in sharpness after a move.
        
        Args:
            prev_sharpness: Sharpness before the move
            curr_sharpness: Sharpness after the move
            
        Returns:
            Change in sharpness (positive means the position got sharper)
        """
        return curr_sharpness - prev_sharpness
    
# Function to print sharpness analysis
def print_wdl_sharpness_analysis(sharpness_scores, colorama_fore, colorama_style):
    """
    Print the WDL sharpness analysis in a formatted way.
    """
    from tabulate import tabulate
    
    print("\n" + "="*80)
    print(f"{colorama_fore.BLUE}{colorama_style.BRIGHT}POSITION SHARPNESS ANALYSIS (WDL MODEL){colorama_style.RESET_ALL}")
    print("="*80)
    
    headers = ["#", "Ply", "Position Sharpness", "Description"]
    rows = []
    
    for i, sharpness in enumerate(sharpness_scores):
        ply = sharpness.get('ply', i)
        move_num = (ply // 2) + 1
        side = "White" if ply % 2 == 0 else "Black"
        move = f"{move_num}.{'' if side == 'White' else '..'}"
        
        # Format sharpness score
        sharpness_value = sharpness.get('sharpness', 0.0)
        
        # Determine color based on sharpness
        if sharpness_value >= 7.0:
            color = colorama_fore.RED
            description = "Extremely sharp"
        elif sharpness_value >= 5.0:
            color = colorama_fore.YELLOW
            description = "Very sharp"
        elif sharpness_value >= 3.0:
            color = colorama_fore.GREEN
            description = "Moderately sharp" 
        elif sharpness_value >= 1.0:
            color = colorama_fore.CYAN
            description = "Somewhat sharp"
        else:
            color = colorama_fore.BLUE
            description = "Quiet position"
        
        sharpness_formatted = f"{color}{sharpness_value:.2f}{colorama_style.RESET_ALL}"
        
        rows.append([move, ply, sharpness_formatted, description])
    
    print(tabulate(rows, headers=headers, tablefmt="pretty"))
    print("="*80)
    
    # Calculate cumulative sharpness
    if sharpness_scores:
        # Separate white and black positions
        white_positions = [s.get('sharpness', 0.0) for i, s in enumerate(sharpness_scores) if i % 2 == 0]
        black_positions = [s.get('sharpness', 0.0) for i, s in enumerate(sharpness_scores) if i % 2 == 1]
        
        # Calculate cumulative values
        white_cumulative = sum(white_positions)
        black_cumulative = sum(black_positions)
        overall_cumulative = white_cumulative + black_cumulative
        
        print(f"\nCumulative Position Sharpness: {overall_cumulative:.2f}")
        print(f"White's Cumulative Position Sharpness: {white_cumulative:.2f} (positions where White is to move)")
        print(f"Black's Cumulative Position Sharpness: {black_cumulative:.2f} (positions where Black is to move)")
        print("="*80)