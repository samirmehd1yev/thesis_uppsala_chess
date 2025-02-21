import math
from typing import Optional
from models.data_classes import Info
from models.enums import Judgment


class MateSequence:
    """Holds predefined mate sequence descriptions."""
    CREATED = ("Checkmate is now unavoidable", "Mate Created")
    DELAYED = ("Not the best checkmate sequence", "Mate Delayed")
    LOST = ("Lost forced checkmate sequence", "Mate Lost")


class MoveAnalyzer:
    @staticmethod
    def winning_chances(cp: int) -> float:
        """Convert centipawn evaluation into winning chances using Lichess' formula"""
        return 2 / (1 + math.exp(-0.004 * cp)) - 1

    @staticmethod
    def analyze_move(prev: Info, curr: Info) -> Optional[Judgment]:
        # Handle mate scores first
        if curr.mate is not None:
            if prev.cp is not None:
                if prev.cp > 400 and curr.mate < 0:
                    return Judgment.BLUNDER
                if abs(prev.cp) < 500 and curr.mate < 0:
                    return Judgment.BLUNDER
            return None

        if prev.mate is not None:
            if prev.mate > 0 and curr.cp is not None:
                return Judgment.BLUNDER
            return None

        if prev.cp is None or curr.cp is None:
            print(f"Missing evaluation: {prev.eval} -> {curr.eval}")
            return None

        prev_wc = MoveAnalyzer.winning_chances(prev.cp)
        curr_wc = MoveAnalyzer.winning_chances(curr.cp)
        delta = curr_wc - prev_wc

        # Invert delta if the move maker was Black
        if not prev.color:  # prev.color is True for White moves, False for Black moves
            delta = -delta

        if delta <= -0.3:
            return Judgment.BLUNDER
        elif delta <= -0.2:
            return Judgment.MISTAKE
        elif delta <= -0.1:
            return Judgment.INACCURACY

        return None
