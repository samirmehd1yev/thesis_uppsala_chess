# src/models/enums.py
from enum import Enum

class GamePhase(Enum):
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"

class Judgment(Enum):
    BRILLIANT = "Brilliant"  # New
    GREAT = "Great"  # New
    GOOD = "Good"
    INACCURACY = "Inaccuracy"
    MISTAKE = "Mistake"
    BLUNDER = "Blunder"
