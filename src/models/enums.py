# src/models/enums.py
from enum import Enum

class GamePhase(Enum):
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"

class Judgment(Enum):
    BRILLIANT = "Brilliant"
    GREAT = "Great"
    GOOD = "Good"
    INACCURACY = "Inaccuracy"
    MISTAKE = "Mistake"
    BLUNDER = "Blunder"
