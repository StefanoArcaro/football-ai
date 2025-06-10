import os

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PARENT_DIR, "models")

PLAYER_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "football-player-detection.pt")
PITCH_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "football-pitch-detection.pt")
BALL_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "football-ball-detection.pt")
