from enum import Enum


class DetectionID(Enum):
    """
    Enum class representing different detection IDs used in Soccer AI video analysis.
    Each ID corresponds to a specific class of object detected in the video.

    - BALL: Represents the soccer ball.
    - GOALKEEPER: Represents the goalkeeper.
    - PLAYER: Represents a player on the field.
    - REFEREE: Represents the referee.

    Provides a method to convert class IDs to their corresponding enum members.
    """

    BALL = 0
    GOALKEEPER = 1
    PLAYER = 2
    REFEREE = 3

    @classmethod
    def from_class_id(cls, class_id: int):
        for member in cls:
            if member.value == class_id:
                return member
        raise ValueError(f"Invalid class ID: {class_id}")


class DetectionMode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    Each mode corresponds to a specific type of detection or analysis performed on the video.

    TODO: Add modes descriptions to reflect functionality
    - PITCH_DETECTION
    - PLAYER_DETECTION
    - BALL_DETECTION
    - PLAYER_TRACKING
    - TEAM_CLASSIFICATION
    - RADAR

    Provides a set of predefined modes for video analysis, allowing for flexible configuration of the detection pipeline.
    """

    PITCH_DETECTION = "pitch_detection"
    PLAYER_DETECTION = "player_detection"
    BALL_DETECTION = "ball_detection"
    PLAYER_TRACKING = "player_tracking"
    TEAM_CLASSIFICATION = "team_classification"
    RADAR = "radar"
