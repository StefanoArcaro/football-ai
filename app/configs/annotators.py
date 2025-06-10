import supervision as sv

from app.configs.config import (
    COLOR_PALETTE,
    PRIMARY_SV_COLOR,
    TEXT_CONFIG,
    VERTEX_COLORS,
)
from app.configs.pitch import SoccerPitchConfiguration

# Soccer pitch configuration
CONFIG = SoccerPitchConfiguration()


class Annotators:
    """Factory class for creating all annotators with consistent styling."""

    @staticmethod
    def vertex_label():
        return sv.VertexLabelAnnotator(
            color=VERTEX_COLORS, text_scale=0.5, border_radius=5, **TEXT_CONFIG
        )

    @staticmethod
    def edge():
        return sv.EdgeAnnotator(
            color=PRIMARY_SV_COLOR,
            thickness=2,
            edges=CONFIG.edges,
        )

    @staticmethod
    def triangle():
        return sv.TriangleAnnotator(
            color=PRIMARY_SV_COLOR,
            base=20,
            height=15,
        )

    @staticmethod
    def box():
        return sv.BoxAnnotator(color=COLOR_PALETTE, thickness=2)

    @staticmethod
    def ellipse():
        return sv.EllipseAnnotator(color=COLOR_PALETTE, thickness=2)

    @staticmethod
    def box_label():
        return sv.LabelAnnotator(color=COLOR_PALETTE, **TEXT_CONFIG)

    @staticmethod
    def ellipse_label():
        return sv.LabelAnnotator(
            color=COLOR_PALETTE, text_position=sv.Position.BOTTOM_CENTER, **TEXT_CONFIG
        )


# Create all annotators
VERTEX_LABEL_ANNOTATOR = Annotators.vertex_label()
EDGE_ANNOTATOR = Annotators.edge()
TRIANGLE_ANNOTATOR = Annotators.triangle()
BOX_ANNOTATOR = Annotators.box()
ELLIPSE_ANNOTATOR = Annotators.ellipse()
BOX_LABEL_ANNOTATOR = Annotators.box_label()
ELLIPSE_LABEL_ANNOTATOR = Annotators.ellipse_label()
