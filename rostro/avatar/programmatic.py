"""Programmatic (procedural) face rendering."""

import pygame

from rostro.avatar.face_pack import FaceColors


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FFE4C4").

    Returns:
        RGB tuple.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


class ProgrammaticFace:
    """Renders a simple cartoon face using Pygame primitives."""

    def __init__(self, colors: FaceColors, size: tuple[int, int]) -> None:
        """Initialize the programmatic face.

        Args:
            colors: Face colors configuration.
            size: Rendering size (width, height).
        """
        self.colors = colors
        self.size = size
        self._mouth_level = 0  # 0-3 for lip sync
        self._eyes_closed = False

        # Pre-calculate positions based on size
        self._calculate_positions()

    def _calculate_positions(self) -> None:
        """Calculate face element positions based on size."""
        w, h = self.size
        center_x = w // 2
        center_y = h // 2

        # Face oval
        self.face_rect = pygame.Rect(center_x - w // 3, center_y - h // 3, w * 2 // 3, h * 2 // 3)

        # Eyes
        eye_y = center_y - h // 10
        eye_offset = w // 6
        eye_width = w // 12
        eye_height = h // 8

        self.left_eye_rect = pygame.Rect(
            center_x - eye_offset - eye_width // 2,
            eye_y - eye_height // 2,
            eye_width,
            eye_height,
        )
        self.right_eye_rect = pygame.Rect(
            center_x + eye_offset - eye_width // 2,
            eye_y - eye_height // 2,
            eye_width,
            eye_height,
        )

        # Mouth (base position, changes with level)
        self.mouth_center = (center_x, center_y + h // 6)
        self.mouth_base_width = w // 6
        self.mouth_base_height = h // 20

    def set_mouth_level(self, level: int) -> None:
        """Set mouth opening level for lip sync.

        Args:
            level: Mouth level 0-3 (closed to wide open).
        """
        self.mouth_level = max(0, min(3, level))

    def set_eyes_closed(self, closed: bool) -> None:
        """Set eyes open/closed state.

        Args:
            closed: Whether eyes are closed.
        """
        self._eyes_closed = closed

    def render(self, surface: pygame.Surface) -> None:
        """Render the face to a surface.

        Args:
            surface: Pygame surface to render to.
        """
        # Background
        bg_color = hex_to_rgb(self.colors.background)
        surface.fill(bg_color)

        # Face oval
        face_color = hex_to_rgb(self.colors.face)
        pygame.draw.ellipse(surface, face_color, self.face_rect)

        # Eyes
        eye_color = hex_to_rgb(self.colors.eyes)
        if self._eyes_closed:
            # Draw lines for closed eyes
            pygame.draw.line(
                surface,
                eye_color,
                (self.left_eye_rect.left, self.left_eye_rect.centery),
                (self.left_eye_rect.right, self.left_eye_rect.centery),
                3,
            )
            pygame.draw.line(
                surface,
                eye_color,
                (self.right_eye_rect.left, self.right_eye_rect.centery),
                (self.right_eye_rect.right, self.right_eye_rect.centery),
                3,
            )
        else:
            # Draw ellipses for open eyes
            pygame.draw.ellipse(surface, eye_color, self.left_eye_rect)
            pygame.draw.ellipse(surface, eye_color, self.right_eye_rect)

        # Mouth
        mouth_color = hex_to_rgb(self.colors.mouth)
        self._draw_mouth(surface, mouth_color)

    def _draw_mouth(self, surface: pygame.Surface, color: tuple[int, int, int]) -> None:
        """Draw the mouth based on current level.

        Args:
            surface: Pygame surface.
            color: Mouth color.
        """
        cx, cy = self.mouth_center
        base_w = self.mouth_base_width
        base_h = self.mouth_base_height

        # Scale height based on mouth level (0-3)
        height_multiplier = [1, 2, 3, 4][self._mouth_level]
        mouth_height = base_h * height_multiplier

        mouth_rect = pygame.Rect(cx - base_w // 2, cy - mouth_height // 2, base_w, mouth_height)

        if self._mouth_level == 0:
            # Closed mouth - just a line
            pygame.draw.line(
                surface,
                color,
                (mouth_rect.left, mouth_rect.centery),
                (mouth_rect.right, mouth_rect.centery),
                3,
            )
        else:
            # Open mouth - ellipse
            pygame.draw.ellipse(surface, color, mouth_rect)

    @property
    def mouth_level(self) -> int:
        """Get current mouth level."""
        return self._mouth_level

    @mouth_level.setter
    def mouth_level(self, value: int) -> None:
        """Set mouth level."""
        self._mouth_level = max(0, min(3, value))
