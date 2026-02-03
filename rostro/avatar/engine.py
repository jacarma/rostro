"""Avatar rendering engine."""

import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

import pygame

from rostro.avatar.face_pack import FacePack, FacePackType
from rostro.avatar.programmatic import ProgrammaticFace


class AvatarState(Enum):
    """Avatar visual states."""

    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


class AvatarEngine:
    """2D avatar rendering engine using Pygame."""

    def __init__(
        self,
        face_pack: FacePack | None = None,
        resolution: tuple[int, int] = (800, 600),
        fps: int = 30,
    ) -> None:
        """Initialize the avatar engine.

        Args:
            face_pack: Face pack to use. If None, uses default programmatic.
            resolution: Window resolution (width, height).
            fps: Target frames per second.
        """
        self.face_pack = face_pack or FacePack.default()
        self.resolution = resolution
        self.fps = fps
        self._state = AvatarState.IDLE
        self._mouth_level = 0
        self._current_volume = 0.0

        # Pygame state
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._running = False

        # Programmatic face renderer
        self._programmatic_face: ProgrammaticFace | None = None

        # Blink state
        self._last_blink_time = 0.0
        self._is_blinking = False
        self._blink_duration = 0.15  # seconds

    def initialize(self) -> None:
        """Initialize Pygame and create window."""
        pygame.init()
        pygame.display.set_caption("Rostro")
        self._screen = pygame.display.set_mode(self.resolution)
        self._clock = pygame.time.Clock()
        self._running = True
        self._last_blink_time = time.time()

        # Initialize face renderer based on pack type
        if self.face_pack.pack_type == FacePackType.PROGRAMMATIC:
            self._programmatic_face = ProgrammaticFace(self.face_pack.colors, self.resolution)

    def shutdown(self) -> None:
        """Shutdown Pygame."""
        self._running = False
        pygame.quit()

    @property
    def state(self) -> AvatarState:
        """Get current avatar state."""
        return self._state

    @state.setter
    def state(self, new_state: AvatarState) -> None:
        """Set avatar state.

        Args:
            new_state: New avatar state.
        """
        self._state = new_state

    def set_volume(self, volume: float) -> None:
        """Set current audio volume for lip sync.

        Args:
            volume: Volume level 0.0-1.0.
        """
        self._current_volume = max(0.0, min(1.0, volume))
        # Convert volume to mouth level (0-3)
        if volume < 0.01:
            self._mouth_level = 0
        elif volume < 0.05:
            self._mouth_level = 1
        elif volume < 0.15:
            self._mouth_level = 2
        else:
            self._mouth_level = 3

    def process_events(self) -> bool:
        """Process Pygame events.

        Returns:
            True if should continue running, False to quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def update(self) -> None:
        """Update avatar state (blink timing, etc.)."""
        current_time = time.time()

        # Handle blinking
        if not self._is_blinking:
            if current_time - self._last_blink_time > 4.0:  # Blink every ~4 seconds
                self._is_blinking = True
                self._last_blink_time = current_time
        else:
            if current_time - self._last_blink_time > self._blink_duration:
                self._is_blinking = False

    def render(self) -> None:
        """Render the current frame."""
        if self._screen is None:
            return

        # Update programmatic face state
        if self._programmatic_face is not None:
            self._programmatic_face.set_eyes_closed(self._is_blinking)
            self._programmatic_face.mouth_level = self._mouth_level
            self._programmatic_face.render(self._screen)

        # Draw state indicator
        self._draw_state_indicator()

        pygame.display.flip()

    def _draw_state_indicator(self) -> None:
        """Draw a small indicator showing current state."""
        if self._screen is None:
            return

        # State colors
        state_colors = {
            AvatarState.IDLE: (100, 100, 100),  # Gray
            AvatarState.LISTENING: (0, 200, 0),  # Green
            AvatarState.THINKING: (200, 200, 0),  # Yellow
            AvatarState.SPEAKING: (0, 100, 200),  # Blue
            AvatarState.ERROR: (200, 0, 0),  # Red
        }

        color = state_colors.get(self._state, (100, 100, 100))

        # Draw small circle in bottom-right corner
        pygame.draw.circle(
            self._screen,
            color,
            (self.resolution[0] - 20, self.resolution[1] - 20),
            10,
        )

    def tick(self) -> None:
        """Wait for next frame (maintain FPS)."""
        if self._clock is not None:
            self._clock.tick(self.fps)

    def run_frame(self) -> bool:
        """Run a single frame of the render loop.

        Returns:
            True if should continue, False to quit.
        """
        if not self.process_events():
            return False
        self.update()
        self.render()
        self.tick()
        return True

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        assets_path: Path | None = None,
    ) -> "AvatarEngine":
        """Create avatar engine from configuration.

        Args:
            config: Avatar configuration dictionary.
            assets_path: Path to assets directory.

        Returns:
            Configured AvatarEngine instance.
        """
        face_pack_name = str(config.get("face_pack", "default"))
        res = config.get("resolution", [800, 600])
        resolution: tuple[int, int] = (int(res[0]), int(res[1])) if res else (800, 600)
        fps = int(config.get("fps", 30))

        # Load face pack
        face_pack: FacePack | None = None
        if assets_path is not None:
            face_pack_path = assets_path / "faces" / face_pack_name
            if face_pack_path.exists():
                face_pack = FacePack.load(face_pack_path)

        return cls(
            face_pack=face_pack,
            resolution=resolution,
            fps=fps,
        )
