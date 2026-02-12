"""Avatar rendering engine."""

import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

import pygame

from rostro.avatar.face_pack import FacePack, FacePackType
from rostro.avatar.photo import PhotoFace
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

        # Adaptive lip-sync: track volume history for dynamic thresholds
        self._volume_peak = 0.05  # Initial peak estimate
        self._volume_decay = 0.995  # How fast peak decays (closer to 1 = slower)
        self._volume_attack = 0.3  # How fast peak rises (closer to 1 = faster)

        # Pygame state
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._running = False

        # Face renderers
        self._programmatic_face: ProgrammaticFace | None = None
        self._photo_face: PhotoFace | None = None

        # Blink state
        self._last_blink_time = 0.0
        self._is_blinking = False
        self._blink_duration = 0.15  # seconds

    def initialize(self) -> None:
        """Initialize Pygame and create window."""
        pygame.init()
        pygame.display.set_caption("Rostro")
        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.resolution = self._screen.get_size()
        self._clock = pygame.time.Clock()
        self._running = True
        self._last_blink_time = time.time()

        # Initialize face renderer based on pack type
        if self.face_pack.pack_type == FacePackType.PHOTO:
            self._photo_face = PhotoFace(self.face_pack.path, self.resolution)
        elif self.face_pack.pack_type == FacePackType.PROGRAMMATIC:
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

        Uses adaptive thresholds based on recent volume peak.

        Args:
            volume: Volume level 0.0-1.0.
        """
        self._current_volume = max(0.0, min(1.0, volume))

        # Update adaptive peak tracking
        if volume > self._volume_peak:
            # Fast attack: quickly rise to new peaks
            self._volume_peak = (
                self._volume_attack * volume
                + (1 - self._volume_attack) * self._volume_peak
            )
        else:
            # Slow decay: gradually decrease peak over time
            self._volume_peak *= self._volume_decay

        # Ensure minimum peak to avoid division issues
        peak = max(self._volume_peak, 0.01)

        # Normalize volume relative to peak and convert to mouth level
        # Thresholds at ~15%, ~35%, ~60% of peak
        if volume < peak * 0.15:
            self._mouth_level = 0  # Closed
        elif volume < peak * 0.35:
            self._mouth_level = 1  # Slightly open
        elif volume < peak * 0.60:
            self._mouth_level = 2  # Medium open
        else:
            self._mouth_level = 3  # Wide open

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

        # Handle blinking — only blink when mouth is closed
        if not self._is_blinking:
            if current_time - self._last_blink_time > 4.0:  # Blink every ~4 seconds
                if self._mouth_level == 0:
                    self._is_blinking = True
                self._last_blink_time = current_time
        else:
            if current_time - self._last_blink_time > self._blink_duration:
                self._is_blinking = False

    def render(self) -> None:
        """Render the current frame."""
        if self._screen is None:
            return

        # Render face based on type
        if self._photo_face is not None:
            self._photo_face.render(self._screen, self._mouth_level, self._is_blinking)
        elif self._programmatic_face is not None:
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
