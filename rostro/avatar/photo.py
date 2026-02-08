"""Photographic face rendering using full-frame images."""

from pathlib import Path

import pygame


class PhotoFace:
    """Renders avatar using pre-made photographic images.

    Expects 5 image files in the pack directory (png, jpg, or jpeg):
        m0 — mouth closed, eyes open
        m1 — mouth slightly open, eyes open
        m2 — mouth medium, eyes open
        m3 — mouth wide open, eyes open
        blink — eyes closed, mouth closed
    """

    MOUTH_STEMS = ["m0", "m1", "m2", "m3"]
    BLINK_STEM = "blink"
    EXTENSIONS = [".png", ".jpg", ".jpeg"]

    def __init__(self, pack_path: Path, window_size: tuple[int, int]) -> None:
        """Initialize the photo face renderer.

        Loads all images once, scales and crops them to fill the window
        (cover mode, centered).

        Args:
            pack_path: Path to face pack directory containing the images.
            window_size: Target window size (width, height).
        """
        self._window_size = window_size

        self._mouth_surfaces: list[pygame.Surface] = []
        for stem in self.MOUTH_STEMS:
            img_path = self._find_image(pack_path, stem)
            surface = self._load_and_crop(img_path, window_size)
            self._mouth_surfaces.append(surface)

        blink_path = self._find_image(pack_path, self.BLINK_STEM)
        self._blink_surface = self._load_and_crop(blink_path, window_size)

    @classmethod
    def _find_image(cls, pack_path: Path, stem: str) -> Path:
        """Find an image file by stem, trying multiple extensions.

        Args:
            pack_path: Directory containing images.
            stem: Filename without extension (e.g. "m0", "blink").

        Returns:
            Path to the first matching file.

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        for ext in cls.EXTENSIONS:
            path = pack_path / f"{stem}{ext}"
            if path.exists():
                return path
        tried = ", ".join(f"{stem}{ext}" for ext in cls.EXTENSIONS)
        raise FileNotFoundError(f"Photo face image not found in {pack_path}: tried {tried}")

    @staticmethod
    def _load_and_crop(image_path: Path, target_size: tuple[int, int]) -> pygame.Surface:
        """Load an image and apply cover-crop to fill target size.

        Args:
            image_path: Path to the PNG image.
            target_size: Target (width, height) to fill.

        Returns:
            Cropped and scaled Surface matching target_size.
        """
        img = pygame.image.load(str(image_path))
        img_w, img_h = img.get_size()
        win_w, win_h = target_size

        scale = max(win_w / img_w, win_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        scaled = pygame.transform.smoothscale(img, (scaled_w, scaled_h))

        offset_x = (scaled_w - win_w) // 2
        offset_y = (scaled_h - win_h) // 2
        cropped = scaled.subsurface((offset_x, offset_y, win_w, win_h)).copy()

        return cropped.convert()

    def render(self, surface: pygame.Surface, mouth_level: int, is_blinking: bool) -> None:
        """Render the appropriate face image to the surface.

        Args:
            surface: Pygame surface to render to.
            mouth_level: Mouth opening level 0-3.
            is_blinking: Whether the avatar is currently blinking.
        """
        if is_blinking:
            surface.blit(self._blink_surface, (0, 0))
        else:
            level = max(0, min(3, mouth_level))
            surface.blit(self._mouth_surfaces[level], (0, 0))
