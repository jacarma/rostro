"""Face pack loader and configuration."""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml


class FacePackType(Enum):
    """Type of face pack rendering."""

    PROGRAMMATIC = auto()  # Built-in procedural rendering
    SPRITES = auto()  # PNG sprite-based rendering


@dataclass
class FaceColors:
    """Colors for programmatic face rendering."""

    face: str = "#FFE4C4"
    eyes: str = "#333333"
    mouth: str = "#CC6666"
    background: str = "#2D2D2D"

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "FaceColors":
        """Create from dictionary."""
        return cls(
            face=data.get("face", "#FFE4C4"),
            eyes=data.get("eyes", "#333333"),
            mouth=data.get("mouth", "#CC6666"),
            background=data.get("background", "#2D2D2D"),
        )


@dataclass
class FacePack:
    """Face pack configuration and assets."""

    name: str
    version: str
    author: str
    pack_type: FacePackType
    path: Path

    # For programmatic type
    colors: FaceColors = field(default_factory=FaceColors)

    # For sprites type (Phase 2)
    resolution: tuple[int, int] = (512, 512)
    base_image: str | None = None
    eyes_open: str | None = None
    eyes_closed: str | None = None
    mouth_images: list[str] = field(default_factory=list)
    overlays: dict[str, str] = field(default_factory=dict)
    blink_interval_ms: int = 4000

    @classmethod
    def load(cls, path: Path) -> "FacePack":
        """Load a face pack from a directory.

        Args:
            path: Path to face pack directory containing manifest.yaml.

        Returns:
            Loaded FacePack instance.
        """
        manifest_path = path / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Face pack manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            data: dict[str, Any] = yaml.safe_load(f)

        pack_type_str = data.get("type", "programmatic")
        pack_type = (
            FacePackType.SPRITES if pack_type_str == "sprites" else FacePackType.PROGRAMMATIC
        )

        pack = cls(
            name=data.get("name", "Unknown"),
            version=data.get("version", "1.0"),
            author=data.get("author", "Unknown"),
            pack_type=pack_type,
            path=path,
        )

        if pack_type == FacePackType.PROGRAMMATIC:
            colors_data = data.get("colors", {})
            pack.colors = FaceColors.from_dict(colors_data)

        elif pack_type == FacePackType.SPRITES:
            res = data.get("resolution", [512, 512])
            pack.resolution = (int(res[0]), int(res[1])) if res else (512, 512)
            layers = data.get("layers", {})

            pack.base_image = layers.get("base")

            eyes = layers.get("eyes", {})
            pack.eyes_open = eyes.get("open")
            pack.eyes_closed = eyes.get("closed")
            pack.blink_interval_ms = eyes.get("blink_interval_ms", 4000)

            pack.mouth_images = layers.get("mouth", [])
            pack.overlays = data.get("overlays", {})

        return pack

    @classmethod
    def default(cls) -> "FacePack":
        """Create the default programmatic face pack.

        Returns:
            Default FacePack instance.
        """
        return cls(
            name="Default Programmatic",
            version="1.0",
            author="Rostro",
            pack_type=FacePackType.PROGRAMMATIC,
            path=Path("."),
            colors=FaceColors(),
        )
