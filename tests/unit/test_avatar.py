"""Tests for avatar module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rostro.avatar.engine import AvatarEngine, AvatarState
from rostro.avatar.face_pack import FaceColors, FacePack, FacePackType
from rostro.avatar.programmatic import ProgrammaticFace, hex_to_rgb


class TestFaceColors:
    """Tests for FaceColors."""

    def test_default_values(self):
        """Test default color values."""
        colors = FaceColors()
        assert colors.face == "#FFE4C4"
        assert colors.eyes == "#333333"
        assert colors.mouth == "#CC6666"
        assert colors.background == "#2D2D2D"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "face": "#FFFFFF",
            "eyes": "#000000",
        }
        colors = FaceColors.from_dict(data)
        assert colors.face == "#FFFFFF"
        assert colors.eyes == "#000000"
        # Should use defaults for missing
        assert colors.mouth == "#CC6666"


class TestFacePack:
    """Tests for FacePack."""

    def test_default_pack(self):
        """Test creating default face pack."""
        pack = FacePack.default()
        assert pack.name == "Default Programmatic"
        assert pack.pack_type == FacePackType.PROGRAMMATIC
        assert isinstance(pack.colors, FaceColors)

    def test_load_from_manifest(self):
        """Test loading from manifest file."""
        with TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest = Path(tmpdir) / "manifest.yaml"
            manifest.write_text(
                """
name: "Test Face"
version: "1.0"
author: "Test"
type: programmatic
colors:
  face: "#AABBCC"
"""
            )

            pack = FacePack.load(Path(tmpdir))
            assert pack.name == "Test Face"
            assert pack.pack_type == FacePackType.PROGRAMMATIC
            assert pack.colors.face == "#AABBCC"

    def test_load_missing_manifest(self):
        """Test loading from missing manifest raises error."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                FacePack.load(Path(tmpdir))


class TestHexToRgb:
    """Tests for hex_to_rgb conversion."""

    def test_convert_black(self):
        """Test converting black."""
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_convert_white(self):
        """Test converting white."""
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)

    def test_convert_red(self):
        """Test converting red."""
        assert hex_to_rgb("#FF0000") == (255, 0, 0)

    def test_convert_without_hash(self):
        """Test converting without hash prefix."""
        assert hex_to_rgb("00FF00") == (0, 255, 0)


class TestProgrammaticFace:
    """Tests for ProgrammaticFace."""

    def test_init(self):
        """Test initialization."""
        colors = FaceColors()
        face = ProgrammaticFace(colors, (800, 600))
        assert face.mouth_level == 0

    def test_set_mouth_level(self):
        """Test setting mouth level."""
        face = ProgrammaticFace(FaceColors(), (800, 600))

        face.mouth_level = 2
        assert face.mouth_level == 2

        # Should clamp values
        face.mouth_level = 5
        assert face.mouth_level == 3

        face.mouth_level = -1
        assert face.mouth_level == 0

    def test_set_eyes_closed(self):
        """Test setting eyes closed state."""
        face = ProgrammaticFace(FaceColors(), (800, 600))

        face.set_eyes_closed(True)
        assert face._eyes_closed is True

        face.set_eyes_closed(False)
        assert face._eyes_closed is False


class TestAvatarState:
    """Tests for AvatarState enum."""

    def test_states_exist(self):
        """Test that all states exist."""
        assert AvatarState.IDLE
        assert AvatarState.LISTENING
        assert AvatarState.THINKING
        assert AvatarState.SPEAKING
        assert AvatarState.ERROR


class TestAvatarEngine:
    """Tests for AvatarEngine."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        engine = AvatarEngine()
        assert engine.resolution == (800, 600)
        assert engine.fps == 30
        assert engine.face_pack is not None

    def test_init_custom(self):
        """Test initialization with custom values."""
        pack = FacePack.default()
        engine = AvatarEngine(
            face_pack=pack,
            resolution=(1024, 768),
            fps=60,
        )
        assert engine.resolution == (1024, 768)
        assert engine.fps == 60

    def test_state_property(self):
        """Test state getter/setter."""
        engine = AvatarEngine()
        assert engine.state == AvatarState.IDLE

        engine.state = AvatarState.SPEAKING
        assert engine.state == AvatarState.SPEAKING

    def test_set_volume(self):
        """Test volume to mouth level conversion."""
        engine = AvatarEngine()

        engine.set_volume(0.0)
        assert engine._mouth_level == 0

        engine.set_volume(0.03)
        assert engine._mouth_level == 1

        engine.set_volume(0.10)
        assert engine._mouth_level == 2

        engine.set_volume(0.5)
        assert engine._mouth_level == 3

    def test_from_config(self):
        """Test creating from config dict."""
        config = {
            "face_pack": "default",
            "resolution": [1280, 720],
            "fps": 24,
        }
        engine = AvatarEngine.from_config(config)
        assert engine.resolution == (1280, 720)
        assert engine.fps == 24
