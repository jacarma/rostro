"""Tests for avatar module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pygame
import pytest

from rostro.avatar.engine import AvatarEngine, AvatarState
from rostro.avatar.face_pack import FaceColors, FacePack, FacePackType
from rostro.avatar.photo import PhotoFace
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

    def test_load_photo_manifest(self):
        """Test loading a photo type manifest."""
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.yaml"
            manifest.write_text(
                """
name: "Test Photo"
version: "1.0"
author: "Test"
type: photo
"""
            )
            pack = FacePack.load(Path(tmpdir))
            assert pack.name == "Test Photo"
            assert pack.pack_type == FacePackType.PHOTO

    def test_load_photo_manifest_with_persona(self):
        """Test that voice, voice_instructions, system_prompt load from manifest."""
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.yaml"
            manifest.write_text(
                """
name: "Persona Pack"
version: "1.0"
author: "Test"
type: photo
voice: sage
voice_instructions: "Speak slowly."
system_prompt: "You are a pirate."
"""
            )
            pack = FacePack.load(Path(tmpdir))
            assert pack.voice == "sage"
            assert pack.voice_instructions == "Speak slowly."
            assert pack.system_prompt == "You are a pirate."

    def test_persona_fields_default_to_none(self):
        """Test that packs without persona fields have None values."""
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.yaml"
            manifest.write_text(
                """
name: "Basic Pack"
version: "1.0"
author: "Test"
type: programmatic
"""
            )
            pack = FacePack.load(Path(tmpdir))
            assert pack.voice is None
            assert pack.voice_instructions is None
            assert pack.system_prompt is None


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

    def test_blink_skipped_when_mouth_open(self):
        """Test that blink is skipped when mouth_level > 0."""
        engine = AvatarEngine()
        engine._mouth_level = 2
        engine._last_blink_time = 0.0  # long ago
        engine._is_blinking = False

        with patch("rostro.avatar.engine.time") as mock_time:
            mock_time.time.return_value = 100.0  # well past 4s
            engine.update()

        # Should NOT blink because mouth is open
        assert engine._is_blinking is False
        # But timer should still reset
        assert engine._last_blink_time == 100.0

    def test_blink_allowed_when_mouth_closed(self):
        """Test that blink occurs when mouth_level == 0."""
        engine = AvatarEngine()
        engine._mouth_level = 0
        engine._last_blink_time = 0.0
        engine._is_blinking = False

        with patch("rostro.avatar.engine.time") as mock_time:
            mock_time.time.return_value = 100.0
            engine.update()

        assert engine._is_blinking is True


def _create_dummy_png(
    path: Path, width: int, height: int, color: tuple[int, int, int] = (128, 128, 128)
) -> None:
    """Create a dummy PNG file for testing."""
    surface = pygame.Surface((width, height))
    surface.fill(color)
    pygame.image.save(surface, str(path))


class TestPhotoFace:
    """Tests for PhotoFace."""

    @pytest.fixture(autouse=True)
    def _init_pygame(self):
        """Initialize pygame for tests."""
        pygame.init()
        pygame.display.set_mode((1, 1))
        yield
        pygame.quit()

    def _create_pack_dir(
        self, tmpdir: str, img_w: int = 1024, img_h: int = 1024, ext: str = ".png"
    ) -> Path:
        """Create a face pack directory with dummy images."""
        pack_path = Path(tmpdir)
        for stem in PhotoFace.MOUTH_STEMS:
            _create_dummy_png(pack_path / f"{stem}{ext}", img_w, img_h, (100, 100, 100))
        _create_dummy_png(pack_path / f"{PhotoFace.BLINK_STEM}{ext}", img_w, img_h, (50, 50, 50))
        return pack_path

    def test_load_all_images(self):
        """Test that all 5 images are loaded successfully."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir)
            face = PhotoFace(pack_path, (800, 600))
            assert len(face._mouth_surfaces) == 4
            assert face._blink_surface is not None

    def test_missing_image_raises_error(self):
        """Test that missing image raises FileNotFoundError."""
        with TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir)
            # Only create some images, not all
            _create_dummy_png(pack_path / "m0.png", 100, 100)
            with pytest.raises(FileNotFoundError):
                PhotoFace(pack_path, (800, 600))

    def test_cover_crop_square_to_landscape(self):
        """Test cover crop: 1024x1024 image to 800x600 window."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir, 1024, 1024)
            face = PhotoFace(pack_path, (800, 600))
            # All surfaces should match window size
            for s in face._mouth_surfaces:
                assert s.get_size() == (800, 600)
            assert face._blink_surface.get_size() == (800, 600)

    def test_cover_crop_landscape_to_portrait(self):
        """Test cover crop: wide image to tall window."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir, 1920, 1080)
            face = PhotoFace(pack_path, (600, 800))
            for s in face._mouth_surfaces:
                assert s.get_size() == (600, 800)

    def test_render_selects_mouth_surface(self):
        """Test that render uses the correct mouth surface."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir)
            face = PhotoFace(pack_path, (800, 600))
            target = pygame.Surface((800, 600))

            # Should not raise for any valid mouth level
            for level in range(4):
                face.render(target, level, False)

    def test_render_blink(self):
        """Test that render uses blink surface when blinking."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir)
            face = PhotoFace(pack_path, (800, 600))
            target = pygame.Surface((800, 600))

            # Should use blink surface
            face.render(target, 0, True)

    def test_render_clamps_mouth_level(self):
        """Test that out-of-range mouth levels are clamped."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir)
            face = PhotoFace(pack_path, (800, 600))
            target = pygame.Surface((800, 600))

            # Should not raise even with out-of-range values
            face.render(target, -1, False)
            face.render(target, 5, False)

    def test_finds_jpeg_images(self):
        """Test that PhotoFace finds .jpg and .jpeg images."""
        with TemporaryDirectory() as tmpdir:
            pack_path = self._create_pack_dir(tmpdir, ext=".jpg")
            face = PhotoFace(pack_path, (800, 600))
            assert len(face._mouth_surfaces) == 4
            assert face._blink_surface is not None

    def test_finds_mixed_extensions(self):
        """Test that PhotoFace handles mixed png/jpg/jpeg extensions."""
        with TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir)
            # Mix of extensions like the real fatlady pack
            _create_dummy_png(pack_path / "m0.jpg", 100, 100)
            _create_dummy_png(pack_path / "m1.jpeg", 100, 100)
            _create_dummy_png(pack_path / "m2.png", 100, 100)
            _create_dummy_png(pack_path / "m3.jpeg", 100, 100)
            _create_dummy_png(pack_path / "blink.jpeg", 100, 100)
            face = PhotoFace(pack_path, (800, 600))
            assert len(face._mouth_surfaces) == 4
