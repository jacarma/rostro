"""Silero VAD ONNX model wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime  # type: ignore[import-untyped]

_MODEL_PATH = Path(__file__).parent / "data" / "silero_vad.onnx"

# Silero expects 512 samples at 16kHz
SILERO_FRAME_SAMPLES = 512
SILERO_SAMPLE_RATE = 16000


class SileroVAD:
    """Lightweight wrapper around the Silero VAD ONNX model.

    Only supports 16kHz, batch=1. Call with 512 float32 samples
    to get a voice confidence score (0.0-1.0).
    """

    def __init__(self, model_path: Path = _MODEL_PATH) -> None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def reset_states(self) -> None:
        """Reset internal RNN state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, audio: "np.ndarray[Any, np.dtype[np.float32]]") -> float:
        """Run inference on a 512-sample float32 frame.

        Args:
            audio: 1-D array of 512 float32 samples at 16kHz.

        Returns:
            Voice confidence between 0.0 and 1.0.
        """
        x = audio.reshape(1, -1)
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(SILERO_SAMPLE_RATE, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_inputs)
        self._state = state
        self._context = x[:, -64:]

        return float(out[0][0])
