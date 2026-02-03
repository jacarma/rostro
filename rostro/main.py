"""Main entry point for Rostro."""

import argparse
import sys
from pathlib import Path

from rostro.runtime.controller import RuntimeController


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Rostro - A lightweight voice-driven 2D avatar assistant"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        from rostro import __version__

        print(f"Rostro v{__version__}")
        return 0

    # Create and run controller
    try:
        controller = RuntimeController(config_path=args.config)
        controller.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
