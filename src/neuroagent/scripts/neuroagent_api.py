"""Start the Neuroagent API."""

import argparse
from pathlib import Path

import uvicorn


def get_parser() -> argparse.ArgumentParser:
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host used by the app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used by the app",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=None,
        help=(
            "Path to the env file for app config. See example at"
            " https://github.com/BlueBrain/neuroagent/.env.example. Reads from local"
            " '.env' file in cwd by default."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers used by the app.",
    )
    return parser


def main() -> None:
    """Run main logic."""
    parser = get_parser()
    args = parser.parse_args()
    uvicorn.run(
        "neuroagent.app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        env_file=args.env,
    )


if __name__ == "__main__":
    main()
