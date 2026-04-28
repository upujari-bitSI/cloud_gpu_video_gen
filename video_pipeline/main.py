"""
Main entry point for the AI Video Generation Pipeline.

Example:
    python main.py --niche "AI replacing jobs story"
    python main.py --niche "ancient Mars civilization" --no-approval
"""
import argparse
import asyncio
import logging
import sys
from agents.orchestrator import Orchestrator
from config import config


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.OUTPUT_DIR / "pipeline.log"),
        ],
    )


async def main():
    parser = argparse.ArgumentParser(description="AI Video Generation Pipeline")
    parser.add_argument("--niche", required=True, help="Story niche/topic")
    parser.add_argument("--no-approval", action="store_true",
                        help="Skip the human approval step for characters")
    args = parser.parse_args()

    if args.no_approval:
        config.HUMAN_APPROVAL_REQUIRED = False

    setup_logging()
    orchestrator = Orchestrator()
    state = await orchestrator.run(args.niche)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Title:        {state.story.title}")
    print(f"Scenes:       {len(state.scenes)}")
    print(f"Characters:   {len(state.characters)}")
    print(f"Final video:  {state.final_video_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
