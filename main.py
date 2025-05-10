import asyncio

from app.agent.manus import Manus
from app.logger import logger
from app.schema import AgentState


async def main():
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        while True:
            if agent.state == AgentState.FINISHED:
                logger.info(
                    "Agent has finished its previous task. Ready for new input."
                )
                agent.state = AgentState.IDLE
                agent.current_step = 0

            prompt = input("Enter your prompt (or '退出'/'exit' to quit): ")
            if not prompt.strip():
                logger.warning(
                    "Empty prompt provided. Please enter a command or '退出'/'exit'."
                )
                continue

            if prompt.lower() in ["退出", "exit"]:
                logger.info("Exiting application as per user request.")
                break

            logger.warning("Processing your request...")
            try:
                response_summary = await agent.run(request=prompt)
                logger.info(
                    "Request processing completed for this cycle. Waiting for next input."
                )
            except Exception as e:
                logger.error(
                    f"An error occurred during agent execution: {e}", exc_info=True
                )
                logger.info(
                    "Agent encountered an error. Ready for new input or exit command."
                )

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user (Ctrl+C).")
    finally:
        logger.info("Cleaning up agent resources before exiting...")
        await agent.cleanup()
        logger.info("Cleanup complete. Program terminated.")


if __name__ == "__main__":
    asyncio.run(main())
