import asyncio
import time

from app.agent.manus import Manus
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.schema import AgentState


async def run_flow():
    # Create Manus agent instance once
    manus_agent = await Manus.create()
    agents = {
        "manus": manus_agent,
    }

    try:
        while True:
            if manus_agent.state == AgentState.FINISHED:
                logger.info(
                    "Manus agent has finished its previous task. Ready for new input for the flow."
                )
                manus_agent.state = AgentState.IDLE
                manus_agent.current_step = 0
                # Consider resetting manus_agent.memory if needed for long conversations

            prompt = input(
                "Enter your prompt for the flow (or '退出'/'exit' to quit): "
            )
            if not prompt.strip():
                logger.warning(
                    "Empty prompt provided. Please enter a command or '退出'/'exit'."
                )
                continue

            if prompt.lower() in ["退出", "exit"]:
                logger.info("Exiting flow execution as per user request.")
                break

            # Create flow instance for each request, using the potentially reset agent
            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,  # Or allow user to specify/default
                agents=agents,
            )
            logger.warning("Processing your request with the flow...")

            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600,  # 60 minute timeout for the entire execution
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Flow request processed in {elapsed_time:.2f} seconds")
                logger.info(f"Flow result: {result}")  # Log the actual result
                logger.info(
                    "Flow processing completed for this cycle. Waiting for next input."
                )
            except asyncio.TimeoutError:
                logger.error("Flow request processing timed out after 1 hour.")
                logger.info(
                    "Operation terminated due to timeout. Ready for new input or exit command."
                )
            except Exception as e_inner:
                logger.error(
                    f"Error during flow execution: {str(e_inner)}", exc_info=True
                )
                logger.info(
                    "Flow encountered an error. Ready for new input or exit command."
                )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user (Ctrl+C).")
    # Removed the generic outer Exception catch to avoid masking KeyboardInterrupt or specific errors
    finally:
        logger.info("Cleaning up agent resources for run_flow...")
        if "manus_agent" in locals() and manus_agent:  # Ensure agent was created
            await manus_agent.cleanup()
        logger.info("Cleanup complete for run_flow. Program terminated if exiting.")


if __name__ == "__main__":
    asyncio.run(run_flow())
