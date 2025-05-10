#!/usr/bin/env python
import argparse
import asyncio
import sys

from app.agent.mcp import MCPAgent
from app.config import config
from app.logger import logger
from app.schema import AgentState


class MCPRunner:
    """Runner class for MCP Agent with proper path handling and configuration."""

    def __init__(self):
        self.root_path = config.root_path
        self.server_reference = config.mcp_config.server_reference
        self.agent: MCPAgent | None = None

    async def initialize(
        self,
        connection_type: str,
        server_url: str | None = None,
    ) -> None:
        """Initialize the MCP agent with the appropriate connection."""
        logger.info(f"Initializing MCPAgent with {connection_type} connection...")
        self.agent = MCPAgent()

        if connection_type == "stdio":
            await self.agent.initialize(
                connection_type="stdio",
                command=sys.executable,
                args=["-m", self.server_reference],
            )
        else:  # sse
            await self.agent.initialize(connection_type="sse", server_url=server_url)

        logger.info(f"Connected to MCP server via {connection_type}")

    async def run_interactive(self) -> None:
        """Run the agent in interactive mode with continuous input."""
        if not self.agent:
            logger.error("Agent not initialized. Cannot run in interactive mode.")
            return

        logger.info("MCP Agent Interactive Mode (type '退出' or 'exit' to quit)")
        while True:
            if self.agent.state == AgentState.FINISHED:
                logger.info(
                    "MCP Agent has finished its previous task. Ready for new input."
                )
                self.agent.state = AgentState.IDLE
                self.agent.current_step = 0

            user_input = input("\nEnter your request (or '退出'/'exit' to quit): ")
            if not user_input.strip():
                logger.warning(
                    "Empty prompt provided. Please enter a command or '退出'/'exit'."
                )
                continue

            if user_input.lower() in ["退出", "exit"]:
                logger.info("Exiting interactive mode as per user request.")
                break

            logger.info(f"Processing request: {user_input}")
            try:
                response = await self.agent.run(user_input)
                logger.info(f"Agent response: {response}")
                logger.info(
                    "Request processing completed for this cycle. Waiting for next input."
                )
            except Exception as e:
                logger.error(f"Error during agent execution: {e}", exc_info=True)
                logger.info(
                    "Agent encountered an error. Ready for new input or exit command."
                )

    async def run_single_prompt(self, prompt: str) -> None:
        """Run the agent with a single prompt."""
        if not self.agent:
            logger.error("Agent not initialized. Cannot run single prompt.")
            return
        logger.info(f"Processing single prompt: {prompt}")
        await self.agent.run(prompt)
        logger.info("Single prompt processing completed.")

    async def run_default(self) -> None:
        """Run the agent in default mode, which is now continuous interactive mode."""
        if not self.agent:
            logger.error("Agent not initialized. Cannot run in default mode.")
            return

        logger.info(
            "MCP Agent Default (Interactive) Mode (type '退出' or 'exit' to quit)"
        )
        while True:
            if self.agent.state == AgentState.FINISHED:
                logger.info(
                    "MCP Agent has finished its previous task. Ready for new input."
                )
                self.agent.state = AgentState.IDLE
                self.agent.current_step = 0

            prompt = input("Enter your prompt (or '退出'/'exit' to quit): ")
            if not prompt.strip():
                logger.warning(
                    "Empty prompt provided. Please enter a command or '退出'/'exit'."
                )
                continue

            if prompt.lower() in ["退出", "exit"]:
                logger.info("Exiting default mode as per user request.")
                break

            logger.info(f"Processing request: {prompt}")
            try:
                response = await self.agent.run(prompt)
                logger.info(f"Agent response: {response}")
                logger.info(
                    "Request processing completed for this cycle. Waiting for next input."
                )
            except Exception as e:
                logger.error(f"Error during agent execution: {e}", exc_info=True)
                logger.info(
                    "Agent encountered an error. Ready for new input or exit command."
                )

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        if self.agent:
            if hasattr(self.agent, "shutdown_mcp_connections") and callable(
                getattr(self.agent, "shutdown_mcp_connections")
            ):
                await self.agent.shutdown_mcp_connections()
            else:
                await self.agent.cleanup()
        logger.info("MCP session ended and resources cleaned up.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the MCP Agent")
    parser.add_argument(
        "--connection",
        "-c",
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection type: stdio or sse",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/sse",
        help="URL for SSE connection",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument("--prompt", "-p", help="Single prompt to execute and exit")
    return parser.parse_args()


async def run_mcp() -> None:
    """Main entry point for the MCP runner."""
    args = parse_args()
    runner = MCPRunner()

    try:
        await runner.initialize(args.connection, args.server_url)

        if args.prompt:
            await runner.run_single_prompt(args.prompt)
        elif args.interactive:
            await runner.run_interactive()
        else:
            await runner.run_default()

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Error running MCPAgent: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_mcp())
