from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message

# Import ToolCallAgent for instanceof check, allow it to fail gracefully if not always present
try:
    from app.agent.toolcall import ToolCallAgent
except ImportError:
    ToolCallAgent = None


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in a runnable state.
        """
        # Allow run if IDLE. If FINISHED, it means main.py might be starting a new "session"
        # after a previous one fully completed. Reset to IDLE for the new session.
        if self.state == AgentState.FINISHED:
            logger.debug(f"Agent was in FINISHED state. Resetting to IDLE for new run.")
            self.state = AgentState.IDLE
            self.current_step = 0  # Ensure steps are reset for a conceptually new run
            # Memory clearing/management for distinct sessions might be needed here
            # depending on desired long-term conversation behavior.

        if self.state != AgentState.IDLE:
            raise RuntimeError(
                f"Cannot run agent from state: {self.state}. Expected IDLE."
            )

        if request:
            self.update_memory("user", request)

        self.current_step = 0  # Reset step count for this invocation of run()

        # Manually manage state instead of using state_context to preserve FINISHED state
        # previous_agent_state = self.state # Should be IDLE here
        self.state = AgentState.RUNNING
        results: List[str] = []

        try:
            while self.current_step < self.max_steps:
                if self.state != AgentState.RUNNING:
                    logger.info(
                        f"Agent state is {self.state} at beginning of step {self.current_step + 1}. Terminating run loop."
                    )
                    break

                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()  # This calls think() then act()
                results.append(f"Step {self.current_step}: {step_result}")

                if self.is_stuck():  # Check for stuck state
                    self.handle_stuck_state()

                if self.state == AgentState.FINISHED:
                    logger.info(
                        f"Agent run completed: state set to FINISHED by a tool at step {self.current_step}."
                    )
                    break

                pause_condition_met = False

                # Log Condition_X components with CRITICAL level
                condition_x_part1 = ToolCallAgent is not None
                condition_x_part2 = (
                    isinstance(self, ToolCallAgent) if ToolCallAgent else False
                )
                logger.critical(
                    f"[CRITICAL_COND_X_CHECK] Step {self.current_step}: ToolCallAgent is not None: {condition_x_part1}"
                )
                logger.critical(
                    f"[CRITICAL_COND_X_CHECK] Step {self.current_step}: isinstance(self, ToolCallAgent): {condition_x_part2}"
                )

                if condition_x_part1 and condition_x_part2:  # Condition_X
                    current_tool_calls = getattr(self, "tool_calls", None)

                    # Log Condition_Y components with CRITICAL level
                    condition_y_part1 = not current_tool_calls
                    condition_y_part2 = bool(
                        step_result
                        and step_result.strip()
                        and step_result.strip()
                        != "Thinking complete - no action needed"
                    )
                    logger.critical(
                        f"[CRITICAL_COND_Y_CHECK] Step {self.current_step}: not current_tool_calls: {condition_y_part1}"
                    )
                    logger.critical(
                        f"[CRITICAL_COND_Y_CHECK] Step {self.current_step}: step_result is valid for pause: {condition_y_part2}"
                    )
                    logger.critical(
                        f"[CRITICAL_COND_Y_CHECK] Step {self.current_step}: current_tool_calls value: {current_tool_calls}"
                    )
                    logger.critical(
                        f"[CRITICAL_COND_Y_CHECK] Step {self.current_step}: step_result value snippet: '{str(step_result)[:100]}...'"
                    )

                    if condition_y_part1 and condition_y_part2:  # Condition_Y
                        logger.info(
                            f"[PAUSE_TRIGGERED_LOOP_SIMPLIFIED] Step {self.current_step}: No tools selected, and agent provided a response. Setting state to IDLE."
                        )
                        self.state = AgentState.IDLE
                        pause_condition_met = True
                elif ToolCallAgent is None:
                    logger.debug(
                        f"[PAUSE_CHECK_SKIPPED] Step {self.current_step}: ToolCallAgent is None, skipping pause logic block."
                    )
                elif not isinstance(self, ToolCallAgent):
                    logger.debug(
                        f"[PAUSE_CHECK_SKIPPED] Step {self.current_step}: self is not isinstance of ToolCallAgent ({type(self)} vs {ToolCallAgent}), skipping pause logic block."
                    )

                if pause_condition_met:
                    break

                if self.current_step >= self.max_steps:
                    logger.info(
                        f"Reached max steps ({self.max_steps}). Current run cycle terminating."
                    )
                    results.append(
                        f"Terminated current run: Reached max steps ({self.max_steps})."
                    )
                    if (
                        self.state == AgentState.RUNNING
                    ):  # If not already finished by a tool
                        self.state = (
                            AgentState.IDLE
                        )  # Normal completion due to max_steps
                    break

            # If loop exited for reasons other than explicit state change (e.g., conditions unmet)
            # and state is still RUNNING, default to IDLE. This is a safeguard.
            if self.state == AgentState.RUNNING:
                logger.warning(
                    "Agent run loop exited while state was still RUNNING. Forcing to IDLE."
                )
                self.state = AgentState.IDLE

        except Exception as e:
            logger.error(f"Error during agent run: {e}", exc_info=True)
            self.state = AgentState.ERROR
            results.append(f"Error encountered: {str(e)}")
            # Do not re-raise, allow run to return and main loop to decide action based on ERROR state.
        finally:
            # Ensure agent is not left in RUNNING state.
            # If it's FINISHED (by Terminate) or ERROR (by exception) or IDLE (by max_steps/pause), that's respected.
            if (
                self.state == AgentState.RUNNING
            ):  # Should ideally be unreachable if logic above is correct
                logger.error(
                    "CRITICAL: Agent state was RUNNING in finally block. Forcing to IDLE."
                )
                self.state = AgentState.IDLE

            # SANDBOX_CLIENT.cleanup() has been removed from here.
            # It should be handled by the main entry point script's final cleanup.

        return "\\n".join(results) if results else "No steps executed or run ended."

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
