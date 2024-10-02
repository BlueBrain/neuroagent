from typing import Any, AsyncIterator
from pydantic import BaseModel, Field, ValidationError
from langgraph import StateGraph, NodeInterruption
from neuroagent.tools.bluenaas_tool import BlueNaaSTool, InputBlueNaaS, BlueNaaSOutput
from neuroagent.tools.get_me_model_tool import GetMEModelTool
from neuroagent.tools.electrophys_tool import ElectrophysFeatureTool
from neuroagent.app.dependencies import get_settings, get_kg_token, get_httpx_client

class BluenaasSimAgent(BaseAgent):
    """Agent for running BlueNaaS simulations with iterative configuration improvement."""

    async def arun(self, query: str) -> Any:
        """Run the agent against a query."""
        state_graph = StateGraph()
        state_graph.add_node("parse_input", self.parse_input)
        state_graph.add_node("validate_config", self.validate_config)
        state_graph.add_node("prompt_user_for_missing_fields", self.prompt_user_for_missing_fields)
        state_graph.add_node("finalize_config", self.finalize_config)
        state_graph.add_node("run_simulation", self.run_simulation)
        state_graph.add_node("process_results", self.process_results)

        state_graph.add_edge("parse_input", "validate_config")
        state_graph.add_edge("validate_config", "prompt_user_for_missing_fields", condition=lambda x: not x["valid"])
        state_graph.add_edge("validate_config", "finalize_config", condition=lambda x: x["valid"])
        state_graph.add_edge("prompt_user_for_missing_fields", "validate_config")
        state_graph.add_edge("finalize_config", "run_simulation")
        state_graph.add_edge("run_simulation", "process_results")

        initial_state = {"query": query}
        result = await state_graph.run(initial_state)
        return result

    async def parse_input(self, state: dict) -> dict:
        """Parse user input to create initial simulation configuration."""
        # Implement parsing logic here
        parsed_config = {
            "me_model_id": None,  # Placeholder, should be parsed from user input
            "currentInjection": {
                "injectTo": "soma",
                "stimulus": {
                    "stimulusType": "current_clamp",
                    "stimulusProtocol": "fire_pattern",
                    "amplitudes": [0.05]
                }
            },
            "recordFrom": [
                {"section": "soma", "offset": 0.5}
            ],
            "conditions": {
                "celsius": 34.0,
                "vinit": -70.0,
                "hypamp": 0.1,
                "max_time": 1000.0,
                "time_step": 0.025,
                "seed": 42
            },
            "simulationType": "single-neuron-simulation",
            "simulationDuration": 1000
        }
        state["config"] = parsed_config
        return state

    async def validate_config(self, state: dict) -> dict:
        """Validate the simulation configuration using Pydantic."""
        try:
            config = InputBlueNaaS(**state["config"])
            state["valid"] = True
        except ValidationError as e:
            state["valid"] = False
            state["errors"] = e.errors()
        return state

    async def prompt_user_for_missing_fields(self, state: dict) -> dict:
        """Prompt the user for missing fields in the configuration."""
        # Implement logic to prompt user for missing fields
        missing_fields = [error["loc"][0] for error in state["errors"]]
        user_response = await self.metadata["llm"].ainvoke({
            "messages": [
                {"role": "system", "content": f"The following fields are missing or invalid: {missing_fields}"},
                {"role": "user", "content": "Please provide the missing values."}
            ]
        })
        # Update state with user-provided values
        state["config"].update(user_response)
        return state

    async def finalize_config(self, state: dict) -> dict:
        """Finalize the simulation configuration and prompt user for approval."""
        user_response = await self.metadata["llm"].ainvoke({
            "messages": [
                {"role": "system", "content": "Here is the final simulation configuration:"},
                {"role": "system", "content": str(state["config"])},
                {"role": "user", "content": "Do you approve this configuration? (yes/no)"}
            ]
        })
        if user_response.lower() != "yes":
            raise NodeInterruption("User did not approve the configuration.")
        return state

    async def run_simulation(self, state: dict) -> dict:
        """Run the simulation using the BlueNaaSTool."""
        tool = BlueNaaSTool(metadata=self.metadata)
        result = await tool._arun(**state["config"])
        state["simulation_result"] = result
        return state

    async def process_results(self, state: dict) -> dict:
        """Process the simulation results and run electrophysiological analysis."""
        # Implement logic to process simulation results and run electrophysiological analysis
        return state