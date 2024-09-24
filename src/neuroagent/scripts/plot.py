"""Plotting tool first iteration."""

import io
import logging
import os
import uuid
from typing import Literal

import boto3
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain.agents.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

load_dotenv()


def plot_line(t: list[float], v: list[float]) -> str:
    """Plot the result of simulation as a line plot."""
    logger.info("Plotting the simulation results")
    plt.switch_backend("Agg")
    plt.figure()
    plt.plot(t, v)
    plt.title("Hard to read title that shows if the LLM can access the plot or not.")

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format="png")
    image_stream.seek(0)
    return image_stream


@tool
def simulate(type: Literal["single_cell"], plot: bool = False):
    """Run a cell simulation. Optionally plot the results and interpret it."""
    logger.info(f"Entering simulation tool with inputs: {type=}, {plot=}.")
    if type == "single_cell":
        t = np.linspace(0, 100)
        v = np.sin(t)
    else:
        raise ValueError("Incorrect simulation type")
    if plot:
        image_stream = plot_line(t, v)
        client = boto3.client("s3")
        uuid_str = str(uuid.uuid4()) + ".png"
        client.upload_fileobj(image_stream, "test-agent-image", uuid_str)
        response = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": "test-agent-image",
                "Key": uuid_str,
                "ResponseContentDisposition": "inline",
                "ResponseContentType": "image/png",
            },
            ExpiresIn=60 * 15,
        )
        llm = ChatOpenAI(
            api_key=os.getenv("NEUROAGENT_OPENAI__TOKEN"), model="gpt-4o-mini"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are an expert in interpreting plot outputs of neuroscience simulations."
                ),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Access the plot and tell me what is its title.",
                        },
                        {"type": "image_url", "image_url": {"url": response}},
                    ],
                ),
            ]
        )
        chain = prompt | llm
        interpretation = chain.invoke(message)
        return interpretation.content


logging.basicConfig(
    format="[%(levelname)s]  %(asctime)s %(name)s  %(message)s", level=logging.INFO
)

llm = ChatOpenAI(
    api_key=os.getenv("NEUROAGENT_OPENAI__TOKEN"), model="gpt-4o-mini", temperature=0
)
agent = create_react_agent(llm, [simulate], interrupt_after=["tools"])
message = {
    "messages": [
        (
            "human",
            "Run a cell simulation and plot the results. Then give me the title of the plot.",
        )
    ]
}

answer = agent.invoke(message)
print(answer["messages"][-1].content)
