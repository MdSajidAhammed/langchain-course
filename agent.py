from dotenv import load_dotenv

load_dotenv()
import json
import os



from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


from rich import box
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import pprint
from rich.table import Table
from rich.text import Text


console = Console()




def display_agent_result(result):

    console.print("\n")

    # Main Header
    console.print(
        Panel(
            "[bold white]LANGCHAIN AGENT EXECUTION[/bold white]",
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )

    for message in result["messages"]:

        # =====================================
        # 1. AI RESPONSE
        # =====================================

        if isinstance(message, AIMessage):

            if message.content:

                console.print(
                    Panel(
                        Markdown(message.content),
                        title="[bold green]AI FINAL RESPONSE[/bold green]",
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )

        # =====================================
        # 2. TOOL OUTPUT
        # =====================================

        elif isinstance(message, ToolMessage):

            try:

                data = json.loads(message.content)

                results = data.get("results", [])

                # Search Information
                console.print(
                    Panel(
                        f"[bold cyan]Search Query:[/bold cyan]\n"
                        f"{data.get('query', 'N/A')}\n\n"
                        f"[bold cyan]Total Results:[/bold cyan] "
                        f"{len(results)}",
                        title="[bold yellow]SEARCH INFORMATION[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )

                # Individual Search Results
                for index, item in enumerate(results, start=1):

                    title = item.get("title", "N/A")

                    url = item.get("url", "N/A")

                    score = item.get("score", 0)

                    content = item.get("content", "No description available.")

                    # Result Details Table
                    table = Table(
                        show_header=False, box=box.SIMPLE, padding=(0, 1), expand=True
                    )

                    table.add_column("Field", style="bold cyan", width=20)

                    table.add_column("Information", style="white", overflow="fold")

                    table.add_row("Job / Page Title", title)

                    table.add_row("Website URL", url)

                    table.add_row(
                        "Relevance Score",
                        (
                            f"{score * 100:.2f}%"
                            if isinstance(score, (int, float))
                            else str(score)
                        ),
                    )

                    # Result Description
                    table.add_row(
                        "Description",
                        Text(content[:800] + ("..." if len(content) > 800 else "")),
                    )

                    console.print(
                        Panel(
                            table,
                            title=f"[bold]SEARCH RESULT {index:02d}[/bold]",
                            border_style="bright_blue",
                            box=box.ROUNDED,
                            padding=(1, 2),
                        )
                    )

            except (json.JSONDecodeError, TypeError):

                console.print(
                    Panel(
                        str(message.content),
                        title="[bold yellow]TOOL OUTPUT[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    )
                )

    # Execution Completed
    console.print("\n[bold green]✓ AGENT EXECUTION COMPLETED[/bold green]\n")





llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

result = agent.invoke(
    {
        "messages": HumanMessage(
            content="Search for 3 job posting for an ai engineer using langchain in the Bay area and list their details"
        )
    }
)

display_agent_result(result)

print("\n" * 50)

pprint(result)

print("\n" * 50)

rprint(result)





# for message in result["messages"]:

#     if isinstance(message, AIMessage):
#         print(message.content)

# for message in result["messages"]:

#     print("Type:", type(message).__name__)
#     print("Content:", message.content)
#     print("-" * 40)
