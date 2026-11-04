
import inspect
import os
import re

from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ==========================================
# 1. Tools
# ==========================================

@traceable(name="get_product_price")
def get_product_price(product: str) -> int | str:
    """Look up the price of a product in the catalog."""
    print(f">> Execute get_product_price(product={product})")
    prices = {
        "laptop": 999,
        "smartphone": 699,
        "headphones": 5678,
        "tablet": 499,
    }
    return prices.get(product.lower(), "Product not found")


@traceable(name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price. Available tiers: bronze, silver, gold."""
    print(
        f">> Execute apply_discount(price={price}, "
        f"discount_tier={discount_tier})"
    )
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# ==========================================
# 2. Tool Description
# ==========================================

def get_tool_description(tool_dict) -> str:
    """Get descriptions of available tools."""
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(
            f"{tool_name}{signature}: {docstring}"
        )
    return "\n".join(descriptions)


tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

# ==========================================
# 3. ReAct Prompt
# ==========================================

react_prompt = f"""
Answer the following question using the available tools.
Available Tools: {tool_descriptions}
Use this exact format:
Question: the input question
Thought: what you need to do
Action: one of [{tool_names}]
Action Input: tool arguments
Observation: tool execution result
Thought: what to do next
Final Answer: the final answer

Important Rules:
1. Use only the available tools.
2. Tool arguments must follow the function parameter order.
3. Separate multiple arguments using commas.
4. Do not include parameter names in Action Input.
5. Use Final Answer when you have enough information.
6. Do not invent tool results.

Examples:
Action: get_product_price
Action Input: laptop
Action: apply_discount
Action Input: 999, gold

Begin!
Question: {{question}}
{{agent_scratchpad}}
"""

# ==========================================
# 4. OpenAI Chat
# ==========================================

@traceable(name="OpenAI Chat", run_type="llm")
def openai_chat_traced(messages, option):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **option,
    )

# ==========================================
# 5. Argument Parser
# ==========================================

def parse_tool_arguments(tool_name: str, raw_input: str):
    original_function = getattr(
        tools[tool_name],
        "__wrapped__",
        tools[tool_name],
    )
    signature = inspect.signature(original_function)
    parameters = list(signature.parameters.values())

    # Parse comma-separated arguments.
    raw_args = [
        value.strip().strip("'\"")
        for value in raw_input.split(",")
        if value.strip()
    ]

    if len(raw_args) != len(parameters):
        raise ValueError(
            f"Expected {len(parameters)} arguments, "
            f"but received {len(raw_args)}."
        )

    args = []
    for value, parameter in zip(raw_args, parameters):
        annotation = parameter.annotation
        if annotation is float:
            value = float(value)
        elif annotation is int:
            value = int(value)
        elif annotation is str:
            value = str(value)
        args.append(value)
    return args

# ==========================================
# 6. Agent Loop
# ==========================================

@traceable(name="Raw ReAct Agent")
def run_agent(question: str):
    print(f"\nQuestion: {question}")
    print("=" * 60)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration}")
        print("-" * 40)

        # Build the complete prompt.
        full_prompt = react_prompt.format(
            question=question,
            agent_scratchpad=scratchpad,
        )

        # Call the LLM.
        response = openai_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            option={
                "stop": ["\nObservation"],
                "temperature": 0.0,
            },
        )
        output = response.choices[0].message.content or ""
        print(f"\nModel Output:\n{output}")

        # ----------------------------------
        # Check Final Answer
        # ----------------------------------
        final_answer_match = re.search(
            r"Final Answer:\s*(.*)",
            output,
            re.IGNORECASE,
        )
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # ----------------------------------
        # Parse Action
        # ----------------------------------
        action_match = re.search(
            r"Action:\s*(.+)",
            output,
            re.IGNORECASE,
        )
        action_input_match = re.search(
            r"Action Input:\s*(.+)",
            output,
            re.IGNORECASE,
        )
        if not action_match or not action_input_match:
            print("\nERROR: Action or Action Input not found.")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(f"\nSelected Tool: {tool_name}")
        print(f"Raw Input: {tool_input_raw}")

        # ----------------------------------
        # Execute Tool
        # ----------------------------------
        if tool_name not in tools:
            observation = f"Tool '{tool_name}' not found."
            print(f"ERROR: {observation}")
        else:
            try:
                args = parse_tool_arguments(tool_name, tool_input_raw)
                print(f"Parsed Arguments: {args}")
                observation = tools[tool_name](*args)
                print(f"Tool Result: {observation}")
            except Exception as error:
                observation = f"Tool execution error: {error}"
                print(observation)

        # ----------------------------------
        # Update Scratchpad
        # ----------------------------------
        scratchpad += (
            f"{output}\n"
            f"Observation: {observation}\n"
            f"Thought:"
        )

    print("\nMaximum iterations reached without a final answer.")
    return None

# ==========================================
# 7. Run Agent
# ==========================================

if __name__ == "__main__":
    print("Hello Raw ReAct Agent!")
    print()
    result = run_agent("What is the price of a laptop with applying a gold discount?")
    print("\nAgent Result:", result) 
