import re
import inspect
 
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

# ----- Tools (langchain tool decorator)--#


@traceable(name="get_product_price")
def get_product_price(product: str) -> str:
    """Look up the price of a product in the catalog"""
    print(f"  >>Ececute get_product_price(product = {product} )")
    prices = {"laptop": 999, "smartphone": 699, "headphones": 5678, "tablet": 499}
    return prices.get(product.lower(), "Product not found")


@traceable(name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price
    Available tiers: bronze , silver , gold"""
    print(f"  >>Ececute apply_discount(price = {price}, discount = {discount_tier})")
    discount_percentages = {"broze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)



tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_description(tool_dict) -> str:
    """Get a description of the tools available"""
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        print(f"Signature for {tool_name}: {signature}")
        param_list = ", ".join(f"{name}: {param.annotation.__name__} == {param}" for name, param in signature.parameters.items())
        print(f"Parameters for {tool_name}: {param_list}")
        docstring = inspect.getdoc(original_function)
        print(f"Docstring for {tool_name}: {docstring}")
        if docstring:
            # Remove newlines and extra spaces
            docstring = re.sub(r"\s+", " ", docstring.strip())
            descriptions.append(f"{tool_name}({param_list}): {docstring}")
    return f"Available tools:\n{'\n'.join(descriptions)}"

print(get_tool_description(tools))
