from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage , SystemMessage , ToolMessage
from langsmith import traceable


MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

#----- Tools (langchain tool decorator)--#

@tool
def get_product_price(product: str) -> str:
        """ Look up the price of a product in the catalog """
        print(f"  >>Ececute get_product_price(product = {product} )")
        prices = {
            "laptop": 999,
            "smartphone": 699,
            "headphones": 5678 ,
            "tablet": 499 
        }
        return prices.get(product.lower(), "Product not found")

@tool
def apply_discount(price: float , discount_tier: str) -> float:
        """ Apply a discount to a price
         Available tiers: bronze , silver , gold"""
        print(f"  >>Ececute apply_discount(price = {price}, discount = {discount_tier})")
        discount_percentages = {
                "broze": 5,
                "silver": 12 , 
                "gold": 23
        }
        discount =  discount_percentages.get(discount_tier ,  0)
        return round(price * (1 - discount/100) , 2)

@traceable(name= "langchain agent loop")
def run_agent(question:str):
        tools = [get_product_price , apply_discount]
        tools_dict = {t.name: t for t in tools}
        llm = init_chat_model(f"openai:{MODEL}" , temperature = 0)
        llm_with_tools = llm.bind_tools(tools)

        print(f"Question:{question}")
        print("=" * 60)
        messages = [
              SystemMessage(
                      content = (
                              "You are a helpful assistant that can answer questions about product prices and apply discounts. "
                              "You have access to the following tools: "
                              "1. get_product_price(product: str) -> str: Look up the price of a product in the catalog. "
                              "2. apply_discount(price: float, discount_tier: str) -> float: Apply a discount to a price. "
                              "Available tiers: bronze, silver, gold."
                              "STRICT RULES - YOU must follow these rules: " 
                              "- Only use the tools provided. "
                              "- Always call a tool before responding to the user. "
                              "- If a tool is not available, inform the user. "
                              "- Do not make up prices or discounts. "
                              "Do not assume any information that is not provided by the tools. "
                              "- If you need to perform a calculation, use the apply_discount tool. "
                      )
              ) ,
              HumanMessage(content = question), 
        ]
        for iteration in range(1 , MAX_ITERATIONS+1):
                print(f"Iteration {iteration} --------------------")
                ai_message = llm_with_tools.invoke(messages)
                tool_calls = ai_message.tool_calls
                if not tool_calls:
                        print(f"\nFinal Answer: {ai_message.content}")
                        return ai_message.content
                tool_call = tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                print(f"Tool Call: {tool_name} with args: {tool_args}")
                tool_to_use = tools_dict.get(tool_name)
                if tool_to_use is None:
                       raise ValueError(f"Tool {tool_name} not found")
                observation = tool_to_use.invoke(tool_args)
                print(f"[Tool Result]: {observation}")

                messages.append(ai_message)
                messages.append(
                        ToolMessage(
                                content=str(observation),
                                tool_call_id=tool_call_id
                        )
                )
        print("Max iterations reached without a final answer.")
        return None
if __name__ == "__main__":
        print("hello Langchain Agents (.bind_tools)!")
        print()
        result = run_agent("What is the price of a laptop with applying a gold discount?")
