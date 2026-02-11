# Test _build_search_query with actual instructions from debug logs
import sys
sys.path.insert(0, '/Users/yima/Downloads/agentbeats-tutorial/scenarios/webshop/webshop-agent/src')

from agent import Agent

agent = Agent()

instructions = [
    ("Episode 1", "Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls, daily wear with color: xnj-tshirt348-black, and size: large, and price lower than 50.00 dollars"),
    ("Episode 2", "Find me men's t-shirts & tanks with short sleeve, fashion design, long sleeve, button closure with color: b-gray, and size: x-large, and price lower than 30.00 dollars"),
    ("Episode 3", "Find me machine wash, wash cold women's fashion hoodies & sweatshirts for dry clean, tumble dry with color: navy blue, and size: small, and price lower than 80.00 dollars"),
    ("Episode 4", "Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: mint spring, and size: xx-large tall, and price lower than 60.00 dollars"),
    ("Episode 5", "Find me eco friendly throw blankets with fleece throw with color: paris eiffel tower1goo8867, and size: 39x59in, and price lower than 70.00 dollars"),
]

print("Testing _build_search_query with debug log instructions:")
print()
for name, instruction in instructions:
    query = agent._build_search_query(instruction)
    print(f"{name}:")
    print(f"  Instruction: {instruction[:80]}...")
    print(f"  Search query: '{query}'")
