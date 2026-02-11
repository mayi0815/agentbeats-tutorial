#!/usr/bin/env python3
"""Test direct de l'agent avec les instructions de debug."""
import sys
import json
sys.path.insert(0, '/Users/yima/Downloads/agentbeats-tutorial/scenarios/webshop/webshop-agent/src')

from agent import Agent

# Créer un agent
agent = Agent()

# Simuler les payloads des épisodes
test_cases = [
    {
        "name": "Episode 1",
        "payload": {
            "observation": "WebShop [SEP] Instruction: Find me hand wash women's sweaters...",
            "instruction": "Find me hand wash women's sweaters with long sleeve, stretch fabric, polyester spandex for teen girls, daily wear with color: xnj-tshirt348-black, and size: large, and price lower than 50.00 dollars",
            "available_actions": {"has_search_bar": True, "clickables": []},
            "step": 0,
        }
    },
    {
        "name": "Episode 3",
        "payload": {
            "observation": "WebShop [SEP] Instruction: Find me machine wash women's fashion hoodies...",
            "instruction": "Find me machine wash, wash cold women's fashion hoodies & sweatshirts for dry clean, tumble dry with color: navy blue, and size: small, and price lower than 80.00 dollars",
            "available_actions": {"has_search_bar": True, "clickables": []},
            "step": 0,
        }
    },
    {
        "name": "Episode 5",
        "payload": {
            "observation": "WebShop [SEP] Instruction: Find me eco friendly throw blankets...",
            "instruction": "Find me eco friendly throw blankets with fleece throw with color: paris eiffel tower1goo8867, and size: 39x59in, and price lower than 70.00 dollars",
            "available_actions": {"has_search_bar": True, "clickables": []},
            "step": 0,
        }
    },
]

print("Testing Agent._select_action directly:")
print()

for test in test_cases:
    print(f"=== {test['name']} ===")
    print(f"Instruction: {test['payload']['instruction'][:60]}...")
    
    # Reset le cache avant chaque test
    agent._session_cache.reset()
    
    # Appeler _select_action
    action = agent._select_action(test['payload'])
    print(f"Action: {action}")
    print()
