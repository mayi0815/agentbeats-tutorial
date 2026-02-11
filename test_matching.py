#!/usr/bin/env python3
"""Test color/size matching."""
import sys
sys.path.insert(0, 'scenarios/webshop/webshop-agent/src')
from agent import Agent

agent = Agent()

# Test Episode 3 with navy blue
instruction = "Find me machine wash, wash cold women's fashion hoodies & sweatshirts for dry clean, tumble dry with color: navy blue, and size: small, and price lower than 80.00 dollars"
color = agent._desired_option_value(instruction, 'color')
size = agent._desired_option_value(instruction, 'size')

print('Instruction:', instruction[:70], '...')
print('Desired color:', repr(color))
print('Desired size:', repr(size))

# Simulate available options
color_options = ['bass purple', 'blue', 'deep blue', 'green', 'gray', 'wine red', 'dark gray', 'navy', 'pink']
size_options = ['small', 'medium', 'large', 'x-large', 'xx-large', '3x-large', '4x-large']
clickables = color_options + size_options + ['buy now']

# Debug decomposition
print('\nDebugging color matching:')
print('  _normalize(color):', repr(agent._normalize(color)))
print('  _split_compound_color(color):', agent._split_compound_color(color))

for opt in color_options:
    print(f'  Option "{opt}": norm={agent._normalize(opt)}, parts={agent._split_compound_color(opt)}')

matched_color = agent._match_option_value(color, color_options, clickables)
matched_size = agent._match_option_value(size, size_options, clickables)

print()
print('Available colors:', color_options)
print('Matched color:', repr(matched_color))
print()
print('Available sizes:', size_options)
print('Matched size:', repr(matched_size))
