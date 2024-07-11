# Identify and remove duplicates from the MathInstruct dataset
import json
from collections import Counter

# Load the dataset
with open('MathInstruct-500.json', 'r') as file:
    data = json.load(file)

# Extract instructions
instructions = [item['instruction'] for item in data]

# Count occurrences of each instruction
instruction_counts = Counter(instructions)

# Find duplicates
duplicates = {instruction: count for instruction, count in instruction_counts.items() if count > 1}

# Filter out items with duplicate instructions
filtered_data = [item for item in data if instruction_counts[item['instruction']] == 1]

# Save the new dataset
with open('NewMathInstruct.json', 'w') as new_file:
    json.dump(filtered_data, new_file, indent=4)

# Output the number of instructions that are shown more than once
print(len(duplicates))
