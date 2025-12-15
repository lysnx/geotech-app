import os

# Read the current file
with open('src/ui/components.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove BOM if present
if content.startswith('\ufeff'):
    content = content[1:]

# Replace escaped quotes with regular quotes
content = content.replace('\\"', '"')

# Write back
with open('src/ui/components.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('Fixed components.py')
