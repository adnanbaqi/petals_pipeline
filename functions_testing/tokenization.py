import sys
import re

def count_tokens(file_path):
    # Regular expression to match words (identifiers, keywords), numbers, strings, and common operators/symbols
    token_pattern = r'\b\w+\b|"[^"]*"|\'[^\']*\'|\d+|[+\-*/=<>!&|%^~]+'
    
    token_count = 0
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            tokens = re.findall(token_pattern, content)
            token_count = len(tokens)
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return token_count

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    num_tokens = count_tokens(file_path)
    print(f"Number of Tokens: {num_tokens}")
