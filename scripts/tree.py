import os

def print_directory_tree(start_path, indent=''):
    # This function remains unchanged
    for item in sorted(os.listdir(start_path)):
        item_path = os.path.join(start_path, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}")
            print_directory_tree(item_path, indent + '    ')
        else:
            print(f"{indent}📄 {item}")

# --- EXAMPLE USAGE ---

# To scan the whole project, you would use '.'
# startverzeichnis = '.'

# To scan ONLY the folder "src" and its contents, you do this:
startverzeichnis = 'saved_models_and_logs'  # Assuming a folder named 'src' exists

# Check if the path exists before starting
if os.path.isdir(startverzeichnis):
    print(f"Scanning only directory: {os.path.abspath(startverzeichnis)}")
    print("-" * 20)
    print_directory_tree(startverzeichnis)
else:
    print(f"Error: The directory '{startverzeichnis}' was not found.")