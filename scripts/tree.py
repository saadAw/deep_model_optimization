import os

def print_directory_tree(start_path, indent=''):
    for item in sorted(os.listdir(start_path)):
        item_path = os.path.join(start_path, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}")
            print_directory_tree(item_path, indent + '    ')
        else:
            print(f"{indent}📄 {item}")

# Beispiel: Startpfad festlegen (z. B. aktuelles Verzeichnis)
startverzeichnis = '.'  # oder ein anderer Pfad
print_directory_tree(startverzeichnis)
