import re
from collections import defaultdict

def split_csv_line(line):
    """Split a CSV line into cells, handling commas inside double quotes."""
    pattern = r',(?=(?:[^"]*"[^"]*")*[^"]*$)'
    cells = re.split(pattern, line)
    return [cell.strip('"') for cell in cells]  # Strip double quotes from each cell

def process_csv_file(file_path):
    descriptions = defaultdict(list)  # Dictionary to store lists of structured descriptions indexed by organization name
    
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Read the first line (header)
        header_cells = split_csv_line(header)

        for line in f:
            line = line.strip()  # Remove newline characters
            cells = split_csv_line(line)
            
            organization = cells[4]
            # Ensure we only loop up to the minimum of the length of header_cells and cells
            description = "\n".join([f"  {header_cells[i]}: {cells[i]}" for i in range(min(len(header_cells), len(cells))) if header_cells[i] and cells[i]])
            
            descriptions[organization].append(description)

    return descriptions

# Specify the path to your CSV file
file_path = "dataset.csv"
descriptions_dict = process_csv_file(file_path)

# Write the descriptions to a file
with open('output.md', 'w', encoding='utf-8') as f:
    for org, desc_list in descriptions_dict.items():
        f.write(f"#### {org}\n\n")
        for desc in desc_list:
            f.write(f"  ```yaml\n{desc}\n  ```\n\n")
        f.write('-'*40 + '\n')
