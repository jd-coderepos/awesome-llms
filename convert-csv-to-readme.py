import re

def split_csv_line(line):
    """Split a CSV line into cells, handling commas inside double quotes."""
    pattern = r',(?=(?:[^"]*"[^"]*")*[^"]*$)'
    return re.split(pattern, line)

def process_csv_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Remove newline characters
            cells = split_csv_line(line)
            escaped_cells = [cell.replace(',', '\\,') for cell in cells]
            print(len(escaped_cells))

# Specify the path to your CSV file
file_path = "dataset.csv"
process_csv_file(file_path)