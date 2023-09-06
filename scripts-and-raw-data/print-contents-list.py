import re
from collections import OrderedDict

def split_csv_line(line):
    """Split a CSV line into cells, handling commas inside double quotes."""
    pattern = r',(?=(?:[^"]*"[^"]*")*[^"]*$)'
    cells = re.split(pattern, line)
    return [cell.strip('"') for cell in cells]  # Strip double quotes from each cell

def rreplace(s, old, new, occurrence):
    """Replace the last occurrence of a substring."""
    li = s.rsplit(old, occurrence)
    return new.join(li)

def format_value(value):
    """Format the value according to the provided template."""
    # Replace the last comma with ', and' for the display value
    if value.count(',') > 0:
        display_value = rreplace(value, ',', ', and', 1)
    else:
        display_value = value
    display_value = display_value.replace('\\,', ',')
    
    # Lowercase the string, replace commas and spaces with hyphens, and add 'and' before the last segment for the link value
    link_value = display_value.lower().replace(', ', '-').replace(',', '-and-').replace(' ', '-')
    
    return f"[{display_value}](#{link_value})"

def process_csv_file(file_path):
    unique_values = {}  # Initialize an empty dictionary to store unique values and their earliest dates
    
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Read the first line (header)
        header_cells = split_csv_line(header)
        print("Header:", header_cells)

        for line in f:
            line = line.strip()  # Remove newline characters
            cells = split_csv_line(line)
            
            # Check if the value is already in the dictionary
            value = cells[4]
            date = cells[3]
            if value not in unique_values:
                unique_values[value] = date
            else:
                if date < unique_values[value]:
                    unique_values[value] = date

    # Sort the unique values based on their earliest date
    sorted_values = OrderedDict(sorted(unique_values.items(), key=lambda x: x[1]))
    
    # Print the formatted unique values
    for value in sorted_values.keys():
        print(format_value(value))

# Specify the path to your CSV file
file_path = "dataset.csv"
process_csv_file(file_path)
