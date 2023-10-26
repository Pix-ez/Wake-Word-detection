import os
import csv

# Define the paths to the two folders
folder1_path = "D:\\ml\\wake\\dataset\\1"
folder2_path = "D:\\ml\\wake\\dataset\\0"

# Define the name of the CSV file to create
csv_file_name = "data.csv"

# Initialize a list to store the data
data = []

# Function to collect data from a folder
def collect_data_from_folder(folder_path, label):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_location = os.path.join(root, file_name)
            data.append({"name": file_name, "location": file_location, "label": label})

# Collect data from the first folder with label "folder1"
collect_data_from_folder(folder1_path, "1")

# Collect data from the second folder with label "folder2"
collect_data_from_folder(folder2_path, "0")

# Write the data to a CSV file
with open(csv_file_name, mode='w', newline='') as csv_file:
    fieldnames = ["name", "location", "label"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header row
    writer.writeheader()
    
    # Write the data rows
    for row in data:
        writer.writerow(row)

print(f"CSV file '{csv_file_name}' created successfully.")
