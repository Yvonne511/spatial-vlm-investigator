import json
import os

# Function to calculate task-specific accuracy, including correct and total counts
def calculate_task_accuracy(data):
    task_results = {}
    
    # Iterate through each result and calculate the accuracy for each task
    for result in data["results"]:
        task = result["task"]
        is_correct = result["is_correct"]
        
        # Initialize task if not present in task_results
        if task not in task_results:
            task_results[task] = {"correct": 0, "total": 0}
        
        # Update task-specific counts
        task_results[task]["total"] += 1
        if is_correct:
            task_results[task]["correct"] += 1
    
    # Calculate accuracy for each task and store it in a dictionary
    task_accuracy = {}
    for task, counts in task_results.items():
        accuracy = counts["correct"] / counts["total"]
        accuracy_percentage = accuracy * 100  # Convert to percentage
        task_accuracy[task] = {
            "correct": counts["correct"],  # Count of correct answers for the task
            "total": counts["total"],      # Total number of answers for the task
            "accuracy": accuracy,          # Accuracy for the task (decimal)
            "accuracy_percentage": f"{accuracy_percentage:.2f}%"  # Accuracy as percentage (2 decimal places)
        }
    
    return task_accuracy

# Path to the "checkpoints copy" folder and the "checkpoints" folder
input_folder = "checkpoints copy"  # Folder containing the original files
output_folder = "checkpoints"      # Folder where updated files will be saved

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each file in the "checkpoints copy" folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Check if the file is a JSON file
    if filename.endswith(".json"):
        print(f"Processing file: {filename}")

        # Read the JSON data from the file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Calculate the task-specific accuracy
        task_accuracy = calculate_task_accuracy(data)

        # Reorder the fields so "task_accuracy" appears second
        output_data = {
            "accuracy": data["accuracy"],  # Keep the original "accuracy" field first
            "task_accuracy": task_accuracy,  # "task_accuracy" as the second field
            "total": data["total"],  # Keep the "total" field after
            "correct": data["correct"],  # Keep the "correct" field after
            "last_updated": data["last_updated"],
            "processed_batches": data["processed_batches"],
            "results": data["results"]  # Keep the "results" field last
        }

        # Write the updated data to the corresponding file in the "checkpoints" folder
        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, "w") as output_file:
            json.dump(output_data, output_file, indent=4)

        print(f"Task-specific accuracy has been written to {output_file_path}")
