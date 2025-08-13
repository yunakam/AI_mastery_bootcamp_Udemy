import os

# File to store tasks
FILE_NAME = "tasks.txt"


# Load tasks from file
def load_tasks():
    tasks = {}
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r") as file:
            for line in file:  # each line is a task
                task_id, task, status = line.strip().split(" | ")
                tasks[int(task_id)] = {"title": task, "status": status}
    return tasks


# Save tasks into file
def save_tasks(tasks):
    with open(FILE_NAME, "w") as file:
        for task_id, task in tasks.items():
            file.write(f"{task_id} | {task['title']} | {task['status']}\n")

            
# Add a new task
def add_task(tasks):
    title = input("Enter task title: ")
    task_id = max(tasks.keys(), default=0) + 1  # Get the next task ID
    tasks[task_id] = {"title": title, "status": "Pending"}
    save_tasks(tasks)
    print(f"Task '{title}' added. (ID: {task_id})")
    
    
# View all tasks
def view_tasks(tasks):
    if not tasks:
        print("No tasks available.")
        return
    else:
        print("\n=== Task List ===")
        for task_id, task in tasks.items():
            print(f"[{task_id}] {task['title']} - {task['status']}")
            

# Update a task's status
def update_task(tasks):
    task_id = int(input("Enter task ID to update: "))
    if task_id in tasks:
        task = tasks[task_id]
        if task["status"] == "Completed":            
            print(f"Task [{task_id}] is completed. Do you want to mark it as Pending? (yes:1 / no:0)")
            if input() != "1":
                task["status"] = "Pending"
                print(f"[{task_id}] {task['title']} marked as Pending.")
                save_tasks(tasks)
            elif input() == "0":
                print("Update cancelled.")
            else:
                print("Invalid input. Update cancelled.")
        elif task["status"] == "Pending":
            print(f"Did you complete Task [{task_id}]? (yes:1 / no:0)")
            if input() == "1":
                task["status"] = "Completed"
                print(f"[{task_id}] {task['title']} marked as Completed.")
                save_tasks(tasks)
            elif input() == "0":
                print("Update cancelled.")
            else:
                print("Invalid input. Update cancelled.")
    else:
        print(f"Task ID {task_id} not found.")
        
        
# Delete a task
def delete_task(tasks):
    task_id = int(input("Enter task ID to delete: "))
    if task_id in tasks:
        del tasks[task_id]
        save_tasks(tasks)
        print(f"Task ID {task_id} deleted.")
    else:
        print(f"Task ID {task_id} not found.")
        
        
# Main function to run the task manager
def main():
    tasks = load_tasks()
    
    while True:
        print("\n=== TASK MANAGER ========")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Update Task")
        print("4. Delete Task")
        print("5. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == "1":
            add_task(tasks)
        elif choice == "2":
            view_tasks(tasks)
        elif choice == "3":
            update_task(tasks)
        elif choice == "4":
            delete_task(tasks)
        elif choice == "5":
            print("Exiting Task Manager.")
            break
        else:
            print("Invalid choice. Please try again.")
            
if __name__ == "__main__":
    main()