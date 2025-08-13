# Exercise 2
def write_items_to_file(file_name, items):
    with open(file_name, "w") as file:
        for item in items:
            file.write(f"{item}\n")

def read_items_from_file(file_name):
    try:
        with open(file_name, "r") as file:
            items = file.readlines()
            print("Items in the file:")
            for item in items:
                print(item.strip())
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except IOError:
        print("An error occurred while reading the file.")
    except PermissionError:
        print("Permission denied. You do not have access to this file.")

fruits = ["Apple", "Banana", "Cherry", "Date"]
read_items_from_file("fruits.txt")
write_items_to_file("fruits.txt", fruits)
read_items_from_file("fruits.txt")