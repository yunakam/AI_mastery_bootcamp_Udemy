# Additional Exercise: Copy the contents of one file to another
def copy_file_contents(source_file, destination_file):
    try:
        with open(source_file, "r") as src:
            content = src.read()
        with open(destination_file, "w") as dest:
            dest.write(content)
        print(f"Contents copied from {source_file} to {destination_file}.")
        
    except FileNotFoundError:
        print(f"File not found: {source_file}. Please check the file path.")
    except IOError:
        print(f"An error occurred while reading or writing the file.")
    except PermissionError:
        print(f"Permission denied. You do not have access to {source_file} or {destination_file}.")
        
        
# Example usage
copy_file_contents("sample.txt", "copied_sample.txt")
        