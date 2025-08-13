import os
print(os.getcwd())  # Get the current working directory
# os.mkdir("test_dir1")  # Create a new directory 
# os.mkdir("test_dir2")
# os.remove("test_dir1")  # Remove a directory
# os.rename("test_dir2", "test_dir2_renamed")  # Rename a directory


# Create directories named "SectionXX"
# for i in range(2, 44):
#     dir_name = f"Section{i:02}"  # Format the number with leading zeros
#     os.mkdir(dir_name)


# Replace a string in file names within the current directory
def replace_in_filenames(directory, old_str, new_str):
    for filename in os.listdir(directory):
        if old_str in filename:
            new_filename = filename.replace(old_str, new_str)
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed: {filename} -> {new_filename}")

replace_in_filenames(os.getcwd(), "2-9", "Day7")


import sys
# print(sys.version)  # Print the Python version
# print(sys.path)  # Print the Python path
# print(sys.argv)