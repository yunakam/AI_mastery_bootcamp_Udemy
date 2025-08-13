with open("sample.txt", "w") as file:
    file.write("yeah!\n")
    file.writelines("Line 1\nLine 2\nLine 3\n")

with open("sample.txt", "r") as file:
    content = file.read()
    print(content)
    
# Exception Handling
try:
    with open("sample.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found. Please check the file path.")
except IOError:
    print("An error occurred while reading the file.")
except PermissionError:
    print("Permission denied. You do not have access to this file.")
    

# Exercise 1
def count_words_and_lines(file_name):
    try:
        with open(file_name, "r") as file:
            lines = file.readlines()
            line_count = len(lines)
            word_count = sum(len(line.split()) for line in lines)
            # return len(word_count), len(line_count)
            print(f"Number of lines: {line_count}, Number of words: {word_count}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None, None
    except IOError:
        print("An error occurred while reading the file.")
        return None, None

count_words_and_lines("sample.txt")