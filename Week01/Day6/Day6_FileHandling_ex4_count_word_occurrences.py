# Additional Exercise: Count the number of occurrences of a specific word in a file
def count_word_occurrences(file_name, word):
    try:
        with open(file_name, "r") as file:
            content = file.read()
            word_count = content.lower().split().count(word.lower())
            if word_count == 0:
                print(f"The word '{word}' does not occur in the file '{file_name}'.")
            else:
                print(f"The word '{word}' occurs {word_count} times in the file '{file_name}'.")
    except FileNotFoundError:
        print(f"File not found: {file_name}. Please check the file path.")
    except IOError:
        print("An error occurred while reading the file.")
    except PermissionError:
        print("Permission denied. You do not have access to this file.")        
        
# Example usage
count_word_occurrences("sample.txt", "line")
count_word_occurrences("sample.txt", "hello")