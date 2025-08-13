# Additional Exercise: Log messages with timestamps into a file

from datetime import datetime

def log_message(file_name, message):
    try:
        with open(file_name, "a") as file:
            timestamp_raw = datetime.now() # Get the raw timestamp - 2025-08-11 21:39:23.991566
            timestamp = timestamp_raw.strftime("%Y-%m-%d %H:%M:%S") # Format the timestamp - 2025-08-11 21:39:23
            file.write(f"{timestamp} - {message}\n")
        print(f"Message logged: {message}")
    except IOError:
        print("An error occurred while writing to the file.")
    except PermissionError:
        print("Permission denied. You do not have access to this file.")
        
# Example usage
log_message("log.txt", "This is a log message.")