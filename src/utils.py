from datetime import datetime

def add_to_log(message: str):
    """
    Prints messages to the screen, as Python print function but with the incident time.

    Args:
        message (str): The script to be printed.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
