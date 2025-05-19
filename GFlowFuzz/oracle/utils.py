def write_to_file(content: str, file_name: str) -> None:
    """
    Write content to a file.
    
    Args:
        content: Content to write
        file_name: File path to write to
    """
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Error writing to file {file_name}: {e}")