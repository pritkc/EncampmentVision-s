import os
import ast

def find_function_definition(module_path, function_name):
    """Find the implementation of a function in a module."""
    try:
        with open(module_path, 'r') as file:
            source_code = file.read()
        
        # Parse the source code
        tree = ast.parse(source_code)
        
        # Find the function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                lineno = node.lineno
                # Get the last line number of the function
                end_lineno = find_function_end(source_code, lineno)
                return lineno, end_lineno, source_code.split('\n')[lineno-1:end_lineno]
        
        return None, None, "Function definition not found."
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def find_function_end(source_code, start_line):
    """Find the end line of a function definition."""
    lines = source_code.split('\n')
    indent_level = None
    
    for i, line in enumerate(lines[start_line:], start=start_line):
        # Skip empty lines or comments at the beginning
        if line.strip() == '' or line.strip().startswith('#'):
            continue
        
        # Determine the indentation level of the function
        if indent_level is None:
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # If we find a line with same or less indentation, that's the end
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= indent_level and line.strip() != '':
            return i
    
    # If we reach the end of the file
    return len(lines)

def search_in_file(file_path):
    """Search for the create_custom_fasterrcnn_with_bn function in the given file."""
    from detection_system.model_adapter import create_custom_fasterrcnn_with_bn
    
    # Call the function to get its implementation
    start, end, implementation = find_function_definition(file_path, "create_custom_fasterrcnn_with_bn")
    
    if start:
        print(f"\nFunction found in {file_path} at lines {start}-{end}:\n")
        print('\n'.join(implementation))
        return True
    else:
        print(f"\nFunction not found in {file_path}")
        return False

def main():
    # Define the paths to search in
    file_path = "detection_system/model_adapter.py"
    
    # Search directly in the known file
    if os.path.exists(file_path):
        if search_in_file(file_path):
            return
    
    # Search in multiple directories if the direct search failed
    directories_to_search = ["detection_system", "utils", "models"]
    
    for directory in directories_to_search:
        if not os.path.exists(directory):
            continue
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if search_in_file(file_path):
                        return

    print("\nFunction implementation not found in any of the searched locations.")

if __name__ == "__main__":
    main() 