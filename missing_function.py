"""
This script directly checks if there's an implementation of create_custom_fasterrcnn_with_bn
or if it's imported from elsewhere but not defined in the project.
"""
import inspect
import re
import importlib
import sys
import ast
import os

def get_function_source(func):
    try:
        return inspect.getsource(func)
    except Exception as e:
        return f"Error getting source: {str(e)}"

def find_create_function_in_module(module_path):
    try:
        # Read file content
        with open(module_path, 'r') as f:
            content = f.read()
        
        # Parse the code
        tree = ast.parse(content)
        
        # Find function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and 'create_custom_fasterrcnn_with_bn' in node.name:
                # Get the function source code
                function_lines = content.split('\n')[node.lineno-1:node.end_lineno]
                return '\n'.join(function_lines)
                
        return "Function not found in module."
    except Exception as e:
        return f"Error analyzing module: {str(e)}"

# Try to import the function directly
try:
    from detection_system.model_adapter import create_custom_fasterrcnn_with_bn
    print("Function found via import!")
    print(get_function_source(create_custom_fasterrcnn_with_bn))
except ImportError as e:
    print(f"Import error: {str(e)}")
    # If import fails, try to find the function in the file
    file_path = "detection_system/model_adapter.py"
    if os.path.exists(file_path):
        print(f"\nSearching in {file_path}...")
        result = find_create_function_in_module(file_path)
        print(result)
    else:
        print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error: {str(e)}")

# Search in related modules
try:
    import detection_system
    print("\nOther related functions:")
    for name in dir(detection_system):
        if "create" in name.lower() or "faster" in name.lower():
            print(f"- {name}")
except Exception as e:
    print(f"Error exploring module: {str(e)}") 