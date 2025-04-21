import re
import inspect
import os
import importlib.util
import sys

def find_definitions(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find class definitions
    class_pattern = r"class\s+(\w+)"
    classes = re.findall(class_pattern, content)
    
    # Find function definitions
    func_pattern = r"def\s+(\w+)"
    funcs = re.findall(func_pattern, content)
    
    return classes, funcs

def list_functions_in_module(module_path):
    """
    List all functions defined in a module, including their signatures and docstrings.
    """
    # Get the absolute path
    absolute_path = os.path.abspath(module_path)
    
    # Check if file exists
    if not os.path.exists(absolute_path):
        print(f"Error: File does not exist: {absolute_path}")
        return
    
    # Get module name from file path
    module_name = os.path.splitext(os.path.basename(absolute_path))[0]
    
    # Load module from file path
    spec = importlib.util.spec_from_file_location(module_name, absolute_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Get all functions
    functions = inspect.getmembers(module, inspect.isfunction)
    classes = inspect.getmembers(module, inspect.isclass)
    
    # Print all functions
    print(f"\nFunctions in module {module_name}:\n")
    for name, func in functions:
        if func.__module__ == module_name:  # Only functions defined in this module
            signature = inspect.signature(func)
            doc = func.__doc__ or "No docstring provided"
            print(f"{name}{signature}")
            print(f"  {doc.strip()}\n")
    
    # Print all classes and their methods
    print(f"\nClasses in module {module_name}:\n")
    for class_name, cls in classes:
        if cls.__module__ == module_name:  # Only classes defined in this module
            print(f"{class_name}")
            
            # Get class methods
            methods = inspect.getmembers(cls, inspect.isfunction)
            for method_name, method in methods:
                if not method_name.startswith('_'):  # Skip private/special methods
                    signature = inspect.signature(method)
                    doc = method.__doc__ or "No docstring provided"
                    print(f"  .{method_name}{signature}")
                    print(f"    {doc.strip()}\n")

# Path to the file
file_path = "detection_system/model_adapter.py"

try:
    classes, funcs = find_definitions(file_path)
    
    print(f"Classes defined in {file_path}:")
    for cls in classes:
        print(f"- {cls}")
    
    print(f"\nFunctions defined in {file_path}:")
    for func in funcs:
        print(f"- {func}")
        
    # Check if any function contains "fasterrcnn" or "rcnn" in its name
    rcnn_funcs = [f for f in funcs if "fasterrcnn" in f.lower() or "rcnn" in f.lower()]
    if rcnn_funcs:
        print("\nFunctions related to RCNN:")
        for func in rcnn_funcs:
            print(f"- {func}")
    else:
        print("\nNo functions related to RCNN found in this file.")
    
    # Open model_utils.py to check for functions there
    try:
        utils_path = "utils/model_utils.py"
        utils_classes, utils_funcs = find_definitions(utils_path)
        
        print(f"\nFunctions in {utils_path} that may be relevant:")
        for func in utils_funcs:
            if "model" in func.lower() or "rcnn" in func.lower() or "create" in func.lower():
                print(f"- {func}")
    except Exception as e:
        print(f"\nError analyzing utils/model_utils.py: {str(e)}")
    
except Exception as e:
    print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Change this path to the module you want to inspect
    file_path = "detection_system/model_adapter.py"
    list_functions_in_module(file_path) 