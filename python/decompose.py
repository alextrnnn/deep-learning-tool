import re
import json
import numpy as np

def simplify_expression(expr, forward_values):
    # Remove spaces from the input
    expr = expr.replace(" ", "")
    
    # Define patterns for each level of precedence in the correct order
    operators = [
        (r'(\w+\s*\^\s*\w+)', '^'),     # Power 
        (r'(\w+\s*[\*/]\s*\w+)', '*/'), # Multiplication and Division
        (r'(\w+\s*[\+-]\s*\w+)', '+-')  # Addition and Subtraction
    ]
    
    intermediate_steps = []
    intermediate_var = "v"
    index = 1
    
    nodes = []
    links = []
    backward_gradients = {}

    # Helper function to compute an operation
    def compute_operation(op, left, right):
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right
        elif op == '^':
            return left ** right
        else:
            raise ValueError(f"Unsupported operator: {op}")

    # Add user-provided variables to the graph nodes
    for var, value in forward_values.items():
        nodes.append({
            "id": var,
            "label": var,
            "forward_value": value,
            "backward_gradient": 0.0  # Initialize gradient
        })

    # Helper function to process a simple expression without parentheses
    def process_simple_expression(simple_expr):
        nonlocal index
        for pattern, ops in operators:
            while re.search(pattern, simple_expr):
                match = re.search(pattern, simple_expr)
                sub_expr = match.group(0)
                var_name = f"{intermediate_var}{index}"
                
                # Parse operation
                for op in ops:
                    if op in sub_expr:
                        operands = sub_expr.split(op)
                        left, right = operands[0].strip(), operands[1].strip()
                        operation = op
                        break
                
                # Compute forward pass value
                forward_values[var_name] = compute_operation(operation, forward_values[left], forward_values[right])
                
                # Avoid duplicate variable by only creating a new one if necessary
                if sub_expr not in [step.split(" = ")[1] for step in intermediate_steps]:
                    intermediate_steps.append(f"{var_name} = {sub_expr}")
                    
                    # Add the node for the new variable
                    nodes.append({
                        "id": var_name, 
                        "label": sub_expr, 
                        "forward_value": forward_values[var_name],
                        "backward_gradient": 0.0  # Initialize gradient
                    })
                    
                    # Add links from operands to the new variable
                    for operand in [left, right]:
                        links.append({"source": operand, "target": var_name})
                    
                    # Replace the expression with the variable
                    simple_expr = simple_expr[:match.start()] + var_name + simple_expr[match.end():]
                    index += 1
                else:
                    # Replace with the existing variable name
                    existing_var = [step.split(" = ")[0] for step in intermediate_steps if step.split(" = ")[1] == sub_expr][0]
                    simple_expr = simple_expr[:match.start()] + existing_var + simple_expr[match.end():]
                    
        return simple_expr

    # Main function to handle parentheses by processing innermost expressions first
    while '(' in expr:
        # Find innermost parenthesis expression
        inner_expr = re.search(r'\([^()]+\)', expr).group(0)
        # Remove parentheses
        inner_expr_content = inner_expr[1:-1]
        # Process the content inside the parentheses
        result = process_simple_expression(inner_expr_content)
        # Replace the parentheses expression in the main expression with its result
        expr = expr.replace(inner_expr, result, 1)
    
    # Finally, process any remaining expression outside parentheses
    expr = process_simple_expression(expr)
    

    # Add backward pass computation
    backward_gradients[expr] = 1.0  # Initialize gradient for result
    for step in reversed(intermediate_steps):
        var_name, sub_expr = step.split(" = ")
        for op in "+-*/^":
            if op in sub_expr:
                left, right = sub_expr.split(op)
                left, right = left.strip(), right.strip()
                
                if left not in backward_gradients:
                    backward_gradients[left] = 0.0
                
                if right not in backward_gradients:
                    backward_gradients[right] = 0.0
                
                # Propagate gradients based on operation
                if op == '+':
                    backward_gradients[left] += 1
                    backward_gradients[right] += 1
                elif op == '-':
                    backward_gradients[left] += 1
                    backward_gradients[right] += - 1
                elif op == '*':
                    backward_gradients[left] += forward_values[right]
                    backward_gradients[right] += forward_values[left]
                elif op == '/':
                    backward_gradients[left] += 1.0 / forward_values[right]
                    backward_gradients[right] += -forward_values[left] / (forward_values[right] ** 2)
                elif op == '^':
                    backward_gradients[left] += forward_values[right] * (forward_values[left] ** (forward_values[right] - 1))
                    backward_gradients[right] += forward_values[left] ** forward_values[right] * np.log(forward_values[left])
                backward_gradients[left] *= backward_gradients[var_name]
                backward_gradients[right] *= backward_gradients[var_name]

    # Update gradients in nodes
    for node in nodes:
        node["backward_gradient"] = backward_gradients.get(node["id"], 0.0)
    
    # Create the output JSON
    graph = {"nodes": nodes, "links": links}
    return graph

# Get the expression from the user
expression = input("Enter an expression: ")

# Extract variables from the expression
variables = set(re.findall(r'\b[a-zA-Z_]\w*\b', expression))

# Prompt the user for variable values
forward_values = {}
for var in variables:
    forward_values[var] = float(input(f"Enter value for {var}: "))

# Simplify and process the expression
graph_data = simplify_expression(expression, forward_values)

# Save the graph data to a JSON file
with open("graph_data.json", "w") as json_file:
    json.dump(graph_data, json_file, indent=2)

print("Graph data saved to graph_data.json")
