import re

def simplify_expression(expr):
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
    
    # Helper function to process a simple expression without parentheses
    def process_simple_expression(simple_expr):
        nonlocal index
        for pattern, ops in operators:
            while re.search(pattern, simple_expr):
                match = re.search(pattern, simple_expr)
                sub_expr = match.group(0)
                var_name = f"{intermediate_var}{index}"
                
                # Avoid duplicate variable by only creating a new one if necessary
                if sub_expr not in [step.split(" = ")[1] for step in intermediate_steps]:
                    intermediate_steps.append(f"{var_name} = {sub_expr}")
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
    
    # Add the final result
    intermediate_steps.append(expr)
    
    # Join and return all intermediate steps
    return "\n".join(intermediate_steps)

expression = input("Enter an expression:")
print(simplify_expression(expression))
