#!/usr/bin/env python3
"""
Utility functions for working with PySR template expressions.

This module provides helper functions for combining template sub-expressions
into human-readable equations.
"""

import sys
import re


def combine_template_equation(equation_str, expression_spec_dict):
    """
    Combine template sub-expressions into a final human-readable equation.

    When using TemplateExpressionSpec, PySR returns equations like:
        "f = #1 + #2; g = #1^2"

    where #1, #2, etc. refer to the arguments specified in the combine field.

    This function substitutes the sub-expressions into the combine template to get:
        "sin(x1 + x2) + x3^2" (if combine was "sin(f(x1, x2)) + g(x3)")

    Args:
        equation_str: String containing sub-expression definitions with placeholders (e.g., "f = #1 + #2; g = #1")
        expression_spec_dict: The expression_spec dictionary with 'expressions', 'variable_names', 'combine'

    Returns:
        Combined equation string, or original if parsing fails
    """
    if not expression_spec_dict:
        return equation_str

    try:
        # Parse the equation string to extract sub-expressions
        # Format: "f = <expr>; g = <expr>; ..." or "f = <expr>\ng = <expr>"
        parts = equation_str.replace('\n', '; ').split(';')
        sub_exprs = {}

        for part in parts:
            part = part.strip()
            if '=' not in part:
                continue
            name, expr = part.split('=', 1)
            name = name.strip()
            expr = expr.strip()
            sub_exprs[name] = expr

        # Get the combine template
        combine_template = expression_spec_dict['combine']

        # Replace each sub-expression in the combine template
        result = combine_template
        for expr_name in expression_spec_dict['expressions']:
            if expr_name not in sub_exprs:
                continue

            sub_expr = sub_exprs[expr_name]

            # Replace #1, #2, etc. with variable names from the combine field
            # Example: if combine = "f(m, m_0) + g(c)", then for function f:
            #   - #1 refers to the 1st argument 'm' in f(m, m_0)
            #   - #2 refers to the 2nd argument 'm_0' in f(m, m_0)
            # We extract the arguments for this specific function and map placeholders accordingly

            # Use a more robust pattern that handles nested parentheses
            # We need to find expr_name followed by parentheses and extract arguments
            def find_function_call(template, func_name):
                """Find function call and extract arguments, handling nested parentheses."""
                # Use word boundary to avoid matching partial names (e.g., 'g' in 'log')
                pattern = rf'\b{re.escape(func_name)}\s*\('
                match = re.search(pattern, template)
                if not match:
                    return None

                start = match.end()
                paren_count = 1
                i = start

                while i < len(template) and paren_count > 0:
                    if template[i] == '(':
                        paren_count += 1
                    elif template[i] == ')':
                        paren_count -= 1
                    i += 1

                if paren_count == 0:
                    args_str = template[start:i-1]
                    return args_str
                return None

            args_str = find_function_call(combine_template, expr_name)

            if args_str:
                # Split arguments (simple split by comma - may need improvement for nested commas)
                args = [arg.strip() for arg in args_str.split(',')]

                # Replace #i with the corresponding argument from the combine field
                # #1 refers to the 1st argument in the function call, #2 to the 2nd, etc.
                for i, arg in enumerate(args, 1):
                    placeholder = f'#{i}'
                    # Use the variable name directly from the combine template
                    sub_expr = sub_expr.replace(placeholder, arg)

            # Now replace the function call with the substituted expression
            # Use the robust function to replace the entire call
            def replace_function_call(template, func_name, replacement):
                """Replace function call with replacement, handling nested parentheses."""
                # Use word boundary to avoid matching partial names
                pattern = rf'\b{re.escape(func_name)}\s*\('
                match = re.search(pattern, template)
                if not match:
                    return template

                start_pos = match.start()
                paren_start = match.end()
                paren_count = 1
                i = paren_start

                while i < len(template) and paren_count > 0:
                    if template[i] == '(':
                        paren_count += 1
                    elif template[i] == ')':
                        paren_count -= 1
                    i += 1

                if paren_count == 0:
                    # Replace from start_pos to i with replacement
                    return template[:start_pos] + replacement + template[i:]
                return template

            result = replace_function_call(result, expr_name, f'({sub_expr})')

        return result

    except Exception as e:
        print(f"Warning: Failed to combine template equation: {e}", file=sys.stderr)
        print(f"  Returning original equation: {equation_str}", file=sys.stderr)
        return equation_str
