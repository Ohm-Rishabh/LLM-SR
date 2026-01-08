#!/usr/bin/env python3
"""
Unit tests for template equation combining functionality in PySR tool.

This tests the combine_template_equation function which converts PySR's
template sub-expression format into human-readable combined equations.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path to import template_utils module
sys.path.insert(0, str(Path(__file__).parent.parent))
from template_utils import combine_template_equation


class TestTemplateCombine(unittest.TestCase):
    """Test cases for combining template sub-expressions."""

    def test_simple_additive_template(self):
        """Test basic additive template: f(m, m_0) + g(c)"""
        equation = "f = #1 / -1.0746335; g = #1"
        expression_spec = {
            "expressions": ["f", "g"],
            "variable_names": ["m", "m_0", "c"],
            "combine": "f(m, m_0) + g(c)"
        }
        expected = "(m / -1.0746335) + (c)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_nested_trigonometric_template(self):
        """Test nested template with sin: sin(f(x1, x2)) + g(x3)"""
        equation = "f = #1 + #2; g = #1^2"
        expression_spec = {
            "expressions": ["f", "g"],
            "variable_names": ["x1", "x2", "x3"],
            "combine": "sin(f(x1, x2)) + g(x3)"
        }
        expected = "sin((x1 + x2)) + (x3^2)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_exponential_template(self):
        """Test exponential template: exp(f(x1, x2))"""
        equation = "f = #1 * #2"
        expression_spec = {
            "expressions": ["f"],
            "variable_names": ["x1", "x2"],
            "combine": "exp(f(x1, x2))"
        }
        expected = "exp((x1 * x2))"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_multiplicative_template(self):
        """Test multiplicative template: f(x1) * g(x2, x3)"""
        equation = "f = #1^2; g = #1 + #2"
        expression_spec = {
            "expressions": ["f", "g"],
            "variable_names": ["x1", "x2", "x3"],
            "combine": "f(x1) * g(x2, x3)"
        }
        expected = "(x1^2) * (x2 + x3)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_complex_physics_template(self):
        """Test complex template: f(x1) * exp(g(x2)) + h(x3)"""
        equation = "f = #1^2; g = -#1 / 2.5; h = #1 * 3.14"
        expression_spec = {
            "expressions": ["f", "g", "h"],
            "variable_names": ["x1", "x2", "x3"],
            "combine": "f(x1) * exp(g(x2)) + h(x3)"
        }
        expected = "(x1^2) * exp((-x2 / 2.5)) + (x3 * 3.14)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_no_template_returns_original(self):
        """Test that without template, original equation is returned"""
        equation = "x1 + x2 * 3.14"
        expression_spec = None

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, equation)

    def test_newline_separated_equations(self):
        """Test equations separated by newlines instead of semicolons"""
        equation = "f = #1 / 2.0\ng = #1^2"
        expression_spec = {
            "expressions": ["f", "g"],
            "variable_names": ["x1", "x2"],
            "combine": "f(x1) + g(x2)"
        }
        expected = "(x1 / 2.0) + (x2^2)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_single_variable_per_function(self):
        """Test template where each function takes one variable"""
        equation = "f = log(#1); g = sqrt(#1); h = #1^3"
        expression_spec = {
            "expressions": ["f", "g", "h"],
            "variable_names": ["x1", "x2", "x3"],
            "combine": "f(x1) + g(x2) * h(x3)"
        }
        expected = "(log(x1)) + (sqrt(x2)) * (x3^3)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)

    def test_constants_in_subexpressions(self):
        """Test that constants in sub-expressions are preserved"""
        equation = "f = #1 * 2.718 + 1.414; g = #1 / 3.14159"
        expression_spec = {
            "expressions": ["f", "g"],
            "variable_names": ["x", "y"],
            "combine": "f(x) - g(y)"
        }
        expected = "(x * 2.718 + 1.414) - (y / 3.14159)"

        result = combine_template_equation(equation, expression_spec)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
