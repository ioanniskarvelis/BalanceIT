from django import template
import json
import html
import ast

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter(name='json_decode')
def json_decode(value):
    if not value:
        return {"error": "Empty value"}
    if isinstance(value, dict):
        return value
    try:
        # First, decode HTML entities
        html_decoded = html.unescape(str(value))
        # Then, try to parse as a Python literal
        python_dict = ast.literal_eval(html_decoded)
        # Convert Python dict to JSON-compatible dict
        return json.loads(json.dumps(python_dict))
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}", "value": value}
    except (ValueError, SyntaxError) as e:
        return {"error": f"Python literal eval error: {str(e)}", "value": value}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "value": value}