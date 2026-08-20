import os
import sys
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from google import genai
from google.genai import types

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load from .env file if present
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")

if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                clean_key = key.strip().lstrip("\ufeff")
                clean_val = val.strip().strip("'\"")
                os.environ[clean_key] = clean_val

app = Flask(__name__)

# ============================================================
# TOOL DEFINITIONS (PYTHON FUNCTIONS)
# ============================================================

def add_numbers(a: float, b: float) -> dict:
    """Adds two numbers together and returns the calculation breakdown."""
    result = float(a) + float(b)
    return {
        "operation": "addition",
        "a": a,
        "b": b,
        "result": result
    }


def product_info(product_name: str) -> dict:
    """Retrieves detailed product pricing and category information by product name."""
    products = {
        "iphone 15": {
            "name": "iPhone 15",
            "category": "Smartphone",
            "price": 69999,
            "currency": "INR"
        },
        "samsung s24": {
            "name": "Samsung Galaxy S24",
            "category": "Smartphone",
            "price": 74999,
            "currency": "INR"
        },
        "macbook air": {
            "name": "MacBook Air",
            "category": "Laptop",
            "price": 99999,
            "currency": "INR"
        }
    }
    product = products.get(product_name.lower())
    if product:
        return product
    return {
        "error": f"Product '{product_name}' not found."
    }


available_functions = {
    "add_numbers": add_numbers,
    "product_info": product_info
}

# OpenAI schema format for OpenRouter
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Adds two numbers together and returns the calculation breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "product_info",
            "description": "Retrieves detailed product pricing and category information by product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name e.g. iphone 15, samsung s24, macbook air"}
                },
                "required": ["product_name"]
            }
        }
    }
]


# ============================================================
# CHAT LOGIC (OPENROUTER & NATIVE GEMINI)
# ============================================================

def process_message(user_message: str) -> str:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    api_key = openrouter_key or gemini_key

    if not api_key:
        return "❌ Error: Neither OPENROUTER_API_KEY nor GEMINI_API_KEY environment variable is set."

    is_openrouter = bool(openrouter_key) or api_key.startswith("sk-or-v1")

    if is_openrouter:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant with access to tools for calculation and product lookup."},
                {"role": "user", "content": user_message}
            ]
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=messages,
                tools=openai_tools,
            )

            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    if fn_name in available_functions:
                        fn_result = available_functions[fn_name](**fn_args)
                    else:
                        fn_result = {"error": f"Tool '{fn_name}' not found."}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(fn_result)
                    })

                second_response = client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=messages
                )
                final_msg = second_response.choices[0].message
                return final_msg.content or "No response content."
            else:
                return msg.content or "No response content."

        except Exception as e:
            return f"❌ OpenRouter Error: {str(e)}"
    else:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_message,
                config=types.GenerateContentConfig(
                    tools=[add_numbers, product_info]
                )
            )
            return response.text or "No response content."
        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"


# ============================================================
# FLASK WEB SERVER ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please enter a valid message."})

    reply = process_message(user_msg)
    return jsonify({"reply": reply})


# ============================================================
# START WEB APP
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
