import os
import json
from openai import OpenAI
from google import genai
from google.genai import types

# ============================================================
# 1. API KEY & PROVIDER DETECTION
# ============================================================
# Load from .env file if present
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# Allow either OPENROUTER_API_KEY or GEMINI_API_KEY
API_KEY = OPENROUTER_KEY or GEMINI_KEY

if not API_KEY:
    print("❌ ERROR: Neither OPENROUTER_API_KEY nor GEMINI_API_KEY is set.")
    print("Set your OpenRouter key before running:")
    print("  PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...'")
    print("  CMD:        set OPENROUTER_API_KEY=sk-or-v1-...")
    print("  Bash:       export OPENROUTER_API_KEY='sk-or-v1-...'")
    print("Or set GEMINI_API_KEY if using an OpenRouter key there:")
    print("  PowerShell: $env:GEMINI_API_KEY='sk-or-v1-...'")
    exit(1)

# Auto-detect OpenRouter mode (OpenRouter keys start with 'sk-or-v1')
IS_OPENROUTER = bool(OPENROUTER_KEY) or API_KEY.startswith("sk-or-v1")


# ============================================================
# 2. TOOL DEFINITIONS (PYTHON FUNCTIONS)
# ============================================================

def add_numbers(a: float, b: float) -> dict:
    """Adds two numbers together and returns the calculation breakdown."""
    print("\n" + "=" * 60)
    print("🔧 TOOL CALLED: add_numbers")
    print("📥 Arguments received:")
    print(f"   a = {a}")
    print(f"   b = {b}")

    result = float(a) + float(b)

    print(f"⚙️ Calculating: {a} + {b}")
    print(f"📤 Tool result: {result}")
    print("=" * 60)

    return {
        "operation": "addition",
        "a": a,
        "b": b,
        "result": result
    }


def product_info(product_name: str) -> dict:
    """Retrieves detailed product pricing and category information by product name."""
    print("\n" + "=" * 60)
    print("🔧 TOOL CALLED: product_info")
    print(f"📥 Product requested: {product_name}")

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
        print("✅ Product found!")
        print(f"📦 Product: {product['name']}")
        print(f"💰 Price: ₹{product['price']}")
        print(f"🏷️ Category: {product['category']}")
        print("=" * 60)
        return product

    print("❌ Product not found.")
    print("=" * 60)
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
# 3. OPENROUTER AGENT
# ============================================================

def run_openrouter_agent():
    print("\n" + "=" * 60)
    print("🤖 TOOL-CALLING AGENT (Mode: OpenRouter API)")
    print("=" * 60)
    print("\nAvailable tools:")
    print("🔧 add_numbers")
    print("🔧 product_info")
    print("\nType 'exit' to stop.")
    print("=" * 60)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools for calculation and product lookup."}
    ]

    while True:
        try:
            user_input = input("\n👤 You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user_input.strip().lower() == "exit":
            print("\n👋 Goodbye!")
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            print("\n🤖 Sending request to OpenRouter...")
            print("🧠 Model is deciding which tool to use...")

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

                # Request final response with tool outputs
                second_response = client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=messages
                )
                final_msg = second_response.choices[0].message
                messages.append(final_msg)

                print("\n🤖 Assistant:")
                print(final_msg.content)
            else:
                print("\n🤖 Assistant:")
                print(msg.content)

        except Exception as e:
            print("\n❌ ERROR:")
            print(e)


# ============================================================
# 4. NATIVE NATIVE GEMINI AGENT
# ============================================================

def run_gemini_agent():
    print("\n" + "=" * 60)
    print("🤖 TOOL-CALLING AGENT (Mode: Native Gemini API)")
    print("=" * 60)
    print("\nAvailable tools:")
    print("🔧 add_numbers")
    print("🔧 product_info")
    print("\nType 'exit' to stop.")
    print("=" * 60)

    client = genai.Client(api_key=API_KEY)
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=[add_numbers, product_info]
        )
    )

    while True:
        try:
            user_input = input("\n👤 You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user_input.strip().lower() == "exit":
            print("\n👋 Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            print("\n🤖 Sending request to Gemini...")
            print("🧠 Gemini is deciding which tool to use...")

            response = chat.send_message(user_input)

            print("\n🤖 Gemini:")
            print(response.text)

        except Exception as e:
            print("\n❌ ERROR:")
            print(e)


# ============================================================
# 5. START PROGRAM
# ============================================================

if __name__ == "__main__":
    if IS_OPENROUTER:
        run_openrouter_agent()
    else:
        run_gemini_agent()
