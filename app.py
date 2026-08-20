import os
from google import genai
from google.genai import types

# ============================================================
# 1. GEMINI API KEY SETUP
# ============================================================
# Best practice: Load API key from environment variable or set manually
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Set your API key before running the script:")
    print("  PowerShell: $env:GEMINI_API_KEY='your_api_key'")
    print("  CMD:        set GEMINI_API_KEY=your_api_key")
    print("  Bash:       export GEMINI_API_KEY='your_api_key'")
    exit(1)

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. TOOL 1 — ADD NUMBERS
# ============================================================

def add_numbers(a: float, b: float) -> dict:
    """Adds two numbers together and returns the calculation breakdown."""
    print("\n" + "=" * 60)
    print("🔧 TOOL CALLED: add_numbers")
    print("📥 Arguments received:")
    print(f"   a = {a}")
    print(f"   b = {b}")

    result = a + b

    print(f"⚙️ Calculating: {a} + {b}")
    print(f"📤 Tool result: {result}")
    print("=" * 60)

    return {
        "operation": "addition",
        "a": a,
        "b": b,
        "result": result
    }


# ============================================================
# 3. TOOL 2 — PRODUCT INFORMATION
# ============================================================

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


# ============================================================
# 4. REGISTER TOOLS
# ============================================================

tools = [
    add_numbers,
    product_info
]


# ============================================================
# 5. CREATE GEMINI CHAT
# ============================================================

# Standard model identifiers: 'gemini-2.5-flash' or 'gemini-2.0-flash'
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        tools=tools
    )
)


# ============================================================
# 6. RUN AGENT
# ============================================================

def run_agent():

    print("\n" + "=" * 60)
    print("🤖 GEMINI TOOL-CALLING AGENT")
    print("=" * 60)

    print("\nAvailable tools:")
    print("🔧 add_numbers")
    print("🔧 product_info")

    print("\nType 'exit' to stop.")
    print("=" * 60)

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
# 7. START PROGRAM
# ============================================================

if __name__ == "__main__":
    run_agent()
