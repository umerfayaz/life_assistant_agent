from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from ai_sidekick.sidekick import Sidekick

app = Flask(__name__)
CORS(app)

sidekick_instance = None

async def setup_sidekick():
    global sidekick_instance
    sidekick_instance = Sidekick()
    await sidekick_instance.setup()
    return sidekick_instance

async def process_message_async(message, history):
    results = await sidekick_instance.run_superstep(
        message,
        "Provide a helpful response",
        history
    )
    return results

@app.route("/api/setup", methods=["POST"])
def api_setup():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(setup_sidekick())
    return jsonify({"status": "success", "message": "Sidekick initialized"})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    message = data.get("message", "")
    history = data.get("history", [])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(process_message_async(message, history))

    # 🧹 Clean evaluator feedback if it exists
    cleaned_results = []
    for msg in results:
        text = msg.get("content", "")
        if text.startswith("Evaluator Feedback"):
            text = text.replace("Evaluator Feedback:", "").strip()
        cleaned_results.append({
            **msg,
            "content": text
        })

    return jsonify({"messages": cleaned_results})



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=7860)
