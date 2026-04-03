import os
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import anthropic

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─── Load business context and KB ────────────────────────────────────────────

KB_PATH = Path(__file__).parent.parent / "solar_knowledge_base.json"

BUSINESS_CONTEXT = """
You are the dedicated AI agent for N-Tech Energy Solutions LLC — a solar installation
company based in Chico, TX (Wise County). You have deep expertise in solar energy,
the North Texas market, and the specific needs of this business.

COMPANY DETAILS:
- Name: N-Tech Energy Solutions LLC
- Location: Chico, TX (Wise County)
- Specialty: Residential and small commercial solar installations
- Starting price: $2.40/watt (competitive founding customer rate)
- Federal ITC: 30% tax credit available to all customers

SERVICE AREA:
- Wise County, TX (Chico, Decatur, Bridgeport, Boyd, Rhome, Newark, Aurora)
- Parker County, TX (Weatherford, Aledo, Willow Park, Hudson Oaks, Springtown)
- Jack County, TX (Jacksboro, Bryson, Perrin)
- Montague County, TX (Montague, Bowie, Saint Jo, Nocona)

UTILITY FACTS:
- Wise County: mostly CoServ Electric (co-op)
- Parker County: primarily Oncor territory
- ERCOT grid (deregulated Texas electricity market)

LOCAL MARKET:
- North Texas gets ~230+ sunny days/year
- Average residential electric bill: $150-$250/month
- Average payback period: 8-12 years
- 30% Federal ITC through 2032

Today's date: {date}

{knowledge_base}
""".strip()

MODE_PROMPTS = {
    "chat": "You are a helpful solar energy assistant for N-Tech Energy Solutions. Answer questions about solar, the install process, savings, and N-Tech's services. Be friendly, knowledgeable, and concise.",
    "technical": "You are an expert solar installation technician. Answer detailed technical questions about system design, NEC codes, wiring, inverters, panels, battery storage, racking, utility interconnection, and permitting for North Texas.",
    "content": "You are a content writer for N-Tech Energy Solutions. Generate SEO-optimized blog posts, city pages, FAQs, and social media content. Target North Texas homeowners. Tone: friendly, expert, non-pushy.",
    "marketing": "You are a marketing strategist for N-Tech Energy Solutions. Analyze competitors, suggest Google Ads strategies, local marketing opportunities, and customer messaging for the North Texas solar market.",
    "research": "You are a solar industry researcher for N-Tech Energy Solutions. Provide detailed analysis of market trends, utility policies, incentives, and competitor activities in North Texas.",
}

def load_kb():
    if KB_PATH.exists():
        try:
            kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
            lines = ["KNOWLEDGE BASE:"]
            for key, entries in kb.items():
                if key == "_meta" or not entries:
                    continue
                label = key.replace("_", " ").title()
                lines.append(f"\n## {label}")
                for e in entries:
                    lines.append(f"- {e.get('fact', '')}")
            return "\n".join(lines)
        except Exception:
            pass
    return "(no knowledge base yet)"

def build_system_prompt(mode="chat"):
    kb = load_kb()
    base = BUSINESS_CONTEXT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        knowledge_base=kb,
    )
    return base + "\n\n" + MODE_PROMPTS.get(mode, MODE_PROMPTS["chat"])

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    mode = data.get("mode", "chat")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not configured"}), 500

    client = anthropic.Anthropic(api_key=api_key)
    system = build_system_prompt(mode)

    def generate():
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.route("/modes")
def modes():
    return jsonify(list(MODE_PROMPTS.keys()))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
