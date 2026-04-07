import os
import json
import base64
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import anthropic

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─── Web Search ──────────────────────────────────────────────────────────────

SEARCH_TRIGGERS = [
    "recent", "news", "latest", "current", "today", "this year",
    "2025", "2026", "tariff", "policy", "regulation", "incentive",
    "price", "cost", "market", "update", "change", "new",
]

def web_search(query, max_results=5):
    if not DDG_AVAILABLE:
        return ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        lines = [f"LIVE WEB SEARCH RESULTS (searched: {query})\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}")
            lines.append(f"   {r.get('body', '')}")
            lines.append(f"   Source: {r.get('href', '')}\n")
        lines.append("Use these results to answer questions about current or recent information. Cite sources when relevant.\n")
        return "\n".join(lines)
    except Exception:
        return ""

# ─── GitHub Integration (READ ONLY) ──────────────────────────────────────────

GITHUB_REPO = os.environ.get("GITHUB_REPO", "n-tech-es/n-tech-website")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

def github_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def github_list_pages():
    headers = github_headers()
    if not headers:
        return []
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/"
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    if r.status_code != 200:
        return []
    files = [f["name"] for f in r.json() if f["name"].endswith(".html")]
    return sorted(files)

def github_read_file(filename):
    headers = github_headers()
    if not headers:
        return None
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    if r.status_code != 200:
        return None
    data = r.json()
    return base64.b64decode(data["content"]).decode("utf-8")

# ─── Business Context ─────────────────────────────────────────────────────────

KB_PATH = Path(__file__).resolve().parent.parent / "solar_knowledge_base.json"

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
- North Texas gets 229 sunny days/year (Fort Worth NOAA)
- Average residential electric bill: $150-$250/month
- Average payback period: 7-10 years
- 30% Federal ITC through 2032

WEBSITE ACCESS:
You can read website pages when loaded. When asked about changes to a page:
- Describe exactly what needs to change in plain English
- Specify the section, the current text, and what it should say instead
- Do NOT output HTML code — the user will pass your instructions to their developer to implement

Today's date: {date}

{knowledge_base}

{pages_list}
""".strip()

MODE_PROMPTS = {
    "chat": "You are a helpful solar energy assistant for N-Tech Energy Solutions. Answer questions about solar, the install process, savings, and N-Tech's services. Be friendly, knowledgeable, and concise.",
    "technical": "You are an expert solar installation technician. Answer detailed technical questions about system design, NEC codes, wiring, inverters, panels, battery storage, racking, utility interconnection, and permitting for North Texas.",
    "content": "You are a content writer for N-Tech Energy Solutions. Generate SEO-optimized blog posts, city pages, FAQs, and social media content. Target North Texas homeowners. Tone: friendly, expert, non-pushy.",
    "marketing": "You are a marketing strategist for N-Tech Energy Solutions. Analyze competitors, suggest Google Ads strategies, local marketing opportunities, and customer messaging for the North Texas solar market.",
    "research": "You are a solar industry researcher for N-Tech Energy Solutions. Provide detailed analysis of market trends, utility policies, incentives, and competitor activities in North Texas.",
    "website": """You are a website reviewer for N-Tech Energy Solutions. You can read pages and identify what needs to change.

IMPORTANT RULES:
- You read pages and describe changes in plain English only
- Never output HTML code under any circumstances
- When you spot something to fix, describe it clearly:
  * Which section it is in
  * What the current text says
  * What it should say instead
  * Why the change improves the page
- The user will copy your instructions to their developer who will make the actual changes""",
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

def build_system_prompt(mode="chat", page_content=None, page_name=None):
    kb = load_kb()
    pages = github_list_pages()
    pages_list = "WEBSITE PAGES:\n" + "\n".join(f"  - {p}" for p in pages) if pages else ""

    base = BUSINESS_CONTEXT.format(
        date=datetime.now().strftime("%B %d, %Y"),
        knowledge_base=kb,
        pages_list=pages_list,
    )

    system = base + "\n\n" + MODE_PROMPTS.get(mode, MODE_PROMPTS["chat"])

    if page_content and page_name:
        system += f"\n\nCURRENT PAGE LOADED: {page_name}\n\n{page_content}"

    return system

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/pages")
def list_pages():
    pages = github_list_pages()
    return jsonify(pages)

@app.route("/api/file")
def read_file():
    filename = request.args.get("name")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    content = github_read_file(filename)
    if content is None:
        return jsonify({"error": "File not found"}), 404
    return jsonify({"name": filename, "content": content})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    mode = data.get("mode", "chat")
    page_content = data.get("page_content")
    page_name = data.get("page_name")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not configured"}), 500

    client = anthropic.Anthropic(api_key=api_key)
    system = build_system_prompt(mode, page_content, page_name)

    # Web search: always in research mode, or when message contains current-info triggers
    last_msg = messages[-1]["content"].lower() if messages else ""
    should_search = mode == "research" or any(t in last_msg for t in SEARCH_TRIGGERS)
    if should_search and messages:
        query = messages[-1]["content"]
        search_results = web_search(query)
        if search_results:
            system += f"\n\n{search_results}"

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
