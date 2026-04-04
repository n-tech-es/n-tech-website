import os
import json
import base64
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import anthropic

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ─── GitHub Integration ───────────────────────────────────────────────────────

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
    """List all HTML files in the repo root."""
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
    """Read a file from the GitHub repo."""
    headers = github_headers()
    if not headers:
        return None, None
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    if r.status_code != 200:
        return None, None
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    sha = data["sha"]
    return content, sha

def github_write_file(filename, content, commit_message):
    """Write a file to the GitHub repo."""
    headers = github_headers()
    if not headers:
        return False, "GitHub token not configured"

    # Get current SHA if file exists
    _, sha = github_read_file(filename)

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
    payload = {
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200, 201):
        return True, None
    return False, r.json().get("message", "Unknown error")

# ─── Load business context and KB ────────────────────────────────────────────

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

WEBSITE FILE ACCESS:
You can read and edit website files. When asked to update a page:
1. Tell the user what changes you will make
2. Output the complete updated HTML wrapped in a code block: ```html ... ```
3. The user can then click "Save to Website" to publish it

Today's date: {date}

{knowledge_base}

{pages_list}
""".strip()

MODE_PROMPTS = {
    "chat": "You are a helpful solar energy assistant for N-Tech Energy Solutions. Answer questions about solar, the install process, savings, and N-Tech's services. Be friendly, knowledgeable, and concise.",
    "technical": "You are an expert solar installation technician. Answer detailed technical questions about system design, NEC codes, wiring, inverters, panels, battery storage, racking, utility interconnection, and permitting for North Texas.",
    "content": "You are a content writer for N-Tech Energy Solutions. Generate SEO-optimized blog posts, city pages, FAQs, and social media content. Target North Texas homeowners. Tone: friendly, expert, non-pushy. When generating full HTML pages, wrap them in ```html ... ``` code blocks.",
    "marketing": "You are a marketing strategist for N-Tech Energy Solutions. Analyze competitors, suggest Google Ads strategies, local marketing opportunities, and customer messaging for the North Texas solar market.",
    "research": "You are a solar industry researcher for N-Tech Energy Solutions. Provide detailed analysis of market trends, utility policies, incentives, and competitor activities in North Texas.",
    "website": "You are the website editor for N-Tech Energy Solutions. You can read and update website pages. When asked to edit a page, output the complete updated HTML in a ```html ... ``` code block so the user can save it directly to the website. Be precise and preserve all existing styles, scripts, and structure unless specifically asked to change them.",
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
        system += f"\n\nCURRENT PAGE LOADED: {page_name}\n```html\n{page_content}\n```"

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
    content, sha = github_read_file(filename)
    if content is None:
        return jsonify({"error": "File not found"}), 404
    return jsonify({"name": filename, "content": content, "sha": sha})

@app.route("/api/file", methods=["POST"])
def write_file():
    data = request.json
    filename = data.get("name")
    content = data.get("content")
    message = data.get("message", f"Update {filename} via N-Tech Solar Agent")

    if not filename or not content:
        return jsonify({"error": "name and content required"}), 400

    ok, error = github_write_file(filename, content, message)
    if ok:
        return jsonify({"success": True})
    return jsonify({"error": error}), 500

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

    def generate():
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=8096,
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
