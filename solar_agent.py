#!/usr/bin/env python3
"""
N-Tech Solar Agent
------------------
A Claude-powered AI agent for N-Tech Energy Solutions LLC.
Researches solar topics, generates website content, and provides
marketing insights for the North Texas service area.

Usage:
    python solar_agent.py
    python solar_agent.py --mode research
    python solar_agent.py --mode content
    python solar_agent.py --mode marketing
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import anthropic

# ─── Knowledge Base ──────────────────────────────────────────────────────────

KB_PATH = Path(__file__).parent / "solar_knowledge_base.json"

KB_CATEGORIES = [
    "installation_technical",
    "equipment_specs",
    "utility_policy",
    "permits_and_codes",
    "market_and_pricing",
    "competitors",
    "incentives",
    "troubleshooting",
    "local_area",
    "general_solar",
]

def load_knowledge_base() -> dict:
    """Load the persistent knowledge base from disk."""
    if KB_PATH.exists():
        try:
            return json.loads(KB_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "_meta": {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "entry_count": 0,
        },
        **{cat: [] for cat in KB_CATEGORIES}
    }

def save_knowledge_base(kb: dict):
    """Persist the knowledge base to disk."""
    kb["_meta"]["last_updated"] = datetime.now().isoformat()
    kb["_meta"]["entry_count"] = sum(
        len(v) for k, v in kb.items() if k != "_meta"
    )
    KB_PATH.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")

def format_kb_for_prompt(kb: dict) -> str:
    """Format the knowledge base as a readable block for the system prompt."""
    lines = ["KNOWLEDGE BASE (what the agent has learned and saved):"]
    total = 0
    for cat in KB_CATEGORIES:
        entries = kb.get(cat, [])
        if entries:
            label = cat.replace("_", " ").title()
            lines.append(f"\n## {label}")
            for e in entries:
                lines.append(f"- [{e.get('date', '?')}] {e.get('fact', '')}")
                if e.get("source"):
                    lines.append(f"  Source: {e['source']}")
            total += len(entries)
    if total == 0:
        lines.append("(empty — use /learn to start building the knowledge base)")
    return "\n".join(lines)

def add_to_kb(kb: dict, category: str, fact: str, source: str = "") -> bool:
    """Add a fact to the knowledge base. Returns True if added."""
    if category not in KB_CATEGORIES:
        category = "general_solar"
    # Avoid duplicates (simple text match)
    existing = [e["fact"].lower() for e in kb.get(category, [])]
    if fact.lower() in existing:
        return False
    kb.setdefault(category, []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "fact": fact.strip(),
        "source": source.strip(),
    })
    return True

def search_kb(kb: dict, query: str) -> list:
    """Simple keyword search across all KB entries."""
    query_lower = query.lower()
    results = []
    for cat in KB_CATEGORIES:
        for entry in kb.get(cat, []):
            if query_lower in entry.get("fact", "").lower():
                results.append({"category": cat, **entry})
    return results


# ─── Business Context ────────────────────────────────────────────────────────

BUSINESS_CONTEXT = """
You are the dedicated AI agent for N-Tech Energy Solutions LLC — a solar installation
company based in Chico, TX (Wise County). You have deep expertise in solar energy,
the North Texas market, and the specific needs of this business.

COMPANY DETAILS:
- Name: N-Tech Energy Solutions LLC
- Location: Chico, TX (Wise County)
- Owner: [N-Tech owner — reference as "the team" or "N-Tech" in content]
- Website: (in development)
- Specialty: Residential and small commercial solar installations
- Starting price: $2.40/watt (competitive founding customer rate)
- Federal ITC: 30% tax credit available to all customers
- Founding Customer Offer: Special pricing for early adopters

SERVICE AREA (primary focus):
- Wise County, TX (HQ — Chico, Decatur, Bridgeport, Boyd, Rhome, Newark, Aurora)
- Parker County, TX (Weatherford, Aledo, Willow Park, Hudson Oaks, Springtown)
- Jack County, TX (Jacksboro, Bryson, Perrin)
- Montague County, TX (Montague, Bowie, Saint Jo, Nocona)
- Greater North Texas / DFW metro fringe

COMPETITIVE ADVANTAGES:
- Local company — not a national chain
- Transparent flat-rate pricing from $2.40/watt
- No-pressure consultations
- Experienced installers
- Handles all permits and utility interconnection paperwork
- Post-install support

KEY COMPETITORS IN NORTH TEXAS:
- Large nationals: Sunrun, Tesla/SolarCity, Sunnova, Freedom Forever, Momentum Solar
- Regional players: Texas Solar Outfitters, SolarTech, PetersenDean TX
- Co-ops / utilities: Oncor territory, CoServ (serves much of Wise County)

UTILITY / GRID FACTS:
- Most of Wise County is served by CoServ Electric (co-op, not Oncor)
- CoServ has net metering but co-op rules differ from Oncor TDUs
- Parker County is primarily Oncor territory
- ERCOT grid (deregulated Texas electricity market)

LOCAL MARKET INSIGHTS:
- North Texas gets ~230+ sunny days/year — excellent solar resource
- Average residential electric bill in the area: $150-$250/month
- Many properties are rural/semi-rural with good roof space and no shading
- Agricultural land — some interest in ground-mount systems
- Strong conservative/independence values → energy independence messaging resonates
- Many homeowners wary of national solar companies after bad experiences

SOLAR EDUCATION TALKING POINTS:
- 30% Federal ITC (Investment Tax Credit) through 2032
- Average payback period: 8-12 years (North Texas)
- System lifespan: 25-30 years
- Panels typically warrant 25 years / 80% output guarantee
- Battery storage (e.g., Enphase IQ Battery, Tesla Powerwall) adds resilience
- Net metering / bill credits for excess generation

CONTENT STRATEGY:
- Target long-tail local SEO keywords (e.g., "solar panels Decatur TX", "solar installation Wise County")
- Blog topics: savings calculations, local utility info, installation process, FAQs
- City pages: one page per major city in service area with localized content
- Tone: Friendly, knowledgeable, no hard sell — educational first

Today's date: {date}

{knowledge_base}
""".strip()

# ─── Mode Prompts ────────────────────────────────────────────────────────────

MODE_PROMPTS = {
    "research": """
You are in RESEARCH MODE. Use web search to find current, accurate information about:
- Solar industry trends, pricing, and technology
- Local utility policies (CoServ, Oncor, ERCOT)
- Competitor activities in North Texas
- Incentives, rebates, and tax credits
- Solar adoption rates in Texas / North Texas counties
- Relevant news that could affect the solar market

Always cite your sources. Summarize findings clearly and highlight what's most
actionable for N-Tech Energy Solutions.

When you find important facts worth remembering, end your response with a section:
SAVE TO KNOWLEDGE BASE:
- category: <one of: installation_technical, equipment_specs, utility_policy, permits_and_codes, market_and_pricing, competitors, incentives, troubleshooting, local_area, general_solar>
- fact: <the specific fact to save>
- source: <URL or source name>
(Repeat for each fact worth saving. The agent will parse and store these automatically.)
""",

    "content": """
You are in CONTENT GENERATION MODE. Create high-quality, SEO-optimized content for
the N-Tech Energy Solutions website. This includes:
- Blog posts (800-1500 words, educational and engaging)
- City/location pages (900-1200 words, locally targeted)
- FAQs
- Meta titles and descriptions
- Social media posts

Content guidelines:
- Target local keywords naturally (city + "solar panels", "solar installation", etc.)
- Include LocalBusiness schema suggestions where relevant
- Mention specific local landmarks, utilities, or characteristics when appropriate
- Include calls to action toward the free consultation / get a quote form
- Tone: Friendly, expert, non-pushy
- Always include: federal 30% tax credit, local utility info, N-Tech's pricing advantage

When generating HTML content (blog posts, city pages), format it to match the
existing N-Tech website style with proper heading hierarchy (h1, h2, h3).

Draw on the knowledge base for accurate local facts when writing content.
""",

    "marketing": """
You are in MARKETING INSIGHTS MODE. Analyze and advise on:
- Local marketing opportunities in Wise County, Parker County, Jack County, Montague County
- Digital marketing strategy (Google Ads, Facebook/Meta, Next Door)
- SEO strategy and keyword opportunities
- Competitor analysis and positioning
- Seasonal campaign timing (spring/summer install season)
- Customer messaging and value propositions
- Referral program ideas
- Community engagement (local events, sponsorships)

Use web search to find current competitor campaigns, local market data, and
marketing benchmarks for solar companies in Texas.

When you find important competitor or market facts, end your response with:
SAVE TO KNOWLEDGE BASE:
- category: competitors (or market_and_pricing)
- fact: <the fact>
- source: <source>
""",

    "technical": """
You are in TECHNICAL MODE. You are an expert solar installation technician and engineer.
Answer detailed technical questions about:
- System design (string sizing, array layout, shading analysis, azimuth/tilt optimization)
- Electrical (NEC Article 690, wire sizing, overcurrent protection, grounding/bonding)
- Inverters (string inverters, microinverters, power optimizers — Enphase, SMA, SolarEdge, Fronius)
- Panels (monocrystalline, polycrystalline, TOPCon, HJT — efficiency, temperature coefficients)
- Battery storage (Enphase IQ Battery, Tesla Powerwall, Franklin WH, installation requirements)
- Racking and mounting (roof penetrations, flashing, rail systems, ground mounts)
- Utility interconnection (CoServ application process, Oncor process, ERCOT rules)
- Permitting (AHJ requirements for Wise County, Parker County, Jack County, Montague County)
- Installation best practices, safety, and common issues
- Commissioning, testing, and troubleshooting

Use web search to look up current specs, codes, or utility requirements when needed.

When you learn something worth retaining, end your response with:
SAVE TO KNOWLEDGE BASE:
- category: <installation_technical, equipment_specs, utility_policy, or permits_and_codes>
- fact: <the specific technical fact>
- source: <source>
""",

    "chat": """
You are in GENERAL ASSISTANT MODE. Answer any questions about:
- Solar technology and how it works
- N-Tech's service area, pricing, and process
- Industry trends and news
- Website content suggestions
- Business strategy and growth ideas

Use web search when you need current data or specific facts.
Draw on the knowledge base for facts already learned.
"""
}

# ─── Agent Core ──────────────────────────────────────────────────────────────

class SolarAgent:
    def __init__(self, mode: str = "chat", save_output: bool = False):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
            print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=api_key)
        self.mode = mode
        self.save_output = save_output
        self.conversation_history = []
        self.output_dir = Path(__file__).parent / "agent_output"
        self.kb = load_knowledge_base()

        if save_output:
            self.output_dir.mkdir(exist_ok=True)

        self._rebuild_system_prompt()

    def _rebuild_system_prompt(self):
        self.system_prompt = (
            BUSINESS_CONTEXT.format(
                date=datetime.now().strftime("%B %d, %Y"),
                knowledge_base=format_kb_for_prompt(self.kb),
            )
            + "\n\n"
            + MODE_PROMPTS.get(self.mode, MODE_PROMPTS["chat"])
        )

    def _parse_and_save_kb_entries(self, response_text: str):
        """Parse any SAVE TO KNOWLEDGE BASE sections from the agent's response."""
        if "SAVE TO KNOWLEDGE BASE:" not in response_text:
            return 0

        saved = 0
        section = response_text.split("SAVE TO KNOWLEDGE BASE:")[-1]
        lines = section.strip().splitlines()

        current = {}
        for line in lines:
            line = line.strip().lstrip("-").strip()
            if line.startswith("category:"):
                current["category"] = line.split(":", 1)[1].strip()
            elif line.startswith("fact:"):
                current["fact"] = line.split(":", 1)[1].strip()
            elif line.startswith("source:"):
                current["source"] = line.split(":", 1)[1].strip()
                # Complete entry — save it
                if current.get("fact"):
                    added = add_to_kb(
                        self.kb,
                        current.get("category", "general_solar"),
                        current["fact"],
                        current.get("source", ""),
                    )
                    if added:
                        saved += 1
                current = {}

        # Catch last entry if no trailing source line
        if current.get("fact"):
            added = add_to_kb(
                self.kb,
                current.get("category", "general_solar"),
                current["fact"],
                current.get("source", ""),
            )
            if added:
                saved += 1

        if saved > 0:
            save_knowledge_base(self.kb)
            self._rebuild_system_prompt()  # Refresh prompt with new KB
            print(f"\n[Knowledge base updated: {saved} new fact(s) saved]")

        return saved

    def _stream_response(self, user_message: str) -> str:
        """Stream response with web search tool handling."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        tools = [{"type": "web_search_20260209", "name": "web_search"}]
        full_response_text = ""
        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            assistant_content = []
            current_text = ""

            with self.client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=8096,
                system=self.system_prompt,
                tools=tools,
                messages=self.conversation_history,
            ) as stream:
                for event in stream:
                    event_type = type(event).__name__

                    if event_type == "ContentBlockStart":
                        block = event.content_block
                        if hasattr(block, "type") and block.type == "tool_use":
                            if block.name == "web_search":
                                query = block.input.get("query", "") if hasattr(block, "input") else ""
                                print(f"\n[Searching: {query or '...'}]", flush=True)

                    elif event_type == "ContentBlockDelta":
                        delta = event.delta
                        if hasattr(delta, "type") and delta.type == "text_delta":
                            print(delta.text, end="", flush=True)
                            current_text += delta.text

                final_msg = stream.get_final_message()
                assistant_content = final_msg.content
                stop_reason = final_msg.stop_reason

            full_response_text += current_text

            if stop_reason == "end_turn":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                break

            if stop_reason == "tool_use":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                tool_results = []
                for block in assistant_content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        if block.name == "web_search":
                            if not any(r["tool_use_id"] == block.id for r in tool_results):
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": ""
                                })
                if tool_results:
                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_results
                    })
                else:
                    break
            else:
                break

        # Auto-parse KB save instructions from the response
        self._parse_and_save_kb_entries(full_response_text)

        return full_response_text

    def save_to_file(self, content: str, prefix: str = "output"):
        """Save generated content to a timestamped file."""
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{prefix}_{timestamp}.md"
        filename.write_text(content, encoding="utf-8")
        print(f"\n[Saved to: {filename}]")
        return filename

    def run(self):
        """Main interactive loop."""
        mode_labels = {
            "research": "Research Mode",
            "content": "Content Generation Mode",
            "marketing": "Marketing Insights Mode",
            "technical": "Technical Mode",
            "chat": "General Assistant Mode",
        }
        label = mode_labels.get(self.mode, "Assistant Mode")
        kb_count = self.kb["_meta"].get("entry_count", 0)

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║          N-Tech Energy Solutions — Solar AI Agent           ║
║                      {label:<38}║
╚══════════════════════════════════════════════════════════════╝
Knowledge base: {kb_count} facts stored

Commands:
  /mode research    — Market research with web search
  /mode content     — Blog posts, city pages, FAQs
  /mode marketing   — Competitor analysis, ad strategy
  /mode technical   — Installation, wiring, equipment, codes
  /mode chat        — General assistant
  /learn <topic>    — Research a topic and save facts to KB
  /recall <topic>   — Search the knowledge base
  /kb               — Show full knowledge base summary
  /save             — Save last response to file
  /clear            — Clear conversation history
  /quit             — Exit
""")

        last_response = ""

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break

                elif cmd == "/clear":
                    self.conversation_history = []
                    print("[Conversation history cleared]")

                elif cmd == "/save":
                    if last_response:
                        self.save_to_file(last_response, prefix=self.mode)
                    else:
                        print("[No response to save yet]")

                elif cmd == "/mode":
                    if arg in MODE_PROMPTS:
                        self.mode = arg
                        self._rebuild_system_prompt()
                        self.conversation_history = []
                        print(f"[Switched to {mode_labels.get(arg, arg)} — history cleared]")
                    elif arg:
                        print(f"[Unknown mode: {arg}. Options: {', '.join(MODE_PROMPTS.keys())}]")
                    else:
                        print(f"[Current mode: {self.mode}]")

                elif cmd == "/learn":
                    if arg:
                        print(f"\n[Researching and learning: {arg}]\n")
                        print("Agent: ", end="", flush=True)
                        old_mode = self.mode
                        self.mode = "research"
                        self._rebuild_system_prompt()
                        try:
                            last_response = self._stream_response(
                                f"Research this topic thoroughly and save the most important facts "
                                f"to the knowledge base: {arg}"
                            )
                        except anthropic.RateLimitError:
                            print("\n[Rate limit hit — wait 60 seconds and try again]")
                        except anthropic.APIError as e:
                            print(f"\n[API Error: {e}]")
                        self.mode = old_mode
                        self._rebuild_system_prompt()
                        print()
                    else:
                        print("[Usage: /learn <topic>]")

                elif cmd == "/recall":
                    if arg:
                        results = search_kb(self.kb, arg)
                        if results:
                            print(f"\n[Found {len(results)} fact(s) matching '{arg}':]")
                            for r in results:
                                cat = r["category"].replace("_", " ").title()
                                print(f"  [{r['date']}] ({cat}) {r['fact']}")
                                if r.get("source"):
                                    print(f"    Source: {r['source']}")
                        else:
                            print(f"[No facts found matching '{arg}' in knowledge base]")
                    else:
                        print("[Usage: /recall <keyword>]")

                elif cmd == "/kb":
                    print("\n" + format_kb_for_prompt(self.kb))
                    total = self.kb["_meta"].get("entry_count", 0)
                    updated = self.kb["_meta"].get("last_updated", "never")
                    print(f"\nTotal: {total} facts | Last updated: {updated}")

                else:
                    print(f"[Unknown command: {cmd}]")

                continue

            # Normal chat
            print("\nAgent: ", end="", flush=True)
            try:
                last_response = self._stream_response(user_input)
                print()
            except anthropic.RateLimitError:
                print("\n[Rate limit hit — wait 60 seconds and try again]")
            except anthropic.APIError as e:
                print(f"\n[API Error: {e}]")
            except Exception as e:
                print(f"\n[Error: {e}]")
                raise


# ─── Quick-Run Helpers ───────────────────────────────────────────────────────

def quick_research(topic: str) -> str:
    agent = SolarAgent(mode="research")
    return agent._stream_response(topic)


def generate_blog_post(topic: str, city: str = None) -> str:
    agent = SolarAgent(mode="content")
    prompt = f"Write a complete, SEO-optimized blog post about: {topic}"
    if city:
        prompt += f"\nFocus on how this applies to homeowners in {city}, TX."
    return agent._stream_response(prompt)


def generate_city_page(city: str, county: str) -> str:
    agent = SolarAgent(mode="content")
    prompt = (
        f"Generate a complete HTML city page for solar installations in {city}, {county}, TX. "
        f"Follow the same structure as other N-Tech city pages. "
        f"Include localized content about the area, CoServ or Oncor utility info if applicable, "
        f"solar savings estimates, and a strong call to action."
    )
    return agent._stream_response(prompt)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N-Tech Solar Agent — Claude-powered solar research & content tool"
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_PROMPTS.keys()),
        default="chat",
        help="Starting mode (default: chat)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Auto-save all responses to agent_output/ directory"
    )
    parser.add_argument(
        "--research",
        metavar="TOPIC",
        help="Run a one-shot research query and exit"
    )
    parser.add_argument(
        "--blog",
        metavar="TOPIC",
        help="Generate a blog post on the given topic and exit"
    )
    parser.add_argument(
        "--city-page",
        metavar="CITY,COUNTY",
        help="Generate a city landing page (e.g. 'Decatur,Wise County') and exit"
    )
    parser.add_argument(
        "--learn",
        metavar="TOPIC",
        help="Research a topic and save facts to the knowledge base, then exit"
    )

    args = parser.parse_args()

    if args.learn:
        print(f"Learning: {args.learn}\n")
        agent = SolarAgent(mode="research")
        agent._stream_response(
            f"Research this topic thoroughly and save the most important facts "
            f"to the knowledge base: {args.learn}"
        )
        print()
        return

    if args.research:
        print(f"Researching: {args.research}\n")
        result = quick_research(args.research)
        if args.save:
            agent = SolarAgent(mode="research", save_output=True)
            agent.save_to_file(result, prefix="research")
        return

    if args.blog:
        print(f"Generating blog post: {args.blog}\n")
        result = generate_blog_post(args.blog)
        if args.save:
            agent = SolarAgent(mode="content", save_output=True)
            agent.save_to_file(result, prefix="blog")
        return

    if args.city_page:
        parts = args.city_page.split(",", 1)
        city = parts[0].strip()
        county = parts[1].strip() if len(parts) > 1 else "North Texas"
        print(f"Generating city page for: {city}, {county}\n")
        result = generate_city_page(city, county)
        if args.save:
            agent = SolarAgent(mode="content", save_output=True)
            agent.save_to_file(result, prefix=f"city_{city.lower().replace(' ', '_')}")
        return

    agent = SolarAgent(mode=args.mode, save_output=args.save)
    agent.run()


if __name__ == "__main__":
    main()
