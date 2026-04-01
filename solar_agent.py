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
""",

    "chat": """
You are in GENERAL ASSISTANT MODE. Answer any questions about:
- Solar technology and how it works
- N-Tech's service area, pricing, and process
- Industry trends and news
- Website content suggestions
- Business strategy and growth ideas

Use web search when you need current data or specific facts.
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

        if save_output:
            self.output_dir.mkdir(exist_ok=True)

        # Build system prompt
        self.system_prompt = (
            BUSINESS_CONTEXT.format(date=datetime.now().strftime("%B %d, %Y"))
            + "\n\n"
            + MODE_PROMPTS.get(mode, MODE_PROMPTS["chat"])
        )

    def _run_agent_loop(self, user_message: str) -> str:
        """Run the agentic loop, handling web search tool calls automatically."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        tools = [{"type": "web_search_20260209", "name": "web_search"}]
        full_response_text = ""

        while True:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8096,
                thinking={"type": "adaptive"},
                system=self.system_prompt,
                tools=tools,
                messages=self.conversation_history,
            )

            # Collect all content blocks for the assistant turn
            assistant_content = response.content

            # Extract text from non-tool-use blocks
            for block in assistant_content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        full_response_text += block.text
                    elif block.type == "thinking":
                        # Skip thinking blocks in display (they're internal)
                        pass

            # If Claude is done (no tool use), we're finished
            if response.stop_reason == "end_turn":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                break

            # Handle tool use (web search)
            if response.stop_reason == "tool_use":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })

                # Process each tool call
                tool_results = []
                for block in assistant_content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        if block.name == "web_search":
                            query = block.input.get("query", "")
                            print(f"\n[Searching: {query}]", flush=True)
                            # The web_search tool result is handled server-side;
                            # we pass back a tool_result with the block id
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": ""  # Server-side tool — result is injected by API
                            })

                if tool_results:
                    self.conversation_history.append({
                        "role": "user",
                        "content": tool_results
                    })
                else:
                    # No tool results to process — break to avoid infinite loop
                    break
            else:
                # Unknown stop reason — exit loop
                break

        return full_response_text

    def _stream_response(self, user_message: str) -> str:
        """Stream response with web search tool handling."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        tools = [{"type": "web_search_20260209", "name": "web_search"}]
        full_response_text = ""
        iteration = 0
        max_iterations = 10  # Safety limit

        while iteration < max_iterations:
            iteration += 1
            assistant_content = []
            current_text = ""

            with self.client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=8096,
                thinking={"type": "adaptive"},
                system=self.system_prompt,
                tools=tools,
                messages=self.conversation_history,
            ) as stream:
                for event in stream:
                    event_type = type(event).__name__

                    if event_type == "ContentBlockStart":
                        block = event.content_block
                        if hasattr(block, "type"):
                            if block.type == "tool_use":
                                if block.name == "web_search":
                                    query = block.input.get("query", "") if hasattr(block, "input") else ""
                                    print(f"\n[Searching: {query or '...'}]", flush=True)

                    elif event_type == "ContentBlockDelta":
                        delta = event.delta
                        if hasattr(delta, "type"):
                            if delta.type == "text_delta":
                                print(delta.text, end="", flush=True)
                                current_text += delta.text

                # Get the final message after stream completes
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
                            query = block.input.get("query", "") if hasattr(block, "input") else ""
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

        return full_response_text

    def save_to_file(self, content: str, prefix: str = "output"):
        """Save generated content to a timestamped file."""
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
            "chat": "General Assistant Mode",
        }
        label = mode_labels.get(self.mode, "Assistant Mode")

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║          N-Tech Energy Solutions — Solar AI Agent           ║
║                      {label:<38}║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /mode research    — Switch to research mode
  /mode content     — Switch to content generation mode
  /mode marketing   — Switch to marketing insights mode
  /mode chat        — Switch to general chat mode
  /save             — Save last response to file
  /clear            — Clear conversation history
  /quit or /exit    — Exit

Type your request and press Enter.
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

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break

                elif cmd == "/clear":
                    self.conversation_history = []
                    print("[Conversation history cleared]")
                    continue

                elif cmd == "/save":
                    if last_response:
                        self.save_output = True
                        self.output_dir.mkdir(exist_ok=True)
                        self.save_to_file(last_response, prefix=self.mode)
                    else:
                        print("[No response to save yet]")
                    continue

                elif cmd == "/mode":
                    if len(parts) > 1:
                        new_mode = parts[1].strip().lower()
                        if new_mode in MODE_PROMPTS:
                            self.mode = new_mode
                            self.system_prompt = (
                                BUSINESS_CONTEXT.format(date=datetime.now().strftime("%B %d, %Y"))
                                + "\n\n"
                                + MODE_PROMPTS[new_mode]
                            )
                            self.conversation_history = []
                            print(f"[Switched to {mode_labels.get(new_mode, new_mode)} — history cleared]")
                        else:
                            print(f"[Unknown mode: {new_mode}. Options: research, content, marketing, chat]")
                    else:
                        print(f"[Current mode: {self.mode}]")
                    continue

                else:
                    print(f"[Unknown command: {cmd}]")
                    continue

            # Run agent
            print("\nAgent: ", end="", flush=True)
            try:
                last_response = self._stream_response(user_input)
                print()  # Newline after streaming completes
            except anthropic.APIError as e:
                print(f"\n[API Error: {e}]")
            except Exception as e:
                print(f"\n[Error: {e}]")
                raise


# ─── Quick-Run Helpers ───────────────────────────────────────────────────────

def quick_research(topic: str) -> str:
    """Run a one-shot research query and return the result."""
    agent = SolarAgent(mode="research")
    return agent._stream_response(topic)


def generate_blog_post(topic: str, city: str = None) -> str:
    """Generate a blog post on the given topic."""
    agent = SolarAgent(mode="content")
    prompt = f"Write a complete, SEO-optimized blog post about: {topic}"
    if city:
        prompt += f"\nFocus on how this applies to homeowners in {city}, TX."
    return agent._stream_response(prompt)


def generate_city_page(city: str, county: str) -> str:
    """Generate a city landing page."""
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
        choices=["research", "content", "marketing", "chat"],
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

    args = parser.parse_args()

    # One-shot modes
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

    # Interactive mode
    agent = SolarAgent(mode=args.mode, save_output=args.save)
    agent.run()


if __name__ == "__main__":
    main()
