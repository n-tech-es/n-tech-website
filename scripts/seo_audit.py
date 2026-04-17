#!/usr/bin/env python3
"""N-Tech Website SEO Audit Agent — powered by Claude API."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "seo-reports"

SYSTEM_PROMPT = """You are an expert SEO auditor for N-Tech Energy Solutions, a solar installation company in North Texas.

BUSINESS CONTEXT:
- Company: N-Tech Energy Solutions LLC
- Service area: 60-mile radius of Chico, TX (Wise, Parker, Jack, Montague Counties)
- Pricing: Starter $2.40/watt | Standard $2.50/watt | Premium $2.70/watt | Power Plus $2.90/watt
- Brand voice: transparent, no-pressure. Always frame solar as an option ("Solar can be..." not "Solar is...")
- Key differentiators: locally installed, you own it from day one, no commissions, no door-to-door sales
- GA Measurement ID must be G-50SQZ12XJX on every page
- Google Ads ID must be AW-17959768934 on every page

REQUIRED ELEMENTS (flag if missing on any page):
1. Google Analytics tag — G-50SQZ12XJX
2. Google Ads tag — AW-17959768934
3. Chamber of Commerce badge (chamberofcommerce.com link in footer)
4. Non-blocking Google Fonts (rel="preload" with onload + noscript fallback)
5. LocalBusiness JSON-LD schema
6. Canonical URL tag
7. Meta description (50–160 characters)
8. Meta keywords tag
9. Exactly one H1 tag
10. Alt text on all images

BRAND VOICE VIOLATIONS TO FLAG:
- "based in Chico" or "based in Wise County" (must say "based in North Texas")
- "Solar is a long-term answer" (must be "Solar can be a long-term answer")
- Any aggressive urgency or commission-sales language

KEYWORD STRATEGY TO CHECK:
- index.html: should target "solar panels north texas", "north texas solar company", "home energy efficiency north texas"
- solar-[city]-tx.html: "[city] solar panels", "[city] TX solar installation", "[city] solar cost"
- energy-audit-[city]-tx.html: "home energy audit [city] TX", "why is my electric bill high [city]"
- the-true-cost-of-solar.html: "how much are solar panels in tx", "solar panel cost texas"
- Blog articles: topical authority keywords matching article subject

INTERNAL LINKING — flag pages with fewer than 4 internal links (excluding nav/footer).

Analyze the provided page data and return a full SEO audit report with these sections:

## Executive Summary
Overall site score (1–10), top 3 wins, top 3 priorities to address first.

## Critical Issues
Missing GA tag, missing schema, missing Chamber badge, broken pages. Each issue: page | problem | fix.

## High Priority
Missing/weak meta descriptions, duplicate H1s, missing keyword tags, brand voice violations.

## Medium Priority
Keyword gap opportunities, thin internal linking, content improvements, meta description length.

## Low Priority
Minor title tag tweaks, alt text, noscript fallbacks, etc.

## Keyword Opportunities
Specific keywords to add to specific pages ranked by search opportunity.

Format in clean, scannable markdown. Be specific — include the exact page filename and exact recommended fix for every issue."""


def extract_page_data(filepath: Path) -> dict:
    """Extract SEO-relevant data from an HTML file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"file": filepath.name, "error": str(e)}

    try:
        soup = BeautifulSoup(content, "lxml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")

    def text(tag):
        return tag.get_text(strip=True) if tag else None

    def attr(tag, a):
        return tag.get(a) if tag else None

    meta_desc = attr(soup.find("meta", attrs={"name": "description"}), "content")
    meta_kw = attr(soup.find("meta", attrs={"name": "keywords"}), "content")
    h1_tags = soup.find_all("h1")
    h2_tags = soup.find_all("h2")

    schemas = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(s.get_text())
            schemas.append(data.get("@type", "?"))
        except Exception:
            pass

    internal_links = sorted({
        a["href"] for a in soup.find_all("a", href=True)
        if a["href"] and not a["href"].startswith(("http", "#", "mailto:", "tel:"))
    })

    imgs_no_alt = [
        img.get("src", "")[:60]
        for img in soup.find_all("img")
        if not img.get("alt")
    ]

    body = soup.find("body")
    word_count = len(body.get_text(" ").split()) if body else 0

    brand_flags = []
    cl = content.lower()
    if "based in chico" in cl or "based in wise county" in cl:
        brand_flags.append("may say 'based in Chico/Wise County' (should be 'based in North Texas')")
    if "solar is a long-term" in cl:
        brand_flags.append("says 'Solar is a long-term answer' (should be 'Solar can be')")

    desc_len = len(meta_desc) if meta_desc else 0

    return {
        "file": filepath.name,
        "title": text(soup.find("title")),
        "title_len": len(text(soup.find("title"))) if soup.find("title") else 0,
        "meta_description": (meta_desc[:200] if meta_desc else None),
        "meta_desc_len": desc_len,
        "meta_desc_ok": 50 <= desc_len <= 160,
        "meta_keywords": (meta_kw[:200] if meta_kw else None),
        "has_meta_keywords": bool(meta_kw),
        "canonical": attr(soup.find("link", attrs={"rel": "canonical"}), "href"),
        "h1_count": len(h1_tags),
        "h1_text": text(h1_tags[0])[:100] if h1_tags else None,
        "h2_count": len(h2_tags),
        "h2_texts": [text(h)[:60] for h in h2_tags[:5]],
        "schema_types": schemas,
        "internal_links": internal_links[:20],
        "internal_link_count": len(internal_links),
        "imgs_no_alt": imgs_no_alt[:5],
        "imgs_no_alt_count": len(imgs_no_alt),
        "has_ga": "G-50SQZ12XJX" in content,
        "has_ga_ads": "AW-17959768934" in content,
        "has_chamber_badge": "chamberofcommerce.com" in content,
        "has_font_preload": ('rel="preload"' in content and "fonts.googleapis.com" in content),
        "has_noscript_font": ("<noscript>" in content and "fonts.googleapis.com" in content),
        "word_count": word_count,
        "brand_flags": brand_flags,
    }


def run_audit(pages_data: list, client: anthropic.Anthropic) -> tuple:
    """Send all page data to Claude with prompt caching. Returns (report_text, usage)."""
    pages_json = json.dumps(pages_data, indent=2, ensure_ascii=False)
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"  Sending {len(pages_data)} pages ({len(pages_json):,} chars) to Claude...")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Audit these {len(pages_data)} pages from the N-Tech Energy Solutions website.\n\n"
                    f"Today's date: {today}\n\n"
                    f"PAGE DATA:\n```json\n{pages_json}\n```\n\n"
                    "Provide a thorough, prioritized SEO audit report in markdown."
                ),
            }
        ],
    )

    usage = response.usage
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    print(
        f"  Tokens — input: {usage.input_tokens:,}  "
        f"cache_write: {cache_write:,}  "
        f"cache_read: {cache_read:,}  "
        f"output: {usage.output_tokens:,}"
    )

    return response.content[0].text, usage


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    html_files = sorted(BASE_DIR.glob("*.html"))
    if not html_files:
        print("No HTML files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(html_files)} HTML files. Extracting SEO data...")

    pages_data, errors = [], []
    for fp in html_files:
        data = extract_page_data(fp)
        if "error" in data:
            errors.append(data)
            print(f"  WARNING: {fp.name} — {data['error']}")
        else:
            pages_data.append(data)
            print(f"  OK  {fp.name}  ({data['word_count']:,} words)")

    print(f"\nExtracted {len(pages_data)} pages. {len(errors)} errors.")
    print("\nRunning Claude SEO analysis...")

    audit_text, usage = run_audit(pages_data, client)

    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M")

    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)

    report = f"""# N-Tech Energy Solutions — SEO Audit Report

**Generated:** {timestamp_display}
**Pages Audited:** {len(pages_data)}
**Parse Errors:** {len(errors)}
**Token Usage:** input {usage.input_tokens:,} | cache_write {cache_write:,} | cache_read {cache_read:,} | output {usage.output_tokens:,}

---

{audit_text}

---

## Pages With Parse Errors

{chr(10).join(f"- `{e['file']}`: {e['error']}" for e in errors) if errors else "_None_"}

---
*Generated by N-Tech SEO Audit Agent · Claude API (claude-sonnet-4-6)*
"""

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"seo_audit_{timestamp_file}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport saved → {report_path}")
    print("\n" + "=" * 60)
    preview = audit_text[:3000]
    print(preview)
    if len(audit_text) > 3000:
        print(f"\n[... {len(audit_text) - 3000:,} more characters — see full report file]")


if __name__ == "__main__":
    main()
