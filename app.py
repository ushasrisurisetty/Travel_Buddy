# concierge_full_project.py
"""
Concierge Itinerary Planner (Complete Project) - patched
- Adds dietary parsing & filtering (counts + allergies)
- Passes dietary context to LLM extractor
- Marks restaurants that likely accommodate dietary needs
- Fixes itinerary date advancement and rotation of daily items
"""

import os
import json
import time
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
# optional config import — if you have a config.py with keys uncomment
from config import SERPAPI_KEY, GEMINI_API_KEY
import streamlit as st
from dateutil import parser as dateparser

# --- Optional: import google genai (Gemini) if available ---
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False

# -----------------------
# Configuration / Defaults
# -----------------------
# SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")

SERPAPI_URL = "https://serpapi.com/search.json"

# -----------------------
# Dietary parsing helpers
# -----------------------
ALLERGEN_KEYWORDS = ["peanut", "peanuts", "nut", "nuts", "gluten", "dairy", "egg", "soy", "shellfish", "sesame"]
WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_dietary_notes(dietary: str):
    """Simple heuristic parser: extracts vegetarian/non-veg counts and allergen keywords.
    Returns: {veg_count, nonveg_count, allergies}
    """
    text = (dietary or "").lower()
    veg_count = 0
    nonveg_count = 0

    # numeric counts like '2 non veg' or '1 vegetarian'
    m = re.search(r"(\d+)\s*vegetarian", text)
    if m:
        veg_count = int(m.group(1))
    m = re.search(r"(\d+)\s*veg(?!etarian)", text)
    if m:
        veg_count = int(m.group(1))
    m = re.search(r"(\d+)\s*(non-?veg|non ?vegetarian|nonveg|meat)", text)
    if m:
        nonveg_count = int(m.group(1))

    # words like 'two non veg and one vegetarian'
    for word, n in WORD_NUMS.items():
        if f"{word} non" in text or f"{n} non" in text:
            nonveg_count = nonveg_count or n
        if f"{word} veg" in text or f"{n} veg" in text:
            veg_count = veg_count or n
        if f"{word} vegetarian" in text:
            veg_count = veg_count or n
        if f"{word} non vegetarian" in text or f"{word} nonveg" in text:
            nonveg_count = nonveg_count or n

    allergies = [k for k in ALLERGEN_KEYWORDS if k in text]

    return {"veg_count": veg_count, "nonveg_count": nonveg_count, "allergies": allergies}


# -----------------------
# Prompt templates (updated schema includes dietary accommodation fields)
# -----------------------
EXTRACTION_PROMPT_TEMPLATE = """
You are a strict extractor. I will provide up to {limit} Google search results (title, snippet, and URL).
Task: From these results extract up to {max_items} items of type {type_label} for the query "{query}".
Output: JSON ONLY (no commentary) — an array of objects matching this schema:

{{
  "type": "restaurant|hotel|attraction|event",
  "name": "string or null",
  "address": "string or null",
  "phone": "string or null",
  "website": "string or null",
  "photos": ["url", ...] (0..3),
  "rating": number or null,
  "price_level": "$|$$|$$$|null",
  "hours": "string or null",
  "distance_m": number or null,
  "tags": ["vegan","family","outdoor", ...],
  "source_urls": ["https://...","https://..."],
  "notes": "short summary string or null",
  "accommodates_dietary": true|false|null,
  "accommodation_details": "string or null"
}}

Rules:
- ONLY extract facts that are present in the title/snippet or in the provided URLs.
- DO NOT INVENT phone numbers, hours, or ratings. Use null if missing.
- Use null for any unknown fields.
- Deduplicate by near-identical names; return the best available record.
- Include source_urls that you used to extract information.
- Return well-formed JSON only.

Here are the SERP results (title / snippet / url). Use them to extract facts:

{serp_snippets}
"""

# -----------------------
# SERP -> snippets builder
# -----------------------
def build_snippets_from_serp(serp_json: dict, limit: int = 8) -> str:
    pieces = []
    if serp_json is None:
        return ""
    if "local_results" in serp_json and serp_json.get("local_results"):
        local = serp_json["local_results"]
        if isinstance(local, dict) and local.get("places"):
            for i, p in enumerate(local.get("places")[:limit], start=1):
                title = p.get("title") or p.get("name") or ""
                snippet = p.get("snippet") or ""
                link = p.get("link") or p.get("url") or ""
                pieces.append(f"{i}) TITLE: {title}\nSNIPPET: {snippet}\nURL: {link}")
    organic = serp_json.get("organic_results") or serp_json.get("organic") or []
    for i, r in enumerate(organic[:limit], start=len(pieces) + 1):
        title = r.get("title", "") or r.get("position_title", "")
        snippet = r.get("snippet", "") or r.get("snippet_highlighted", "")
        link = r.get("link") or r.get("url") or r.get("source")
        pieces.append(f"{i}) TITLE: {title}\nSNIPPET: {snippet}\nURL: {link}")
    return "\n\n".join(pieces)


# -----------------------
# Utilities & caching
# -----------------------
@st.cache_data(show_spinner=False, ttl=600)
def serp_search_google(q: str, num: int = 10) -> dict:
    """Call SerpAPI Google engine. Caches results for TTL to avoid repeated cost during demo."""
    if not SERPAPI_KEY:
        return {"error": "SERPAPI key not configured"}
    params = {
        "engine": "google",
        "q": q,
        "num": num,
        "api_key": SERPAPI_KEY,
    }
    r = requests.get(SERPAPI_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def call_gemini(prompt: str, max_tokens: int = 1500, temperature: float = 0.0) -> str:
    """
    Gemini call WITHOUT thinking_config (compatible with all Gemini models).
    """

    if not GEMINI_SDK_AVAILABLE:
        raise RuntimeError("google-genai SDK not installed or importable.")

    client = genai.Client()

    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=config
    )

    return response.text


# -----------------------
# Core agent: SerpAPI -> Gemini -> normalize (now accepts extra_context)
# -----------------------
def run_agent_serp_then_gemini(query: str, type_label: str, max_items: int = 5, serp_limit: int = 8, extra_context: str = "") -> List[Dict]:
    """
    High-level agent: runs a SerpAPI query, builds context snippets, asks Gemini to extract structured JSON,
    parses the JSON, applies verification heuristics, and returns canonical items.
    """
    serp = serp_search_google(query, num=10)
    if serp.get("error"):
        return [{"type": type_label, "name": None, "notes": f"SERPAPI error: {serp.get('error')}", "source_urls": []}]

    snippets = build_snippets_from_serp(serp, limit=serp_limit)
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        max_items=max_items,
        type_label=type_label,
        query=query.replace("\n", " "),
        serp_snippets=snippets,
        limit=serp_limit
    )

    if extra_context:
        prompt = prompt + "\n\nExtra context for extraction:\n" + extra_context

    # Call Gemini (may raise if not configured)
    try:
        raw = call_gemini(prompt, max_tokens=1200, temperature=0.0)
    except Exception as e:
        return [{"type": type_label, "name": None, "notes": f"Gemini/LLM error: {str(e)}", "source_urls": []}]

    # Parse JSON robustly
    items = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("error"):
            return [{"type": type_label, "name": None, "notes": f"LLM error: {parsed.get('error')}", "raw": raw}]
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [parsed]
    except Exception:
        # attempt substring extraction (best-effort)
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end != -1 and end > start:
                substr = raw[start:end]
                items = json.loads(substr)
            else:
                return [{"type": type_label, "name": None, "notes": "LLM did not return JSON", "raw": raw}]
        except Exception as e:
            return [{"type": type_label, "name": None, "notes": f"Failed parsing LLM output: {str(e)}", "raw": raw}]

    # Post-process and dedupe
    canonical_items = []
    seen_keys = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("title") or None
        key = "".join(ch for ch in (name or "").lower() if ch.isalnum() or ch.isspace()).strip() if name else f"no_name_{len(seen_keys)}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        source_urls = it.get("source_urls") or []
        ver = compute_verification_from_urls(source_urls)
        it["verification_level"] = ver["verification_level"]
        it["verification_reason"] = ver["verification_reason"]
        # Ensure all expected fields present
        for f in ["type", "name", "address", "phone", "website", "photos", "rating", "price_level", "hours", "distance_m", "tags", "source_urls", "notes", "accommodates_dietary", "accommodation_details"]:
            if f not in it:
                it[f] = None
        canonical_items.append(it)
    return canonical_items


# -----------------------
# Simple deterministic verification heuristic
# -----------------------
def compute_verification_from_urls(urls: List[str]) -> Dict[str, str]:
    if not urls:
        return {"verification_level": "suggested", "verification_reason": "No sources found"}
    for u in urls:
        u_lower = u.lower()
        if any(x in u_lower for x in ["ticketmaster", "eventbrite", "opentable", "booking.com", "airbnb.com"]):
            return {"verification_level": "high", "verification_reason": f"Found official ticket/reservation site: {u}"}
    for u in urls:
        if any(x in u.lower() for x in ["yelp.com", "google.com/maps", "tripadvisor", "foursquare.com"]):
            return {"verification_level": "medium", "verification_reason": f"Found public listing: {u}"}
    return {"verification_level": "suggested", "verification_reason": "Please visit the website for booking."}


# -----------------------
# Post-filter: mark dietary matches
# -----------------------
def mark_dietary_matches(items: List[Dict], parsed_diet: Dict):
    allergies = parsed_diet.get("allergies", [])
    veg_count = parsed_diet.get("veg_count", 0)
    nonveg_count = parsed_diet.get("nonveg_count", 0)

    veg_indicators = ["vegetarian", "vegan", "plant-based", "meat-free", "vegetarian options"]
    allergy_indicators = ["allergy", "allergies", "gluten-free", "nut-free", "peanut-free", "we can accommodate", "notify us of allergies"]

    for it in items:
        text_fields = " ".join(filter(None, [
            (it.get("notes") or ""),
            " ".join(it.get("tags") or []),
            " ".join(it.get("source_urls") or []),
            (it.get("address") or ""),
            (it.get("website") or "")
        ])).lower()

        accommodates = False
        details = []
        if veg_count > 0:
            if any(k in text_fields for k in veg_indicators):
                accommodates = True
                details.append("Vegetarian options noted")
        if allergies:
            if any(k in text_fields for k in allergy_indicators):
                accommodates = True
                details.append("Allergy handling mentioned")
            else:
                # If menu mentions the allergen explicitly, flag for caution
                if any(a in text_fields for a in allergies):
                    details.append("Menu mentions allergen — confirm with restaurant")
        it["accommodates_dietary"] = "Yes" if accommodates else None
        it["accommodation_details"] = "; ".join(details) if details else None

    return items


# -----------------------
# Orchestrator (passes dietary context into foodie agent)
# -----------------------
def orchestrate_plan(city: str, date_from: str, date_to: str, party: int, budget: int, dietary: str, max_workers: int = 3):
    foodie_query = f"restaurants in {city}"
    hotels_query = f"hotels in {city} budget {budget} per night"
    scout_query = f"events and things to do in {city} {date_from} to {date_to}"

    parsed_diet = parse_dietary_notes(dietary)
    parts = []
    if parsed_diet.get("nonveg_count"):
        parts.append(f"{parsed_diet['nonveg_count']} non-vegetarian diners")
    if parsed_diet.get("veg_count"):
        parts.append(f"{parsed_diet['veg_count']} vegetarian diners")
    if parsed_diet.get("allergies"):
        parts.append("Allergies: " + ", ".join(parsed_diet.get("allergies")))
    dietary_context = "; ".join(parts)

    agents = {
        "foodie": lambda: run_agent_serp_then_gemini(foodie_query + (" " + dietary if dietary else ""), "restaurant", max_items=6, extra_context=dietary_context),
        "hotels": lambda: run_agent_serp_then_gemini(hotels_query, "hotel", max_items=4),
        "scout": lambda: run_agent_serp_then_gemini(scout_query, "attraction", max_items=6),
    }

    results = {}
    timings = {}
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fn): name for name, fn in agents.items()}
        for fut in as_completed(future_map):
            name = future_map[fut]
            t0 = time.time()
            try:
                res = fut.result()
                results[name] = res
            except Exception as e:
                results[name] = [{"type": name, "name": None, "notes": f"Agent error: {str(e)}", "source_urls": []}]
            timings[name] = time.time() - t0
    total_time = time.time() - start_all
    return results, timings, total_time


# -----------------------
# Itinerary builder (fixed date advancement + rotation of items)
# -----------------------
def build_itinerary_from_results(results: Dict, city: str, date_from: str, date_to: str, party: int, budget: int, dietary: str):
    itinerary = {
        "meta": {
            "city": city,
            "date_from": date_from,
            "date_to": date_to,
            "party": party,
            "budget_per_person": budget,
            "dietary": dietary,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        },
        "days": []
    }
    try:
        d_from = dateparser.parse(date_from).date()
        d_to = dateparser.parse(date_to).date()
        day_count = max(1, (d_to - d_from).days + 1)
    except Exception:
        d_from = date.today()
        day_count = 2

    foodie = results.get("foodie", []) or []
    hotels = results.get("hotels", []) or []
    scout = results.get("scout", []) or []

    def pick_item(lst, idx):
        if not lst:
            return None
        return lst[idx % len(lst)]

    for day in range(day_count):
        current_date = (d_from + timedelta(days=day)).isoformat()
        hotel_for_day = hotels[day] if len(hotels) > day else (hotels[0] if hotels else None)
        morning_item = pick_item(scout, day)
        afternoon_item = pick_item(foodie, day)
        if len(foodie) > (day + 1):
            evening_item = foodie[(day + 1) % len(foodie)]
        else:
            evening_item = pick_item(scout, day + 1)

        day_obj = {
            "day_index": day + 1,
            "date": current_date,
            "hotel": hotel_for_day,
            "morning": [morning_item] if morning_item else [],
            "afternoon": [afternoon_item] if afternoon_item else [],
            "evening": [evening_item] if evening_item else []
        }
        itinerary["days"].append(day_obj)

    return itinerary


# -----------------------
# Streamlit UI helpers
# -----------------------
def sidebar_inputs():
    st.sidebar.header("Trip details")
    city = st.sidebar.text_input("City", value="Chicago")
    col1, col2 = st.sidebar.columns(2)
    date_from = col1.date_input("Start date", value=date.today())
    date_to = col2.date_input("End date", value=date.today())
    party = st.sidebar.number_input("Party size", min_value=1, max_value=20, value=3)
    budget = st.sidebar.number_input("Budget per person", min_value=50, max_value=5000, value=500)
    dietary = st.sidebar.text_input("Dietary notes", "One Vegan")
    max_workers = 3
    # show_timings = st.sidebar.checkbox("Show agent timings", value=True)
    live_mode = st.sidebar.checkbox("Live mode (SerpAPI + Gemini)", value=True)
    return city, date_from.isoformat(), date_to.isoformat(), party, budget, dietary, max_workers, live_mode


def render_item_card(item: Dict):
    # print("ITEM",item)
    st.subheader(item.get("name") or "—")
    cols = st.columns([3, 2])
    with cols[0]:
        st.write(f"**Type:** {item.get('type', 'place')}")
        if item.get("address"):
            st.write(f"**Address:** {item.get('address')}")
        if item.get("phone"):
            st.write(f"**Phone:** {item.get('phone')}")
        if item.get("website"):
            st.markdown(f"**Website:** [{item.get('website')}]({item.get('website')})")
        if item.get("rating"):
            st.write(f"**Rating:** {item.get('rating')}  |  **Price:** {item.get('price_level') or '—'}")
        if item.get("notes"):
            st.write(f"**Notes:** {item.get('notes')}")
        if item.get("source_urls"):
            st.write("**Sources:**")
            for u in (item.get("source_urls") or [])[:4]:
                st.markdown(f"- [{u}]({u})")
    with cols[1]:
        verification = item.get("verification_level") or "suggested"
        st.markdown(f"**Verification:** {verification}")
        reason = item.get("verification_reason")
        if reason:
            st.write(reason)
        # dietary accommodations if present
        if item.get("accommodates_dietary") is not None:
            st.write(f"**Accommodates dietary Requirements:** {item.get('accommodates_dietary')}")
        if item.get("accommodation_details"):
            st.write(item.get("accommodation_details"))
        photos = item.get("photos") or []
        if photos:
            try:
                st.image(photos[0], width=200)
            except Exception:
                pass


# -----------------------
# Main
# -----------------------
def main():
    st.set_page_config(page_title=" Travel Buddy", layout="wide")
    st.title("🧭 Travel Buddy")
    st.markdown(
        "This app finds Location , Restaurants , Hotels and provides curated suggestions with source links and verification badges."
    )

    city, date_from, date_to, party, budget, dietary, max_workers,live_mode = sidebar_inputs()

    if st.button("Generate itinerary"):
        st.info("Your adventure is loading! Travel Buddy is building your itinerary now.")
        if not live_mode:
            st.success("Running in mock mode (no external API calls).")
            results = {
                "foodie": [
                    {
                        "type": "restaurant",
                        "name": "GreenLeaf Vegan Bistro",
                        "address": "123 Green St, Chicago, IL",
                        "phone": "+1-312-555-0100",
                        "website": "https://greenleaf.example",
                        "photos": [],
                        "rating": 4.7,
                        "price_level": "$$",
                        "hours": "Sat 11:00–22:00; Sun 10:00–21:00",
                        "distance_m": 1200,
                        "tags": ["vegan"],
                        "source_urls": ["https://greenleaf.example"],
                        "notes": "Popular brunch spot; recommended dish: avocado tofu scramble",
                        "verification_level": "medium",
                        "verification_reason": "OpenTable link present",
                        "accommodates_dietary": True,
                        "accommodation_details": "Vegetarian options noted"
                    }
                ],
                "hotels": [
                    {
                        "type": "hotel",
                        "name": "StayCentral Hotel",
                        "address": "200 Main St, Chicago, IL",
                        "phone": "+1-312-555-0300",
                        "website": "https://staycentral.example",
                        "photos": [],
                        "rating": 4.2,
                        "price_level": "$$",
                        "distance_m": 800,
                        "tags": ["central"],
                        "source_urls": ["https://staycentral.example"],
                        "notes": "Comfortable, central location. Free breakfast",
                        "verification_level": "suggested",
                        "verification_reason": "Public listing; rates estimated"
                    }
                ],
                "scout": [
                    {
                        "type": "attraction",
                        "name": "Millennium Park",
                        "address": "201 E Randolph St, Chicago, IL",
                        "website": "https://www.chicago.gov/millenniumpark",
                        "photos": [],
                        "rating": 4.8,
                        "distance_m": 600,
                        "tags": ["outdoor", "free"],
                        "source_urls": ["https://www.chicago.gov/millenniumpark"],
                        "notes": "Free outdoor spaces, recommended to visit Cloud Gate",
                        "verification_level": "high",
                        "verification_reason": "Official site"
                    }
                ]
            }
            timings = {"foodie": 0.0, "hotels": 0.0, "scout": 0.0}
            total_time = 0.0
        else:
            if not SERPAPI_KEY:
                st.error("SERPAPI_KEY not set in environment. Switch to mock mode or set SERPAPI_KEY.")
                return
            if not GEMINI_SDK_AVAILABLE:
                st.error("google-genai SDK not available. Install 'google-genai' or switch to mock mode.")
                return
            try:
                results, timings, total_time = orchestrate_plan(city, date_from, date_to, party, budget, dietary, max_workers)
            except Exception as e:
                st.error(f"Orchestrator error: {str(e)}")
                return

            # post-process dietary marks for foodie results
            parsed = parse_dietary_notes(dietary)
            results["foodie"] = mark_dietary_matches(results.get("foodie", []), parsed)

        itinerary = build_itinerary_from_results(results, city, date_from, date_to, party, budget, dietary)

        # st.success(f"Itinerary generated in {total_time:.1f}s")

        st.header(f"Itinerary — {itinerary['meta']['city']} • {itinerary['meta']['date_from']} → {itinerary['meta']['date_to']}")
        st.markdown(f"**Party**: {party} • **Budget per person**: ${budget} • **Dietary**: {dietary}")

        for day in itinerary["days"]:
            st.markdown("---")
            st.subheader(f"Day {day['day_index']} — {day.get('date','')}")
            if day.get("hotel"):
                st.markdown("**Hotel suggestion**")
                render_item_card(day["hotel"])
            if day.get("morning"):
                st.markdown("**Morning**")
                for it in day["morning"]:
                    render_item_card(it)
            if day.get("afternoon"):
                st.markdown("**Afternoon**")
                for it in day["afternoon"]:
                    render_item_card(it)
            if day.get("evening"):
                st.markdown("**Evening**")
                for it in day["evening"]:
                    render_item_card(it)


if __name__ == "__main__":
    main()
