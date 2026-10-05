"""ELP Week 3 — Team 1 Student Starter
Focused AI Research Agent: Planner + Tools

Week 2 already gave you:
    Question -> Planner -> Tavily -> DeepSeek Response

Week 3 goal:
    Question -> Planner -> Appropriate Research Tools -> DeepSeek Response

Your job is to complete the NEW Week 3 TODOs only.

Run:
    python -m pip install -r requirements_team1.txt
    streamlit run team1_week3_student.py

.env:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key

Important:
- Never commit .env.
- SEC asks automated clients to identify themselves. SEC_USER_AGENT is optional here;
  if you add one, use something like: Your Name your-email@example.com
"""

import json
import os
from typing import TypedDict

import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tavily import TavilyClient


# ---------------- Configuration ----------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Nittany AI ELP educational-project contact@example.com",
)

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ---------------- Team 1 Planner ----------------
# This extends Week 2 homework: each task now includes a suggested source type.
TEAM_1_SYSTEM_PROMPT = """
You are the Planner for a single-agent competitive-intelligence research system.

Break the user's business research question into 4-6 specific, non-overlapping
research tasks. For each task, choose the most appropriate source type:

- "web" for current websites, news, products, partnerships, market activity,
  customer information, and general competitive intelligence.
- "financials" for public-company market/financial metrics.
- "filings" for official SEC filings and regulatory disclosures.

If a task uses "financials" or "filings", include the public-company ticker
symbol when you can identify it confidently. Otherwise use null.

Return only valid JSON in exactly this shape:
{
  "tasks": [
    {
      "task": "specific research task",
      "source_type": "web",
      "ticker": null
    }
  ]
}

Do not include markdown or text outside the JSON object.
""".strip()


class ResearchState(TypedDict):
    question: str
    research_plan: list[dict]
    research_results: list[dict]
    response: str


def call_model_json(system_prompt: str, user_text: str) -> dict:
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def plan_node(state: ResearchState) -> dict:
    output = call_model_json(TEAM_1_SYSTEM_PROMPT, state["question"])
    tasks = output.get("tasks")

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner must return a non-empty 'tasks' list.")

    allowed = {"web", "financials", "filings"}
    cleaned = []
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("Each Planner task must be a JSON object.")

        task = str(item.get("task", "")).strip()
        source_type = str(item.get("source_type", "")).strip().lower()
        ticker = item.get("ticker")

        if not task:
            raise ValueError("Every Planner task needs a non-empty 'task'.")
        if source_type not in allowed:
            raise ValueError(
                "source_type must be web, financials, or filings."
            )

        cleaned.append(
            {
                "task": task,
                "source_type": source_type,
                "ticker": str(ticker).upper().strip() if ticker else None,
            }
        )

    return {"research_plan": cleaned}


# ---------------- Research tools ----------------
def search_web(task: str) -> list[dict]:
    """Current web/news/company information through Tavily."""
    response = tavily.search(
        query=task,
        search_depth="advanced",
        max_results=4,
    )
    return [
        {
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
        }
        for result in response.get("results", [])
    ]


def get_financials(ticker: str) -> list[dict]:
    """Small public-company snapshot using yfinance."""
    if not ticker:
        return [{"error": "Planner did not provide a ticker."}]

    # We already create the Yahoo Finance company object for you.
    company = yf.Ticker(ticker)
    fast = company.fast_info

    # TODO 1 — FINANCIAL DATA
    #
    # Build a dictionary called fields.
    #
    # Try to collect these four values from fast:
    #   - last_price
    #   - market_cap
    #   - year_high
    #   - year_low
    #
    # Hint:
    # getattr(fast, "last_price")
    #
    # Some values may be missing, so use try/except.
    # Only add a value to fields if it is not None.

    fields = {}
    for name in ("last_price", "market_cap", "year_high", "year_low"):
        try:
            value = getattr(fast, name)
        except Exception:
            continue
        if value is not None:
            fields[name] = value

    if not fields:
        return [{"error": f"No financial snapshot returned for {ticker}."}]

    return [
        {
            "title": f"{ticker} market snapshot",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "content": json.dumps(fields, default=str),
        }
    ]


def get_sec_filings(ticker: str) -> list[dict]:
    """Recent SEC filing metadata for a public-company ticker."""
    if not ticker:
        return [{"error": "Planner did not provide a ticker."}]

    headers = {"User-Agent": SEC_USER_AGENT}

    # Find the company's SEC CIK number.
    ticker_response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers,
        timeout=20,
    )
    ticker_response.raise_for_status()
    companies = ticker_response.json()

    match = next(
        (
            item
            for item in companies.values()
            if str(item.get("ticker", "")).upper() == ticker.upper()
        ),
        None,
    )

    if not match:
        return [{"error": f"Could not find SEC CIK for ticker {ticker}."}]

    cik = str(match["cik_str"]).zfill(10)

    # Get the company's recent SEC filings.
    submissions_response = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=headers,
        timeout=20,
    )
    submissions_response.raise_for_status()

    recent = (
        submissions_response.json()
        .get("filings", {})
        .get("recent", {})
    )

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])

    results = []

    # TODO 2 — SEC FILINGS
    #
    # Loop through:
    # forms, dates, accessions, and documents together.
    #
    # Only keep these filing types:
    #   10-K
    #   10-Q
    #   8-K
    #
    # For each filing you keep:
    #
    # 1. Remove "-" from the accession number.
    #
    # 2. Build this SEC URL:
    #
    # https://www.sec.gov/Archives/edgar/data/
    # {CIK}/{accession_without_dashes}/{document}
    #
    # 3. Append a dictionary to results containing:
    #       title
    #       url
    #       content
    #
    # Stop after collecting 3 filings.
    #
    # Hint:
    # zip(forms, dates, accessions, documents)

    for form, date, accession, document in zip(
        forms, dates, accessions, documents
    ):
        if form not in {"10-K", "10-Q", "8-K"}:
            continue

        accession_without_dashes = str(accession).replace("-", "")
        url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_without_dashes}/{document}"
        )
        results.append(
            {
                "title": f"{ticker} {form} ({date})",
                "url": url,
                "content": (
                    f"{form} filed on {date}. "
                    f"Accession number {accession}."
                ),
            }
        )
        if len(results) >= 3:
            break

    return results or [
        {"error": f"No recent 10-K, 10-Q, or 8-K found for {ticker}."}
    ]


# ---------------- Week 3 node ----------------
def research_node(state: ResearchState) -> dict:
    """Execute the Planner's FULL plan with the appropriate research tool."""
    research_results = []

    # TODO 3 — EXECUTE THE FULL RESEARCH PLAN
    #
    # Loop through EVERY item in:
    #
    # state["research_plan"]
    #
    # For each task:
    #
    # 1. Read:
    #       task
    #       source_type
    #       ticker
    #
    # 2. Choose the correct tool:
    #
    #       web
    #           -> search_web(...)
    #
    #       financials
    #           -> get_financials(...)
    #
    #       filings
    #           -> get_sec_filings(...)
    #
    # 3. Save each result inside research_results.
    #
    # Each saved item should contain:
    #
    # {
    #     "task": ...,
    #     "source_type": ...,
    #     "ticker": ...,
    #     "results": ...
    # }
    #
    # IMPORTANT:
    # Research ALL Planner tasks.
    # Do not use research_plan[0].

    for item in state["research_plan"]:
        task = item["task"]
        source_type = item["source_type"]
        ticker = item.get("ticker")

        if source_type == "web":
            results = search_web(task)
        elif source_type == "financials":
            results = get_financials(ticker)
        elif source_type == "filings":
            results = get_sec_filings(ticker)
        else:
            results = [{"error": f"Unknown source type: {source_type}"}]

        research_results.append(
            {
                "task": task,
                "source_type": source_type,
                "ticker": ticker,
                "results": results,
            }
        )

    return {"research_results": research_results}


def response_node(state: ResearchState) -> dict:
    research_text = json.dumps(state["research_results"], indent=2, default=str)

    prompt = f"""
You are the response step in a single-agent business research workflow.

Answer the original question using only the research results supplied below.
Combine useful information from multiple Planner tasks. If a tool failed or a
planned area lacks useful information, say so briefly rather than inventing it.

This is still a Week 3 research response, not the final semester report.

Question:
{state["question"]}

Planner research results:
{research_text}
""".strip()

    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"response": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("respond", response_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


research_graph = build_graph()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ELP Team 1 — Week 3", page_icon="🔎")
st.title("Team 1 — Focused AI Research Agent")
st.caption("Planner → Appropriate Tools → Research → Response")

question = st.text_input(
    "Enter a business research question",
    value="How is NVIDIA positioning itself against AMD in the AI chip market?",
)

if st.button("Research", type="primary") and question.strip():
    initial_state: ResearchState = {
        "question": question.strip(),
        "research_plan": [],
        "research_results": [],
        "response": "",
    }

    try:
        with st.spinner("Planning and researching the full question..."):
            result = research_graph.invoke(initial_state)
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
    else:
        st.subheader("Planner output")
        st.json(result["research_plan"])

        st.subheader("Research from all tasks")
        st.json(result["research_results"])

        st.subheader("Response")
        st.write(result["response"])


# ======================================================================
# ELP Week 4 — Team 1 Student Extension
# Evidence & Traceability + SQLite
# ======================================================================

# ---------------- Week 4: Evidence & Traceability ----------------

import sqlite3
from concurrent.futures import ThreadPoolExecutor

EVIDENCE_DB = "team1_evidence.db"


def init_evidence_db():
    """Create the Week 4 evidence table if it does not exist."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                publication_date TEXT,
                source_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence TEXT,
                related_company TEXT,
                report_section TEXT,
                UNIQUE(claim, url)
            )
        """)
        conn.commit()


def save_evidence_records(records: list[dict]):
    """Save validated evidence records to SQLite."""
    columns = [
        "claim",
        "source",
        "url",
        "publication_date",
        "source_type",
        "evidence",
        "confidence",
        "related_company",
        "report_section",
    ]
    rows = [tuple(record.get(col) for col in columns) for record in records]

    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.executemany(
            f"INSERT OR IGNORE INTO evidence ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            rows,
        )
        conn.commit()


def load_evidence_records() -> list[dict]:
    """Read the evidence ledger back from SQLite."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()


class Week4ResearchState(ResearchState):
    evidence_records: list[dict]


EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a business research workflow.

Convert useful research results into structured Evidence Records.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "...",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Rules:
- claim = a factual statement supported by the source.
- evidence = the passage that supports the claim.
- source and url must come from the supplied research.
- Do not guess a publication date.
- confidence must be high, medium, or low.
- Do not create evidence for unsupported conclusions.
""".strip()


def _extract_one(item: dict, result: dict) -> list[dict]:
    """Ask the model for evidence records from a single source result."""
    source_payload = {
        "task": item.get("task"),
        "source_type": item.get("source_type"),
        "title": result.get("title"),
        "url": result.get("url"),
        "content": result.get("content"),
        "published_date": result.get("published_date"),
    }

    try:
        output = call_model_json(
            EVIDENCE_SYSTEM_PROMPT,
            json.dumps(source_payload, indent=2, default=str),
        )
    except Exception:
        # One failed model call should not sink the whole ledger.
        return []

    extracted = output.get("records") if isinstance(output, dict) else None
    if not isinstance(extracted, list):
        return []

    records = []
    for record in extracted:
        if not isinstance(record, dict):
            continue
        # Traceability: never let the model invent provenance.
        record["url"] = source_payload["url"]
        record["source"] = record.get("source") or source_payload["title"]
        record["source_type"] = source_payload["source_type"]
        record["publication_date"] = source_payload["published_date"]
        if not record.get("related_company"):
            record["related_company"] = item.get("ticker")
        records.append(record)
    return records


def extract_evidence_records(research_results: list[dict]) -> list[dict]:
    """Turn Week 3 research results into structured evidence records."""
    jobs = [
        (item, result)
        for item in research_results
        for result in item.get("results", [])
        if isinstance(result, dict) and "error" not in result
    ]

    # Model calls are slow and independent, so run them in parallel.
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = pool.map(lambda job: _extract_one(*job), jobs)

    return [record for batch in batches for record in batch]


def validate_evidence_records(records: list[dict]) -> list[dict]:
    """Keep only records with the minimum Week 4 evidence fields."""
    required = ["claim", "source", "url", "source_type", "evidence"]

    validated = []
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue
        if not all(str(record.get(field) or "").strip() for field in required):
            continue

        key = (record["claim"].strip().lower(), record["url"].strip())
        if key in seen:
            continue
        seen.add(key)

        if record.get("confidence") not in {"high", "medium", "low"}:
            record["confidence"] = None
        validated.append(record)

    return validated


def evidence_node(state: ResearchState) -> dict:
    """Week 4: extract, validate, and persist evidence from Week 3 research."""
    raw_records = extract_evidence_records(state["research_results"])
    validated = validate_evidence_records(raw_records)
    save_evidence_records(validated)
    return {"evidence_records": validated}


def build_week4_graph():
    """
    Extend the existing Week 3 Team 1 flow:

    Question -> Planner -> Research -> Evidence Extraction -> SQLite Ledger
    """
    graph = StateGraph(Week4ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("evidence", evidence_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "evidence")
    graph.add_edge("evidence", END)

    return graph.compile()


week4_graph = build_week4_graph()


# ---------------- Week 4 UI ----------------

st.divider()
st.subheader("Week 4 — Evidence Ledger")
st.caption("Week 3 research → Evidence Extraction → SQLite Evidence Ledger")

if st.button("Build Evidence Ledger", type="primary") and question.strip():
    week4_state = {
        "question": question.strip(),
        "research_plan": [],
        "research_results": [],
        "response": "",
        "evidence_records": [],
    }

    try:
        with st.spinner("Researching and building the evidence ledger..."):
            evidence_result = week4_graph.invoke(week4_state)
    except Exception as error:
        st.error(f"The Week 4 evidence workflow could not finish: {error}")
    else:
        st.subheader("New Evidence Records")
        st.json(evidence_result["evidence_records"])

        stored_records = load_evidence_records()
        st.subheader("Evidence Ledger (SQLite)")
        st.dataframe(stored_records, use_container_width=True)
        st.caption(f"{len(stored_records)} evidence records stored in {EVIDENCE_DB}")
