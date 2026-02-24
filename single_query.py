import time
import re
import asyncio
import csv
from smolagents import OpenAIModel, GoogleSearchTool
from curl_cffi.requests import AsyncSession
from bs4 import BeautifulSoup
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEARCH_POOL_SIZE = 8              # How many URLs per query
REQUIRED_VALID_SITES = 2          # How many fastest sites to keep
NETWORK_TIMEOUT = 3.0             # Seconds per HTTP request
MAX_CHARS_PER_SITE = 800          # Truncate page text per site
MAX_CONCURRENT_LLM_REQUESTS = 4   # Semaphore: max parallel LLM calls

# SerpAPI key for GoogleSearchTool (smolagents expects this env name)
os.environ["SERPAPI_API_KEY"] = "6a52884e04b2ab515fecd1cd1bb04b4dfb7024f1aef8825e1a842c8f14d9e97a"

# ============================ CORE FUNCTIONS ===================================

async def fetch_single_site(session, url):
    """Fast async fetch with aggressive cleaning."""
    try:
        resp = await session.get(url, impersonate="chrome", timeout=NETWORK_TIMEOUT)
        if resp.status_code != 200:
            return None

        if "xml" in resp.headers.get("Content-Type", ""):
            soup = BeautifulSoup(resp.text, "xml")
        else:
            soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(
            [
                "script", "style", "nav", "footer", "header",
                "aside", "form", "iframe", "svg", "button"
            ]
        ):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        if len(text) < 50:
            return None

        return url, text[:MAX_CHARS_PER_SITE]
    except Exception:
        return None


class WebFetcher:
    """Reusable async HTTP client with race-fetch behavior."""

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = AsyncSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def race_fetch(self, urls):
        """Fetch multiple URLs, stop when REQUIRED_VALID_SITES succeed."""
        tasks = [asyncio.create_task(fetch_single_site(self.session, u)) for u in urls]
        valid_results = []

        pending = tasks
        while pending and len(valid_results) < REQUIRED_VALID_SITES:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    res = task.result()
                except Exception:
                    res = None
                if res:
                    valid_results.append(res)
                if len(valid_results) >= REQUIRED_VALID_SITES:
                    break

        for t in pending:
            t.cancel()
        return valid_results

# ============================== PIPELINE ===================================

async def process_query(query, model, search_tool, llm_sem, fetcher: WebFetcher):
    """Full RAG pipeline for a single query; returns metrics dict."""
    t_start_total = time.time()
    loop = asyncio.get_running_loop()

    # --- STEP 1: SEARCH (sync tool in thread) ---
    search_q = f"{query} Bangladesh" if "Bangladesh" not in query else query
    try:
        # GoogleSearchTool returns markdown-like text with links
        search_raw = await loop.run_in_executor(None, search_tool, search_q)

        # Extract https://... tokens and strip markdown wrappers
        raw_urls = re.findall(r'https?://[^\s\)]+', search_raw)
        clean_urls = [u.strip("[]()") for u in raw_urls]

        # Deterministic, unique, limited pool
        urls = sorted(set(clean_urls))[:SEARCH_POOL_SIZE]

        
    except Exception:
        urls = []

    web_output = "No info found."
    t_gpu_start = 0.0
    t_gpu_end = 0.0
    t_queue_enter = 0.0
    urls_used = []

    # --- STEP 2: PARALLEL WEB FETCH ---
    if urls:
        valid_contents = await fetcher.race_fetch(urls)

        # --- STEP 3: LLM INFERENCE (GPU Bound) ---
        if valid_contents:
            urls_used = [u for (u, _) in valid_contents]
            texts = [t for (_, t) in valid_contents]

            context = "\n---\n".join(texts)
            prompt = (
                f"Context (Jan 2026):\n{context}\n\n"
                f"Q: {query}\n"
                f"Answer (max 2 sentences):"
            )

            t_queue_enter = time.time()
            async with llm_sem:
                t_gpu_start = time.time()
                ans_res = await loop.run_in_executor(
                    None,
                    lambda: model([{"role": "user", "content": prompt}]),
                )
                t_gpu_end = time.time()

            web_output = ans_res.content.strip()

    # --- METRICS CALCULATION ---
    t_end_total = time.time()
    total_latency = round(t_end_total - t_start_total, 2)

    if t_gpu_end > 0:
        queue_wait = round(t_gpu_start - t_queue_enter, 2)
        gpu_time = round(t_gpu_end - t_gpu_start, 2)
    else:
        queue_wait = 0.0
        gpu_time = 0.0

    print(
        f"Total: {total_latency}s | GPU: {gpu_time}s | "
        f"Waited: {queue_wait}s -> {query[:30]}..."
    )

    return {
        "query": query,
        "output": web_output,
        "sources": urls_used,
        "Total_Latency_Seconds": total_latency,
        "Queue_Wait_Seconds": queue_wait,
        "GPU_Processing_Seconds": gpu_time,
    }

# ============================== SINGLE-RUN API ==============================

async def run_single_query(query: str, save_csv: bool = False):
    """Initialize model + tools, run one query end-to-end, optionally save CSV."""
    script_start = time.time()

    model = OpenAIModel(
        model_id="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        api_base="http://localhost:5000/v1",
        api_key="no-key",
    )
    search_tool = GoogleSearchTool(provider="serpapi")

    llm_sem = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

    print("Starting single-query run")
    print(f"Concurrency limit (LLM): {MAX_CONCURRENT_LLM_REQUESTS}")
    print("--------------------------------------------------")

    async with WebFetcher() as fetcher:
        result = await process_query(query, model, search_tool, llm_sem, fetcher)

    print("--------------------------------------------------")
    print(f"Script Runtime: {round(time.time() - script_start, 2)}s")

    if save_csv:
        output_file = "single_query_result.csv"
        headers = [
            "query",
            "output",
            "Sources",
            "Total_Latency_Seconds",
            "Queue_Wait_Seconds",
            "GPU_Processing_Seconds",
        ]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            row = result.copy()
            row["Sources"] = " | ".join(result.get("sources", []))
            writer.writerow(row)
        print(f"Saved: {output_file}")

    return result


