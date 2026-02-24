import gradio as gr
import asyncio
import json
import os
import re
import time
import hashlib
from datetime import datetime
from typing import List, Dict
from curl_cffi.requests import AsyncSession
from selectolax.parser import HTMLParser
from smolagents import OpenAIModel
from dotenv import load_dotenv

load_dotenv()
# ==============================================================================
# 🚀 PERFORMANCE SETTINGS
# ==============================================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CACHE_FILE = "rag_cache.json"
MAX_CHARS = 1000       
MAX_RESULTS = 3       
FAST_TIMEOUT = 5  
ADVANCED_TIMEOUT = 10 # Advanced search needs more time

GLOBAL_CACHE = {}

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f: return json.load(f)
    return {}

GLOBAL_CACHE = load_cache()

# ============================ CORE LOGIC ===================================

class FastParser:
    @staticmethod
    def clean_html(html):
        if not html: return ""
        tree = HTMLParser(html)
        for tag in tree.css('script, style, nav, footer, header, svg'): tag.decompose()
        text = tree.body.text(separator=' ', strip=True) if tree.body else ""
        return re.sub(r'\s+', ' ', text).strip()[:MAX_CHARS]

async def perform_search(query: str, session: AsyncSession, is_advanced: bool):
    # Dynamically adjust settings based on the toggle
    search_depth = "advanced" if is_advanced else "fast"
    timeout = ADVANCED_TIMEOUT if is_advanced else FAST_TIMEOUT
    max_results = 5 if is_advanced else MAX_RESULTS # Fetch more context for advanced

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"{query} Bangladesh latest 2026",
        "search_depth": search_depth, 
        "include_answer": 'basic',
        "max_results": max_results
    }
    
    try:
        resp = await session.post("https://api.tavily.com/search", json=payload, timeout=timeout)
        if resp.status_code == 200:
            res = [{"url": r["url"], "content": FastParser.clean_html(r.get("content", ""))} 
                   for r in resp.json().get("results", [])]
            return sorted(res, key=lambda x: x['url'])
    except Exception as e: 
        print(f"Search error: {e}")
        pass
    return []

async def predict(query, is_advanced):
    # Include search mode in the hash so fast/advanced caches are kept separate
    hash_input = f"{query.lower().strip()}_{is_advanced}"
    query_hash = hashlib.md5(hash_input.encode()).hexdigest()
    
    # 1. Immediate Cache Check
    if query_hash in GLOBAL_CACHE:
        c = GLOBAL_CACHE[query_hash]
        mode_str = " (Advanced)" if is_advanced else " (Fast)"
        yield c['output'], "\n".join(c['sources']), f"{c.get('latency', '0.0')}s{mode_str}"
        return

    t_start = time.time()
    current_time = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

    # 2. Setup Model
    model = OpenAIModel(
        model_id="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        api_base="http://localhost:5000/v1",
        api_key="no-key",
        temperature=0.0,
        max_tokens=500 if is_advanced else 250,
    )

    # 3. Status Update 1: Searching
    mode_text = "Advanced" if is_advanced else "Fast"
    yield f"🔎 Running {mode_text} search...", "", "Calculating..."

    async with AsyncSession() as session:
        results = await perform_search(query, session, is_advanced)
    
    if not results:
        yield f"❌ No results found. (Try increasing timeout or switching modes)", "", "N/A"
        return

    # 4. Status Update 2: Thinking
    sources_str = "\n".join([r['url'] for r in results])
    yield "🧠 Synthesizing answer...", sources_str, f"{round(time.time() - t_start, 2)}s"

    context = "\n---\n".join([r["content"] for r in results])
    
    # --- PRECISION LENGTH-CONTROL PROMPT ---
    prompt = f"""SYSTEM: You are a highly accurate and concise search assistant.
Current date and time: {current_time}. Use this as your absolute reference for 'today', 'now', or 'current events'.

CRITICAL INSTRUCTIONS FOR RESPONSE LENGTH:
You must analyze the user's query and classify it into one of two categories before answering. Apply the corresponding length constraint strictly.

CATEGORY 1: Direct Factual Queries
- Definition: Queries asking for specific names, dates, numbers, locations, or binary yes/no facts.
- Examples: 
  * "Who is the current ICT Minister?"
  * "What is the 2026 GDP forecast?"
  * "Did Bangladesh win the match yesterday?"
  * "When does the Dhaka Metro open?"
  * "How much is the toll fee for the Padma Bridge?"
- Constraint: You MUST answer in exactly 1 to 2 sentences. Be direct. Do not add unnecessary fluff.

CATEGORY 2: Complex Explanatory Queries
- Definition: Queries asking for summaries, reasons, processes, multifaceted events, or broad overviews.
- Examples: 
  * "Why did the inflation rate increase this month?"
  * "Explain the current political situation in Dhaka."
  * "What are the details of the new trade agreement?"
  * "Summarize the impact of the recent floods in Sylhet."
  * "How does the newly proposed education curriculum work?"
- Constraint: You MUST answer in 5 to 10 sentences. Provide a comprehensive summary of the context.


SOURCES:
{context}

USER QUERY: {query}
RESPONSE:"""
    
    # Run LLM inference
    loop = asyncio.get_event_loop()
    ans_res = await loop.run_in_executor(None, lambda: model([{"role": "user", "content": prompt}]))
    
    output = ans_res.content.strip()
    latency = f"{round(time.time() - t_start, 2)}s"

    # 5. Final Update & Cache
    GLOBAL_CACHE[query_hash] = {"output": output, "sources": [r['url'] for r in results], "latency": latency}
    with open(CACHE_FILE, 'w') as f: json.dump(GLOBAL_CACHE, f)
    
    yield output, sources_str, latency

def clear_cache_logic():
    global GLOBAL_CACHE
    GLOBAL_CACHE = {}  
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'w') as f:
            json.dump({}, f) 
    return "🧹 Cache Wiped!", "...", "0.0s"

# ============================== GRADIO UI ===================================

with gr.Blocks(title="Bangladesh RAG Explorer") as demo:
    gr.Markdown("# 🇧🇩 Ultra-Fast RAG (2026 Edition)")
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(label="Query", placeholder="What's happening in Dhaka today?")
            
            # --- NEW: Checkbox and Buttons Row ---
            with gr.Row():
                submit_btn = gr.Button("Search", variant="primary")
                advanced_toggle = gr.Checkbox(label="🔬 Advanced Search (Slower, better quality)", value=False)
                
        with gr.Column(scale=1):
            latency_output = gr.Label(label="Latency")
            clear_btn = gr.Button("Clear Cache", variant="stop", size='sm')

    answer_output = gr.Markdown(label="AI Answer")
    
    with gr.Accordion("View Sources", open=False):
        sources_output = gr.Textbox(label="URLs", interactive=False)

    # Wire up the button click
    submit_btn.click(
        fn=predict,
        inputs=[query_input, advanced_toggle],
        outputs=[answer_output, sources_output, latency_output],
        queue=True 
    )
    
    # Wire up the "Enter" key on the textbox to do the same thing
    query_input.submit(
        fn=predict,
        inputs=[query_input, advanced_toggle],
        outputs=[answer_output, sources_output, latency_output],
        queue=True 
    )
    
    clear_btn.click(
        fn=clear_cache_logic,
        inputs=[],
        outputs=[answer_output, sources_output, latency_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)