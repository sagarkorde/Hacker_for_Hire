"""Python baseline crawlers with the same behaviour as the Go crawler.

  --mode asyncio : one event loop, aiohttp, a semaphore bounds concurrency
  --mode threads : requests + ThreadPoolExecutor with N worker threads

Both stay on the seed host, follow links up to --depth, retry 5xx/network errors
--retries times, extract visible text with the same rules as crawler/extract.go,
and print the same JSON summary.
"""
import argparse
import asyncio
import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlsplit, urlunsplit

RE_HREF = re.compile(r"""href\s*=\s*["']([^"'#\s]+)""", re.I)
RE_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
RE_SKIP = re.compile(r"<(script|style|noscript|head|template)\b[^>]*>.*?</(script|style|noscript|head|template)>", re.I | re.S)
RE_TAG = re.compile(r"<[^>]*>", re.S)
RE_SPACE = re.compile(r"\s+")


def extract(page: str):
    m = RE_TITLE.search(page)
    title = html.unescape(m.group(1)).strip() if m else ""
    hrefs = [html.unescape(h) for h in RE_HREF.findall(page)]
    body = RE_TAG.sub(" ", RE_SKIP.sub(" ", page))
    text = RE_SPACE.sub(" ", html.unescape(body)).strip()
    return title, text, hrefs


def same_host(base: str, hrefs):
    host = urlsplit(base).netloc
    out = []
    for h in hrefs:
        u = urlsplit(urljoin(base, h))
        if u.scheme in ("http", "https") and u.netloc == host:
            out.append(urlunsplit((u.scheme, u.netloc, u.path, u.query, "")))
    return out


class Stats:
    def __init__(self):
        self.attempted = self.ok = self.not_found = self.failed = self.retries = self.links = 0
        self.lock = threading.Lock()


def summary(mode, args, st, elapsed):
    return {"crawler": f"python-{mode}", "concurrency": args.c, "max_pages": args.max,
            "attempted": st.attempted, "ok": st.ok, "not_found": st.not_found, "failed_final": st.failed,
            "retries": st.retries, "links_discovered": st.links, "elapsed_s": elapsed,
            "pages_per_min": st.attempted / elapsed * 60, "ok_per_min": st.ok / elapsed * 60,
            "link_quality_ratio": (st.ok / st.attempted) if st.attempted else 0.0}


# ---------------------------------------------------------------- asyncio
async def run_asyncio(args):
    import aiohttp
    st, seen, sem = Stats(), {args.seed}, asyncio.Semaphore(args.c)
    tasks = set()
    conn = aiohttp.TCPConnector(limit=args.c, limit_per_host=args.c)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:

        async def fetch(url):
            for attempt in range(args.retries + 1):
                if attempt:
                    st.retries += 1
                try:
                    async with sess.get(url) as r:
                        body = await r.text(errors="replace")
                        if r.status >= 500 and attempt < args.retries:
                            continue
                        return r.status, body
                except Exception:
                    if attempt == args.retries:
                        return 0, ""
            return 0, ""

        async def visit(url, depth):
            async with sem:
                if st.attempted >= args.max:
                    return
                st.attempted += 1
                status, body = await fetch(url)
            if status == 200:
                st.ok += 1
                _, _, hrefs = extract(body)
                if depth < args.depth:
                    for link in same_host(url, hrefs):
                        if link not in seen:
                            seen.add(link)
                            st.links += 1
                            t = asyncio.create_task(visit(link, depth + 1))
                            tasks.add(t)
                            t.add_done_callback(tasks.discard)
            elif status == 404:
                st.not_found += 1
            else:
                st.failed += 1

        t0 = time.perf_counter()
        first = asyncio.create_task(visit(args.seed, 0))
        tasks.add(first)
        first.add_done_callback(tasks.discard)
        while tasks:
            await asyncio.gather(*list(tasks))
        return summary("asyncio", args, st, time.perf_counter() - t0)


# ---------------------------------------------------------------- threads
def run_threads(args):
    import requests
    st, seen, lock = Stats(), {args.seed}, threading.Lock()
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.c, pool_maxsize=args.c)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    pending = threading.Semaphore(0)
    outstanding = [0]
    pool = ThreadPoolExecutor(max_workers=args.c)

    def fetch(url):
        for attempt in range(args.retries + 1):
            if attempt:
                with lock:
                    st.retries += 1
            try:
                r = sess.get(url, timeout=args.timeout)
                if r.status_code >= 500 and attempt < args.retries:
                    continue
                return r.status_code, r.text
            except Exception:
                if attempt == args.retries:
                    return 0, ""
        return 0, ""

    def submit(url, depth):
        with lock:
            outstanding[0] += 1
        pool.submit(visit, url, depth)

    def visit(url, depth):
        try:
            with lock:
                if st.attempted >= args.max:
                    return
                st.attempted += 1
            status, body = fetch(url)
            if status == 200:
                _, _, hrefs = extract(body)
                new = []
                with lock:
                    st.ok += 1
                    if depth < args.depth:
                        for link in same_host(url, hrefs):
                            if link not in seen:
                                seen.add(link)
                                st.links += 1
                                new.append(link)
                for link in new:
                    submit(link, depth + 1)
            else:
                with lock:
                    if status == 404:
                        st.not_found += 1
                    else:
                        st.failed += 1
        finally:
            with lock:
                outstanding[0] -= 1
                done = outstanding[0] == 0
            if done:
                pending.release()

    t0 = time.perf_counter()
    submit(args.seed, 0)
    pending.acquire()
    elapsed = time.perf_counter() - t0
    pool.shutdown(wait=True, cancel_futures=True)
    return summary("threads", args, st, elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["asyncio", "threads"], required=True)
    ap.add_argument("--seed", default="http://127.0.0.1:8088/")
    ap.add_argument("--c", type=int, default=10)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()
    res = asyncio.run(run_asyncio(args)) if args.mode == "asyncio" else run_threads(args)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
