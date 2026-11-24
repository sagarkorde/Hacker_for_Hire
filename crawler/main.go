// Command hfhcrawl is a concurrent crawler for Tor onion services.
//
// All traffic can be routed through a local Tor SOCKS5 proxy (-proxy
// socks5://127.0.0.1:9050). Concurrency is bounded by a buffered channel used as a
// semaphore; every discovered URL is handled by its own goroutine that waits for a
// slot. The crawler stays on the seed host, follows links up to -depth, retries
// transient failures, extracts visible text and writes one JSON record per page.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"sync/atomic"
	"time"
)

type record struct {
	URL      string  `json:"url"`
	Depth    int     `json:"depth"`
	Status   int     `json:"status"`
	Attempts int     `json:"attempts"`
	Bytes    int     `json:"bytes"`
	FetchMS  float64 `json:"fetch_ms"`
	Title    string  `json:"title,omitempty"`
	Text     string  `json:"text,omitempty"`
	Body     string  `json:"body,omitempty"`
	Links    int     `json:"links"`
	Err      string  `json:"error,omitempty"`
}

type summary struct {
	Crawler         string  `json:"crawler"`
	Concurrency     int     `json:"concurrency"`
	MaxPages        int64   `json:"max_pages"`
	Attempted       int64   `json:"attempted"`
	OK              int64   `json:"ok"`
	NotFound        int64   `json:"not_found"`
	FailedFinal     int64   `json:"failed_final"`
	Retries         int64   `json:"retries"`
	LinksDiscovered int64   `json:"links_discovered"`
	ElapsedS        float64 `json:"elapsed_s"`
	PagesPerMin     float64 `json:"pages_per_min"`
	OKPerMin        float64 `json:"ok_per_min"`
	LinkQuality     float64 `json:"link_quality_ratio"`
}

type crawler struct {
	client   *http.Client
	host     string
	sem      chan struct{}
	wg       sync.WaitGroup
	mu       sync.Mutex
	seen     map[string]bool
	maxPages int64
	maxDepth int
	retries  int
	out      chan record
	bodyRe   *regexp.Regexp

	attempted, ok, notFound, failed, retried, links int64
}

func newClient(proxy string, timeout time.Duration, conc int) (*http.Client, error) {
	tr := &http.Transport{MaxIdleConns: conc * 2, MaxIdleConnsPerHost: conc, MaxConnsPerHost: conc,
		IdleConnTimeout: 90 * time.Second}
	if proxy != "" {
		pu, err := url.Parse(proxy) // e.g. socks5://127.0.0.1:9050 (Tor)
		if err != nil {
			return nil, err
		}
		tr.Proxy = http.ProxyURL(pu)
	}
	return &http.Client{Transport: tr, Timeout: timeout}, nil
}

// fetch downloads one URL, retrying 5xx responses and network errors.
func (c *crawler) fetch(u string) (int, []byte, int, error) {
	var lastErr error
	for attempt := 1; attempt <= c.retries+1; attempt++ {
		if attempt > 1 {
			atomic.AddInt64(&c.retried, 1)
		}
		resp, err := c.client.Get(u)
		if err != nil {
			lastErr = err
			continue
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("http %d", resp.StatusCode)
			if attempt == c.retries+1 {
				return resp.StatusCode, body, attempt, lastErr
			}
			continue
		}
		return resp.StatusCode, body, attempt, nil
	}
	return 0, nil, c.retries + 1, lastErr
}

func (c *crawler) claim(u string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.seen[u] {
		return false
	}
	c.seen[u] = true
	return true
}

func (c *crawler) visit(u string, depth int) {
	defer c.wg.Done()
	c.sem <- struct{}{}
	if atomic.AddInt64(&c.attempted, 1) > c.maxPages {
		atomic.AddInt64(&c.attempted, -1)
		<-c.sem
		return
	}
	t0 := time.Now()
	status, body, attempts, err := c.fetch(u)
	<-c.sem
	rec := record{URL: u, Depth: depth, Status: status, Attempts: attempts, Bytes: len(body),
		FetchMS: float64(time.Since(t0).Microseconds()) / 1000}
	switch {
	case err != nil:
		atomic.AddInt64(&c.failed, 1)
		rec.Err = err.Error()
	case status == http.StatusNotFound:
		atomic.AddInt64(&c.notFound, 1)
	case status == http.StatusOK:
		atomic.AddInt64(&c.ok, 1)
		doc := Extract(string(body))
		rec.Title, rec.Text = doc.Title, doc.Text
		if c.bodyRe != nil {
			rec.Body = ExtractBody(string(body), c.bodyRe)
		}
		if depth < c.maxDepth {
			base, _ := url.Parse(u)
			for _, l := range SameHostLinks(base, doc.Hrefs) {
				if c.claim(l) {
					atomic.AddInt64(&c.links, 1)
					c.wg.Add(1)
					go c.visit(l, depth+1)
				}
			}
		}
		rec.Links = len(doc.Hrefs)
	}
	if c.out != nil {
		c.out <- rec
	}
}

func main() {
	seed := flag.String("seed", "http://127.0.0.1:8088/", "seed URL (an .onion address when using Tor)")
	conc := flag.Int("c", 10, "maximum concurrent requests")
	maxPages := flag.Int64("max", 1000, "maximum pages to request")
	maxDepth := flag.Int("depth", 6, "maximum link depth")
	proxy := flag.String("proxy", "", "proxy URL, e.g. socks5://127.0.0.1:9050 for Tor")
	timeout := flag.Duration("timeout", 30*time.Second, "per-request timeout")
	retries := flag.Int("retries", 1, "retries for 5xx responses and network errors")
	outDir := flag.String("out", "", "directory for pages.jsonl (empty: do not store pages)")
	bodySel := flag.String("body", "", "optional per-market regular expression whose capture groups hold the listing body (stored as \"body\")")
	flag.Parse()

	client, err := newClient(*proxy, *timeout, *conc)
	if err != nil {
		log.Fatal(err)
	}
	su, err := url.Parse(*seed)
	if err != nil {
		log.Fatal(err)
	}
	c := &crawler{client: client, host: su.Host, sem: make(chan struct{}, *conc), seen: map[string]bool{},
		maxPages: *maxPages, maxDepth: *maxDepth, retries: *retries}
	if *bodySel != "" {
		c.bodyRe = regexp.MustCompile(*bodySel)
	}

	var writerDone chan struct{}
	if *outDir != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			log.Fatal(err)
		}
		f, err := os.Create(filepath.Join(*outDir, "pages.jsonl"))
		if err != nil {
			log.Fatal(err)
		}
		c.out = make(chan record, 1024)
		writerDone = make(chan struct{})
		go func() {
			w := bufio.NewWriter(f)
			enc := json.NewEncoder(w)
			for r := range c.out {
				_ = enc.Encode(r)
			}
			w.Flush()
			f.Close()
			close(writerDone)
		}()
	}

	start := time.Now()
	c.claim(*seed)
	c.wg.Add(1)
	go c.visit(*seed, 0)
	c.wg.Wait()
	elapsed := time.Since(start).Seconds()
	if c.out != nil {
		close(c.out)
		<-writerDone
	}
	s := summary{Crawler: "go", Concurrency: *conc, MaxPages: *maxPages, Attempted: c.attempted, OK: c.ok,
		NotFound: c.notFound, FailedFinal: c.failed, Retries: c.retried, LinksDiscovered: c.links,
		ElapsedS: elapsed, PagesPerMin: float64(c.attempted) / elapsed * 60, OKPerMin: float64(c.ok) / elapsed * 60}
	if c.attempted > 0 {
		s.LinkQuality = float64(c.ok) / float64(c.attempted)
	}
	_ = json.NewEncoder(os.Stdout).Encode(s)
}
