// Command server serves a static market site with Tor-like response latency.
//
// Each response is delayed by a log-normally distributed time (median and sigma
// configurable), a configurable share of responses fail transiently with HTTP 503,
// and paths that are not part of the site return 404 (dead links). The site is
// loaded from the JSON produced by analysis/build_site.py.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type site struct {
	Pages map[string]string `json:"pages"`
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8088", "listen address")
	path := flag.String("site", "data/site/site.json", "site JSON")
	median := flag.Float64("median", 1.0, "median latency in seconds")
	sigma := flag.Float64("sigma", 0.5, "log-normal sigma")
	p503 := flag.Float64("p503", 0.02, "probability of a transient 503")
	seed := flag.Int64("seed", 1, "random seed")
	flag.Parse()

	raw, err := os.ReadFile(*path)
	if err != nil {
		log.Fatal(err)
	}
	var s site
	if err := json.Unmarshal(raw, &s); err != nil {
		log.Fatal(err)
	}

	var mu sync.Mutex
	rng := rand.New(rand.NewSource(*seed))
	var served, failed int64
	mu50 := math.Log(*median)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		z := rng.NormFloat64()
		u := rng.Float64()
		mu.Unlock()
		time.Sleep(time.Duration(math.Exp(mu50+*sigma*z) * float64(time.Second)))
		atomic.AddInt64(&served, 1)
		if u < *p503 {
			atomic.AddInt64(&failed, 1)
			http.Error(w, "circuit failed", http.StatusServiceUnavailable)
			return
		}
		page, ok := s.Pages[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(page))
	})
	srv := &http.Server{Addr: *addr, ReadHeaderTimeout: 30 * time.Second}
	log.Printf("serving %d pages on %s (median=%.2fs sigma=%.2f p503=%.3f seed=%d)", len(s.Pages), *addr, *median, *sigma, *p503, *seed)
	log.Fatal(srv.ListenAndServe())
}
