package main

import (
	"net/url"
	"regexp"
	"testing"
)

func TestExtract(t *testing.T) {
	page := `<html><head><title>A &amp; B</title><script>var x=1;</script></head>
<body><style>p{}</style><p>Hello <b>world</b></p><a href="/x#top">x</a><a href='http://other/y'>y</a></body></html>`
	d := Extract(page)
	if d.Title != "A & B" {
		t.Fatalf("title %q", d.Title)
	}
	if d.Text != "Hello world x y" {
		t.Fatalf("text %q", d.Text)
	}
	base, _ := url.Parse("http://host/a/b")
	links := SameHostLinks(base, d.Hrefs)
	if len(links) != 1 || links[0] != "http://host/x" {
		t.Fatalf("links %v", links)
	}
}

func TestExtractBody(t *testing.T) {
	page := `<div class="nav">Home</div><h1>Hack &amp; more</h1><div class='price'>1 BTC</div><div class='desc'><p>Custom <b>job</b></p></div><div>Category: Services/Hacking</div>`
	re := regexp.MustCompile(`(?s)<h1>(.*?)</h1>.*?<div class='desc'>(.*?)</div>`)
	if got := ExtractBody(page, re); got != "Hack & more . Custom job" {
		t.Fatalf("body %q", got)
	}
	if got := ExtractBody("<p>no listing</p>", re); got != "" {
		t.Fatalf("non-listing page gave %q", got)
	}
}
