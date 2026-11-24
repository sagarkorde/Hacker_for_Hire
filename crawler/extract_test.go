package main

import (
	"net/url"
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
