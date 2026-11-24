package main

import (
	"html"
	"net/url"
	"regexp"
	"strings"
)

// Document is the visible content of one HTML page.
type Document struct {
	Title string
	Text  string
	Hrefs []string
}

var (
	reHref  = regexp.MustCompile(`(?i)href\s*=\s*["']([^"'#\s]+)`)
	reTitle = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	reSkip  = regexp.MustCompile(`(?is)<(script|style|noscript|head|template)\b[^>]*>.*?</(script|style|noscript|head|template)>`)
	reTag   = regexp.MustCompile(`(?s)<[^>]*>`)
	reSpace = regexp.MustCompile(`\s+`)
)

// Extract returns the title, visible text and raw link targets of a page.
// Non-visible elements (script, style, noscript, head, template) are removed
// before tags are stripped; character entities are decoded.
func Extract(page string) Document {
	d := Document{}
	if m := reTitle.FindStringSubmatch(page); m != nil {
		d.Title = strings.TrimSpace(html.UnescapeString(m[1]))
	}
	for _, m := range reHref.FindAllStringSubmatch(page, -1) {
		d.Hrefs = append(d.Hrefs, html.UnescapeString(m[1]))
	}
	body := reSkip.ReplaceAllString(page, " ")
	body = reTag.ReplaceAllString(body, " ")
	d.Text = strings.TrimSpace(reSpace.ReplaceAllString(html.UnescapeString(body), " "))
	return d
}

// ExtractBody applies a per-market selector to the raw page: the capture groups of the first match of re
// (for example the listing title and description) are joined with " . ", stripped of tags and decoded.
// It returns "" when the selector does not match, so listing text is never mixed with page furniture.
func ExtractBody(page string, re *regexp.Regexp) string {
	m := re.FindStringSubmatch(page)
	if m == nil {
		return ""
	}
	parts := make([]string, 0, len(m)-1)
	for _, g := range m[1:] {
		t := reTag.ReplaceAllString(reSkip.ReplaceAllString(g, " "), " ")
		t = strings.TrimSpace(reSpace.ReplaceAllString(html.UnescapeString(t), " "))
		if t != "" {
			parts = append(parts, t)
		}
	}
	return strings.Join(parts, " . ")
}

// SameHostLinks resolves hrefs against base and keeps http(s) links on base's host.
func SameHostLinks(base *url.URL, hrefs []string) []string {
	out := make([]string, 0, len(hrefs))
	for _, h := range hrefs {
		u, err := base.Parse(h)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host != base.Host {
			continue
		}
		u.Fragment = ""
		out = append(out, u.String())
	}
	return out
}
