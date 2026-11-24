"""Rule-based hacking-as-a-service (HaaS) detector and entity extractor.

The term lists and the decision threshold were fixed before any evaluation run
(they follow the original pipeline design: a curated dictionary of primary and
secondary terms, corroborated by contact/payment entities) and are not tuned on
the evaluation data. Section 7 of the paper varies the threshold as a sensitivity
analysis only.
"""
import re

PRIMARY = [
    "hack", "hacks", "hacking", "hacker", "hackers", "hacked", "ddos", "d-dos", "denial of service", "stresser",
    "booter", "botnet", "exploit", "exploits", "0day", "zero-day", "zero day", "rat", "remote access trojan",
    "remote administration tool", "keylogger", "crypter", "fud", "malware", "ransomware", "trojan", "spyware",
    "phishing", "scam page", "brute force", "bruteforce", "sql injection", "sqli", "xss", "web shell", "webshell",
    "rdp", "cracking", "account takeover", "password recovery", "recover password", "dox", "doxing", "doxxing",
    "spoofing", "sms bomber", "email bomber",
]
SECONDARY = [
    "service", "services", "hire", "for hire", "custom", "on demand", "we will", "i will", "we can", "i can",
    "guarantee", "guaranteed", "target", "victim", "any website", "any account", "your target", "upon request",
    "per hour", "per day", "results", "proof", "contact me", "message me", "pm me", "within 24",
]
ENTITY_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "jabber": re.compile(r"\b(jabber|xmpp)\b", re.I),
    "icq": re.compile(r"\bicq\b", re.I),
    "skype": re.compile(r"\bskype\b", re.I),
    "telegram": re.compile(r"\b(telegram|t\.me/)", re.I),
    "pgp": re.compile(r"\bpgp\b", re.I),
    "btc_address": re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"),
}
THRESHOLD = 3  # fixed a priori


def _compile(terms):
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![a-z0-9])")


RE_PRIMARY = _compile(PRIMARY)
RE_SECONDARY = _compile(SECONDARY)


def entities(text: str) -> dict:
    return {k: len(p.findall(text)) for k, p in ENTITY_PATTERNS.items()}


def score(text: str, use_secondary=True, use_entities=True):
    t = text.lower()
    p = len(RE_PRIMARY.findall(t))
    s = len(RE_SECONDARY.findall(t)) if use_secondary else 0
    e = int(any(v > 0 for v in entities(text).values())) if use_entities else 0
    return p, s, e, 2 * min(p, 3) + min(s, 3) + e


def predict(text: str, threshold=THRESHOLD, use_secondary=True, use_entities=True) -> tuple:
    """Return (label, score). A listing is HaaS if it has a primary term and score >= threshold."""
    p, s, e, sc = score(text, use_secondary, use_entities)
    if not use_secondary and not use_entities:
        return int(p >= 1), sc
    return int(p >= 1 and sc >= threshold), sc
