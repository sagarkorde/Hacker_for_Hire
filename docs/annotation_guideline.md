# Annotation guideline: hacking-as-a-service (HaaS) listings

Purpose: build a human-labelled test set to replace market category labels (pending validation, Section 8).

Label each listing using only its title and description. Use `results/annotation/annotate.html` (open it in a
browser; choose your annotator letter) or your own sheet `results/annotation/sheet_annotator_<A|B>.csv`. Both show
only an item id, the title and the description. Do not open `key.csv` or `results/annotation_sample.csv`, which hold the vendor, the market
category and the sampling stratum, or `provisional_labels_claude.csv`, and do not look at the other annotator's sheet
until both are finished.

| Label | Use when the listing offers... | Examples |
|---|---|---|
| `HAAS_SERVICE` | to carry out an intrusion or attack on a target the buyer chooses | hack a website/account on request, DDoS or booter/stresser access, database dump to order, doxing |
| `HAAS_TOOL` | software or infrastructure the buyer runs to attack | RAT, botnet kit, keylogger, crypter, exploit kit, phishing page, cracking software |
| `HAAS_GUIDE` | instructions for hacking or intrusion | hacking tutorials, carding-by-hacking guides |
| `OTHER_FRAUD` | fraud goods not involving intrusion | carding, cash-out, stolen cards, counterfeit documents |
| `ACCOUNTS_MEDIA` | resold accounts, subscriptions, pay-TV, likes/followers | Netflix accounts, CCcam, Facebook likes |
| `OTHER` | anything else | e-books unrelated to hacking, VPN, drugs |

Decision rules for the hard cases
- Who picks the target? If the vendor attacks a target named by the buyer, it is `HAAS_SERVICE`. If the buyer gets
  something to run themselves, it is `HAAS_TOOL`. If the buyer gets instructions, it is `HAAS_GUIDE`.
- Ready-made goods are not services, even if they were obtained by hacking: cracked or resold accounts, gift codes and
  subscriptions are `ACCOUNTS_MEDIA`; card data, dumps for cash-out and fake documents are `OTHER_FRAUD`.
- A database dump or account takeover "to order" (the buyer names the site or account) is `HAAS_SERVICE`.
- Carding or scam guides that involve no intrusion are `OTHER_FRAUD`; guides to hacking accounts, sites or networks are
  `HAAS_GUIDE`.

Rules
1. Judge the offer, not the vendor or the category.
2. If a listing fits two labels, choose the first one in the table.
3. If the text is too short to decide, use `OTHER` and add `?` in a comment.
4. The binary HaaS label for evaluation is `HAAS_SERVICE`, `HAAS_TOOL` or `HAAS_GUIDE` (report also the stricter `HAAS_SERVICE` only).

Process
1. `python analysis/annotation_kit.py make` (already done) creates the two sheets, the tool and `key.csv`.
2. Each annotator labels all 400 items alone and saves `sheet_annotator_A.csv` / `sheet_annotator_B.csv` in
   `results/annotation/` (the tool's "Export CSV" button writes this file).
3. `python analysis/annotation_kit.py adjudicate` writes `adjudication.csv`; the two annotators fill the empty
   `adjudicated` cells together.
4. `python analysis/score_annotation.py` reports Cohen's kappa and design-weighted precision, recall and F1 with
   vendor-cluster bootstrap intervals, and the manuscript build picks the results up automatically.
