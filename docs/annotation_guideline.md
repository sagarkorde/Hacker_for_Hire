# Annotation guideline: hacking-as-a-service (HaaS) listings

Purpose: build a human-labelled test set to replace market category labels (pending validation, Section 8).

Label each listing using only its title and description. Hide the `Category` and `stratum` columns while labelling.

| Label | Use when the listing offers... | Examples |
|---|---|---|
| `HAAS_SERVICE` | to carry out an intrusion or attack for the buyer | hack a website/account on request, DDoS on a target, database dump to order, doxing |
| `HAAS_TOOL` | software or infrastructure for attacks | RAT, botnet setup, keylogger, crypter, exploit kit, phishing page, booter access |
| `HAAS_GUIDE` | instructions for hacking or intrusion | hacking tutorials, carding-by-hacking guides |
| `OTHER_FRAUD` | fraud goods not involving intrusion | carding, cash-out, stolen cards, counterfeit documents |
| `ACCOUNTS_MEDIA` | resold accounts, subscriptions, pay-TV, likes/followers | Netflix accounts, CCcam, Facebook likes |
| `OTHER` | anything else | e-books unrelated to hacking, VPN, drugs |

Rules
1. Judge the offer, not the vendor or the category.
2. If a listing fits two labels, choose the first one in the table.
3. If the text is too short to decide, use `OTHER` and add `?` in a comment.
4. The binary HaaS label for evaluation is `HAAS_SERVICE`, `HAAS_TOOL` or `HAAS_GUIDE` (report also the stricter `HAAS_SERVICE` only).

Process: two annotators label independently; compute a chance-corrected agreement statistic on the 400 items; resolve disagreements by discussion into `adjudicated`; then re-score all models against `adjudicated` with the same vendor-disjoint protocol.
