"""Blinded annotation kit for the 400-listing sample.

  python annotation_kit.py make         key.csv (kept from annotators), one sheet per annotator with its own
                                        order, and annotate.html (offline browser tool)
  python annotation_kit.py adjudicate   after both sheets are filled: adjudication.csv listing every
                                        disagreement for the two annotators to resolve together

Annotators see only an item id, the title and the description: no vendor, category, sampling
stratum, price or detector output. Files go to results/annotation/.
"""
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "annotation"
LABELS = ["HAAS_SERVICE", "HAAS_TOOL", "HAAS_GUIDE", "OTHER_FRAUD", "ACCOUNTS_MEDIA", "OTHER"]


def make():
    DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "results" / "annotation_sample.csv"
    s = pd.read_csv(src, keep_default_na=False)
    rng = np.random.default_rng(2026)
    ids = rng.permutation(len(s))
    s.insert(0, "id", [f"L{i + 1:03d}" for i in ids])
    key = s.drop(columns=[c for c in ["annotator_A", "annotator_B", "adjudicated"] if c in s.columns])
    key.sort_values("id").to_csv(DIR / "key.csv", index=False)
    items = key.sort_values("id")[["id", "Item", "Item Description"]].rename(columns={"Item": "title", "Item Description": "description"})
    for name, seed in [("A", 11), ("B", 22)]:
        sheet = items.sample(frac=1, random_state=seed).assign(label="", note="")
        sheet.to_csv(DIR / f"sheet_annotator_{name}.csv", index=False)
    data = json.dumps(items.to_dict("records"), ensure_ascii=False).replace("</", "<\\/")
    (DIR / "annotate.html").write_text(TOOL.replace("__DATA__", data).replace("__LABELS__", json.dumps(LABELS)))
    print("wrote", DIR)


def adjudicate():
    a = pd.read_csv(DIR / "sheet_annotator_A.csv", keep_default_na=False).set_index("id")
    b = pd.read_csv(DIR / "sheet_annotator_B.csv", keep_default_na=False).set_index("id")
    for n, x in [("A", a), ("B", b)]:
        bad = set(x["label"].str.strip().str.upper()) - set(LABELS)
        if bad:
            raise SystemExit(f"sheet {n}: unknown or empty labels {sorted(bad)[:5]}")
    j = a[["title", "description"]].join(a["label"].rename("label_A")).join(b["label"].rename("label_B"))
    j["adjudicated"] = np.where(j["label_A"] == j["label_B"], j["label_A"], "")
    j.reset_index().to_csv(DIR / "adjudication.csv", index=False)
    print(f"{int((j['label_A'] != j['label_B']).sum())} disagreements to resolve in adjudication.csv")


TOOL = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HaaS annotation</title>
<style>
:root{--ink:#1d1d1f;--muted:#6b6b70;--line:#d9d9de;--bg:#fafafa;--acc:#2f5d8a}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0}
header b{font-size:15px}select,button,input{font:inherit;padding:5px 9px;border:1px solid var(--line);border-radius:6px;background:#fff}
button{cursor:pointer}button:hover{border-color:var(--acc)}.bar{flex:1;min-width:120px;height:6px;background:#eee;border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc)}main{max-width:820px;margin:18px auto;padding:0 16px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px}.id{color:var(--muted);font-size:13px}
h2{font-size:18px;margin:4px 0 10px}.desc{white-space:pre-wrap;color:#333;max-height:42vh;overflow:auto}
.labels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:16px}.labels button{text-align:left;padding:9px 10px}
.labels button.on{background:var(--acc);color:#fff;border-color:var(--acc)}.labels small{display:block;opacity:.75;font-size:12px}
.nav{display:flex;gap:8px;align-items:center;margin-top:14px}.nav input{flex:1}.muted{color:var(--muted);font-size:13px}
details{margin-top:18px}table{border-collapse:collapse;font-size:13px}td{border-bottom:1px solid var(--line);padding:4px 8px;vertical-align:top}
</style></head><body>
<header><b>HaaS annotation</b>
<label>Annotator <select id="who"><option value="">choose</option><option>A</option><option>B</option></select></label>
<span id="count" class="muted"></span><div class="bar"><i id="prog" style="width:0"></i></div>
<button id="next0">Next unlabelled</button><button id="export">Export CSV</button>
<label class="muted"><input type="file" id="imp" accept=".csv" style="display:none"><span style="text-decoration:underline;cursor:pointer" onclick="document.getElementById('imp').click()">Import CSV</span></label>
</header>
<main><div class="card" id="card"><p class="muted">Choose your annotator letter to start. Each annotator gets a different order. Work alone and do not look at the other sheet.</p></div>
<details><summary>Labels and rules (from docs/annotation_guideline.md)</summary>
<table>
<tr><td>1 HAAS_SERVICE</td><td>The vendor carries out an intrusion or attack on a target the buyer chooses: hack a named website or account, DDoS or booter/stresser access, database dump to order, doxing.</td></tr>
<tr><td>2 HAAS_TOOL</td><td>Software or infrastructure the buyer runs to attack: RAT, botnet kit, keylogger, crypter, exploit kit, phishing page, cracking software.</td></tr>
<tr><td>3 HAAS_GUIDE</td><td>Instructions for hacking or intrusion.</td></tr>
<tr><td>4 OTHER_FRAUD</td><td>Fraud goods or guides without intrusion: carding, cash-out, cards, fake documents, scam methods.</td></tr>
<tr><td>5 ACCOUNTS_MEDIA</td><td>Ready-made accounts, subscriptions, gift codes, pay-TV or IPTV, likes and followers, even if the accounts were cracked.</td></tr>
<tr><td>6 OTHER</td><td>Anything else, or too little text to decide (add ? in the note).</td></tr>
</table><p class="muted">If two labels fit, take the higher one in this list. Judge the offer in the text only. Keys: 1-6 label, arrows move, N next unlabelled.</p></details>
</main>
<script>
const ITEMS = __DATA__;
const LABELS = __LABELS__;
const HINT = ["attack on buyer's target","attack software/infra","hacking instructions","fraud, no intrusion","accounts, pay-TV, likes","other / unclear"];
let who = "", order = [], pos = 0, ann = {};
function rnd(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function shuffle(n,seed){const r=rnd(seed),a=[...Array(n).keys()];for(let i=n-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
function key(){return "haas-annot-"+who}
function save(){try{localStorage.setItem(key(),JSON.stringify({ann,pos}))}catch(e){}}
function load(){try{const s=JSON.parse(localStorage.getItem(key())||"null");if(s){ann=s.ann||{};pos=s.pos||0}else{ann={};pos=0}}catch(e){ann={};pos=0}}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function render(){
  if(!who){return}
  const it=ITEMS[order[pos]], a=ann[it.id]||{label:"",note:""}, done=Object.values(ann).filter(x=>x.label).length;
  document.getElementById("count").textContent=done+" / "+ITEMS.length+" labelled";
  document.getElementById("prog").style.width=(100*done/ITEMS.length)+"%";
  document.getElementById("card").innerHTML=
   `<div class="id">${esc(it.id)} · item ${pos+1} of ${ITEMS.length}</div><h2>${esc(it.title)}</h2><div class="desc">${esc(it.description||"(no description)")}</div>
    <div class="labels">${LABELS.map((l,i)=>`<button data-l="${l}" class="${a.label===l?"on":""}">${i+1} ${l}<small>${HINT[i]}</small></button>`).join("")}</div>
    <div class="nav"><button id="prev">&larr; Prev</button><input id="note" placeholder="note (optional)" value="${esc(a.note)}"><button id="nxt">Next &rarr;</button></div>`;
  document.querySelectorAll(".labels button").forEach(b=>b.onclick=()=>setLabel(b.dataset.l));
  document.getElementById("prev").onclick=()=>move(-1);document.getElementById("nxt").onclick=()=>move(1);
  document.getElementById("note").oninput=e=>{const id=it.id;ann[id]=ann[id]||{label:"",note:""};ann[id].note=e.target.value;save()};
}
function setLabel(l){const id=ITEMS[order[pos]].id;ann[id]=ann[id]||{label:"",note:""};ann[id].label=l;save();move(1)}
function move(d){pos=Math.max(0,Math.min(ITEMS.length-1,pos+d));save();render()}
function nextUnlabelled(){for(let k=0;k<ITEMS.length;k++){const p=(pos+k)%ITEMS.length;if(!(ann[ITEMS[order[p]].id]||{}).label){pos=p;break}}render()}
document.getElementById("who").onchange=e=>{who=e.target.value;if(!who)return;order=shuffle(ITEMS.length,who==="A"?11:22);load();render()};
document.getElementById("next0").onclick=nextUnlabelled;
document.addEventListener("keydown",e=>{if(!who||e.target.id==="note")return;const n=parseInt(e.key);if(n>=1&&n<=6){setLabel(LABELS[n-1])}else if(e.key==="ArrowRight")move(1);else if(e.key==="ArrowLeft")move(-1);else if(e.key==="n"||e.key==="N")nextUnlabelled()});
function csvq(s){s=String(s??"");return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
document.getElementById("export").onclick=()=>{
  if(!who){alert("Choose an annotator first");return}
  const rows=[["id","title","description","label","note"]].concat(order.map(i=>{const it=ITEMS[i],a=ann[it.id]||{};return[it.id,it.title,it.description,a.label||"",a.note||""]}));
  const blob=new Blob([rows.map(r=>r.map(csvq).join(",")).join("\n")+"\n"],{type:"text/csv"});
  const u=URL.createObjectURL(blob),l=document.createElement("a");l.href=u;l.download="sheet_annotator_"+who+".csv";l.click();URL.revokeObjectURL(u)};
function parseCSV(t){const out=[];let row=[],f="",q=false;for(let i=0;i<t.length;i++){const c=t[i];if(q){if(c==='"'&&t[i+1]==='"'){f+='"';i++}else if(c==='"')q=false;else f+=c}else if(c==='"')q=true;else if(c===","){row.push(f);f=""}else if(c==="\n"){row.push(f);out.push(row);row=[];f=""}else if(c!=="\r")f+=c}if(f||row.length){row.push(f);out.push(row)}return out}
document.getElementById("imp").onchange=e=>{const f=e.target.files[0];if(!f||!who){alert("Choose an annotator first");return}
  f.text().then(t=>{const r=parseCSV(t),h=r[0],iid=h.indexOf("id"),il=h.indexOf("label"),inote=h.indexOf("note");r.slice(1).forEach(x=>{if(x[iid])ann[x[iid]]={label:(x[il]||"").trim().toUpperCase(),note:x[inote]||""}});save();render()})};
</script></body></html>
"""

if __name__ == "__main__":
    {"make": make, "adjudicate": adjudicate}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
