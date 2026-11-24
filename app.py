import json
import re
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# ----------------------------- CONFIG ---------------------------------------
APP_TITLE = "Website to Pitch"
MODEL = "gemini-2.5-flash"          # model not found aaye to koi available Flash model likho
TIMEOUT = 15
MAX_CHARS = 3500
FREE_SINGLE_LIMIT = 5               # per browser session (API key bachane ke liye)
FREE_BULK_ROWS = 10
BANNED = [
    "delve", "game-changer", "game changer", "synergy", "leverage", "revolutionize",
    "cutting-edge", "unlock", "supercharge", "streamline", "seamless", "robust",
    "best-in-class", "world-class", "hope this finds you well", "reaching out",
    "touch base", "circle back", "holistic", "skyrocket",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

st.set_page_config(page_title=APP_TITLE, page_icon="✉️", layout="centered")


# ----------------------------- SCRAPING -------------------------------------
def norm_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    return u if re.match(r"^https?://", u, re.I) else "https://" + u


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def fetch(url: str) -> str:
    err = None
    for u in [url, url.replace("https://", "http://", 1)] if url.startswith("https://") else [url]:
        try:
            r = requests.get(u, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            if "html" not in r.headers.get("Content-Type", "").lower():
                raise ValueError("Not an HTML page")
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        except Exception as e:
            err = e
    raise err


def parse(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "iframe"]):
        t.decompose()
    title = clean(soup.title.get_text()) if soup.title else ""
    meta = ""
    for a in ({"name": "description"}, {"property": "og:description"}):
        m = soup.find("meta", attrs=a)
        if m and m.get("content"):
            meta = clean(m["content"])
            break
    heads = []
    for h in soup.find_all(["h1", "h2"]):
        t = clean(h.get_text())
        if 8 <= len(t) <= 160 and t not in heads:
            heads.append(t)
        if len(heads) >= 8:
            break
    paras, seen = [], set()
    for p in soup.find_all(["p", "li"]):
        t = clean(p.get_text())
        if len(t) >= 45 and t.lower() not in seen:
            seen.add(t.lower())
            paras.append(t)
        if len(paras) >= 25:
            break
    return {"title": title, "meta": meta, "heads": heads, "paras": paras}


def about_link(html: str, base: str):
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        label = (a.get_text() or "").lower() + " " + a["href"].lower()
        if re.search(r"\b(about|about-us|company|our-story|who-we-are)\b", label):
            link = urljoin(base, a["href"])
            if urlparse(link).netloc == urlparse(base).netloc:
                return link
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def scrape(raw_url: str) -> dict:
    url = norm_url(raw_url)
    if not url:
        return {"ok": False, "text": "", "error": "Empty URL"}
    try:
        html = fetch(url)
        d = parse(html)
        if sum(len(p) for p in d["paras"]) < 400:
            link = about_link(html, url)
            if link:
                try:
                    e = parse(fetch(link))
                    d["paras"] += e["paras"]
                    d["heads"] += [h for h in e["heads"] if h not in d["heads"]]
                except Exception:
                    pass
        parts = []
        if d["title"]:
            parts.append(f"TITLE: {d['title']}")
        if d["meta"]:
            parts.append(f"META DESCRIPTION: {d['meta']}")
        if d["heads"]:
            parts.append("HEADINGS: " + " | ".join(d["heads"]))
        if d["paras"]:
            parts.append("CONTENT: " + " ".join(d["paras"]))
        text = "\n".join(parts)[:MAX_CHARS]
        if len(text) < 60:
            return {"ok": False, "text": "", "error": "Not enough readable content (site JS-heavy ya blocked ho sakti hai)"}
        return {"ok": True, "text": text, "error": ""}
    except requests.exceptions.Timeout:
        return {"ok": False, "text": "", "error": "Website ne time par jawab nahi diya"}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        return {"ok": False, "text": "", "error": f"HTTP error {code}"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "text": "", "error": "Website khuli nahi (domain galat ya down)"}
    except Exception as e:
        return {"ok": False, "text": "", "error": f"{type(e).__name__}"}


# ----------------------------- LLM ------------------------------------------
def system_prompt(language: str) -> str:
    return f"""You are an expert B2B cold-outreach copywriter. You write short, warm, human messages \
that sound like a thoughtful person typed them in two minutes, never like a template.

Rules:
- Write in {language}.
- Pitch is EXACTLY 4 lines, returned as 4 separate strings:
  Line 1: Start with "Hi <first name>," then ONE genuine, specific observation from the website text.
  Line 2: A concrete benefit or idea the sender's service could give THIS company, tied to that observation.
  Line 3: Make the idea tangible. No made-up statistics, no fake case studies, no invented facts.
  Line 4: A soft, low-pressure question as CTA. No meeting-booking pressure.
- Each line max ~25 words. Plain conversational words. No emojis, no exclamation marks.
- Use only facts present in the website text.
- Never use: {", ".join(BANNED)}.
- No sign-off, no subject line.

Also return "website_summary" (2 factual sentences in English: what the company does and who it serves)
and "observation_points" (3 short bullet strings: notable things found on the site).

Return ONLY JSON:
{{"website_summary": "...", "observation_points": ["...","...","..."], "pitch_lines": ["l1","l2","l3","l4"]}}"""


def first_name(n: str) -> str:
    n = (n or "").strip()
    return n.split()[0].strip(",.") if n and n.lower() != "nan" else "there"


def bad(lines) -> str:
    if not isinstance(lines, list) or len(lines) != 4 or any(not str(l).strip() for l in lines):
        return "pitch must be exactly 4 non-empty lines"
    j = " ".join(lines).lower()
    for w in BANNED:
        if w in j:
            return f"banned phrase: {w}"
    return ""


def parse_json(t: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (t or "").strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def generate(client, company, founder, service, site_text, language, tries=4) -> dict:
    prompt = (
        f"COMPANY: {company}\nFOUNDER FIRST NAME TO USE: {first_name(founder)}\n\n"
        f"SENDER'S SERVICE:\n{service}\n\nWEBSITE TEXT:\n{site_text}\n"
    )
    feedback, last = "", None
    for i in range(1, tries + 1):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=prompt + feedback,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt(language),
                    temperature=0.8,
                    response_mime_type="application/json",
                ),
            )
            d = parse_json(r.text)
            lines = [clean(str(x)) for x in d.get("pitch_lines", [])]
            p = bad(lines)
            if p:
                feedback = f"\n\nPrevious answer rejected: {p}. Fix it."
                last = p
                continue
            return {
                "summary": clean(str(d.get("website_summary", ""))),
                "points": [clean(str(x)) for x in d.get("observation_points", [])][:3],
                "pitch": "\n".join(lines),
            }
        except Exception as e:
            last = str(e)[:150]
            time.sleep(min(30, 2 ** i * 2) if any(k in last.lower() for k in ("429", "quota", "503", "overload")) else 2)
    raise RuntimeError(f"AI generation failed: {last}")


@st.cache_resource
def get_client():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        st.error("GEMINI_API_KEY secret set nahi hai. Streamlit Cloud > App settings > Secrets mein add karo.")
        st.stop()
    return genai.Client(api_key=key)


# ----------------------------- UI -------------------------------------------
st.title("✉️ Website to Pitch")
st.caption("Kisi bhi company ki website daalo. AI use padhkar founder ke liye personalized 4-line cold message likhega.")

if "used" not in st.session_state:
    st.session_state.used = 0

with st.expander("Apni service likho (ye pitch ka base banti hai)", expanded=True):
    service = st.text_area(
        "Aap kya offer karte ho?",
        value="I help B2B SaaS founders get qualified demo requests through done-for-you LinkedIn content.",
        height=100,
    )
    language = st.selectbox("Pitch language", ["English", "Hinglish", "Hindi"])

tab1, tab2 = st.tabs(["Single website", f"Bulk CSV (free: {FREE_BULK_ROWS} rows)"])

# ---- Single ----
with tab1:
    c1, c2 = st.columns(2)
    company = c1.text_input("Company name", placeholder="Acme Inc")
    founder = c2.text_input("Founder name", placeholder="Rahul Sharma")
    url = st.text_input("Website URL", placeholder="acme.com")

    if st.button("Generate pitch", type="primary", use_container_width=True):
        if not url.strip() or not service.strip():
            st.warning("Website URL aur service dono bharo.")
        elif st.session_state.used >= FREE_SINGLE_LIMIT:
            st.info(f"Free limit ({FREE_SINGLE_LIMIT}) poori ho gayi. Zyada ke liye bulk/paid plan lo.")
        else:
            with st.spinner("Website padh raha hu..."):
                s = scrape(url)
            if not s["ok"]:
                st.error(f"Website scrape nahi hui: {s['error']}")
            else:
                try:
                    with st.spinner("Pitch likh raha hu..."):
                        res = generate(get_client(), company or "the company", founder, service, s["text"], language)
                    st.session_state.used += 1
                    st.subheader("Custom pitch")
                    st.code(res["pitch"], language=None)
                    st.subheader("Website summary")
                    st.write(res["summary"])
                    if res["points"]:
                        st.subheader("Observations")
                        for p in res["points"]:
                            st.markdown(f"- {p}")
                    st.caption("Bhejne se pehle ek baar khud padh lo aur facts verify karo.")
                except Exception as e:
                    st.error(str(e))

# ---- Bulk ----
with tab2:
    st.write("CSV columns: `company_name`, `website_url`, `founder_name`")
    up = st.file_uploader("CSV upload karo", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up, dtype=str, keep_default_na=False)
            df.columns = [c.strip() for c in df.columns]
            miss = [c for c in ["company_name", "website_url", "founder_name"] if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                if len(df) > FREE_BULK_ROWS:
                    st.info(f"Free plan mein sirf pehli {FREE_BULK_ROWS} rows process hongi.")
                    df = df.head(FREE_BULK_ROWS)
                st.dataframe(df, use_container_width=True)
                if st.button("Run bulk", type="primary", use_container_width=True):
                    client = get_client()
                    df["website_summary"], df["custom_pitch"], df["status"] = "", "", ""
                    bar = st.progress(0.0)
                    for i, row in df.iterrows():
                        s = scrape(row["website_url"])
                        if not s["ok"]:
                            df.at[i, "website_summary"] = f"SCRAPE_FAILED: {s['error']}"
                            df.at[i, "status"] = "SCRAPE_FAILED"
                        else:
                            try:
                                r = generate(client, row["company_name"], row["founder_name"], service, s["text"], language)
                                df.at[i, "website_summary"] = r["summary"]
                                df.at[i, "custom_pitch"] = r["pitch"]
                                df.at[i, "status"] = "OK"
                            except Exception as e:
                                df.at[i, "status"] = f"LLM_FAILED: {str(e)[:80]}"
                        bar.progress((i + 1) / len(df))
                        time.sleep(1)
                    st.success("Done!")
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "Download personalized_leads_output.csv",
                        df.to_csv(index=False).encode("utf-8-sig"),
                        "personalized_leads_output.csv",
                        "text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"CSV read nahi hui: {e}")

st.divider()
st.caption("Made by <your name> | <your LinkedIn link>")
