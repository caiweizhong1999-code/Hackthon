"""
make_demo_video.py — records a narrated walkthrough of the Streamlit app as an MP4.

Steps
  1. Synthesize one narration clip per scene with macOS `say`.
  2. Start the app (streamlit, headless, port 8600) and drive it with Playwright in Chrome,
     recording the browser. Each scene lasts at least as long as its narration clip.
  3. Mix the clips at the recorded scene offsets and mux with the video via ffmpeg -> demo_video.mp4

Requirements (in the project venv): pip install playwright imageio-ffmpeg edge-tts ; Google Chrome installed.
Run from the project root:  .venv/bin/python demo_video/make_demo_video.py
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
BUILD = OUT_DIR / "build"
CSV = ROOT / "synthetic_clinical.csv"
PORT = 8600
URL = f"http://localhost:{PORT}"
W, H = 1440, 900
# Neural voice via edge-tts (needs internet; only the narration text is sent). Set DEMO_VOICE=say:Samantha for offline macOS `say`.
VOICE = os.environ.get("DEMO_VOICE", "en-US-AvaNeural")
RATE = os.environ.get("DEMO_RATE", "+0%")            # edge-tts rate, e.g. "+5%"; for say: words per minute
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
AUDIO_EXT = "aiff" if VOICE.startswith("say:") else "mp3"

# ---------------------------------------------------------------------------
# Narration (one entry per scene; captions show the same text)
# ---------------------------------------------------------------------------
NARRATION = {
    "intro": "This is the AI-Powered Clinical Data Explorer. It lets clinical researchers select patient cohorts, "
             "run survival analysis, and predict patient risk, just by asking questions in plain English. "
             "No code required.",
    "upload": "We start by uploading a de-identified dataset: about twelve hundred synthetic oncology patients. "
              "The app validates the file and automatically detects the survival time and event columns.",
    "ask": "Now we simply ask a question. Show overall survival for patients older than sixty-five, split by treatment.",
    "parsed": "The question becomes a structured, validated query: filters, outcome, grouping, and plot type. "
              "With an API key, Claude does the parsing. Offline, as here, a rule-based fallback takes over.",
    "km": "Instantly, we get Kaplan-Meier curves for each treatment arm, with confidence bands, censoring marks, "
          "a number-at-risk table, and the log-rank p-value.",
    "summary": "The summary tab gives a publication-style table: median survival with confidence intervals, "
               "survival at twelve and twenty-four months, hazard ratios, and pairwise comparisons.",
    "table1": "And Table 1 describes the baseline characteristics of the selected cohort, compared across groups.",
    "edit": "Every query is transparent. Researchers can review the parsed JSON, edit it, and re-run. "
            "Here we restrict to stage four patients and compare by PD-L1 status.",
    "train": "Next, risk prediction. One click trains an XGBoost Cox survival model. "
             "Discrimination is reported as the C-index on a held-out validation split.",
    "patient": "Let's enter a new patient: seventy-eight years old, with stage four disease. "
               "The model estimates the two-year mortality risk and assigns a low, medium, or high risk tier.",
    "shap": "SHAP explains the prediction, showing which features pushed this patient's risk up or down.",
    "global": "Across the whole cohort, SHAP ranks the most important drivers, and Kaplan-Meier curves "
              "confirm that the predicted risk tiers separate patients clearly.",
    "outro": "From a plain-English question to cohort selection, survival analysis, and explainable risk prediction, "
             "in seconds. Built with Claude, XGBoost, SHAP, and Streamlit. For research use only.",
}

# ---------------------------------------------------------------------------
# On-page overlays: caption bar, fake cursor, title cards
# ---------------------------------------------------------------------------
SETUP_JS = r"""
() => {
  if (document.getElementById('demo-cap')) return;
  const css = document.createElement('style');
  css.textContent = `
    #demo-cap{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);max-width:1080px;z-index:2147483646;
      background:rgba(17,20,52,.88);color:#fff;font:500 19px/1.45 -apple-system,Helvetica,Arial,sans-serif;
      padding:12px 22px;border-radius:12px;box-shadow:0 8px 28px rgba(0,0,0,.25);text-align:center;
      transition:opacity .35s;opacity:0;pointer-events:none}
    #demo-cursor{position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
      background:rgba(91,63,214,.35);border:2px solid #5b3fd6;z-index:2147483647;pointer-events:none;
      transition:transform .7s cubic-bezier(.22,.8,.3,1)}
    #demo-cursor.click{animation:demo-pulse .45s ease-out}
    @keyframes demo-pulse{0%{box-shadow:0 0 0 0 rgba(91,63,214,.6)}100%{box-shadow:0 0 0 22px rgba(91,63,214,0)}}
    #demo-card{position:fixed;inset:0;z-index:2147483645;display:none;align-items:center;justify-content:center;
      background:linear-gradient(135deg,#171a47 0%,#2b2f7a 55%,#5b3fd6 100%);color:#fff;
      font-family:-apple-system,Helvetica,Arial,sans-serif;transition:opacity .6s}
    #demo-card .inner{max-width:1000px;padding:40px}
    #demo-card h1{font-size:54px;margin:0 0 14px;font-weight:750;letter-spacing:-.5px}
    #demo-card p{font-size:23px;opacity:.88;margin:0 0 30px;line-height:1.5}
    #demo-card ul{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:1fr 1fr;gap:12px 28px}
    #demo-card li{font-size:19px;background:rgba(255,255,255,.1);border-radius:10px;padding:12px 16px}
    #demo-card .foot{margin-top:30px;font-size:15px;opacity:.65}`;
  document.head.appendChild(css);
  const cap = document.createElement('div'); cap.id = 'demo-cap'; document.body.appendChild(cap);
  const cur = document.createElement('div'); cur.id = 'demo-cursor';
  cur.style.transform = 'translate(720px,450px)'; document.body.appendChild(cur);
  const card = document.createElement('div'); card.id = 'demo-card'; document.body.appendChild(card);
}
"""


def setup(page: Page) -> None:
    page.evaluate(SETUP_JS)


def caption(page: Page, text: str | None) -> None:
    setup(page)
    page.evaluate("""t => { const c = document.getElementById('demo-cap');
        if (t) { c.textContent = t; c.style.opacity = 1; } else { c.style.opacity = 0; } }""", text)


def card(page: Page, html: str | None) -> None:
    setup(page)
    page.evaluate("""h => { const c = document.getElementById('demo-card');
        if (h) { c.innerHTML = h; c.style.display = 'flex'; c.style.opacity = 1; }
        else { c.style.opacity = 0; setTimeout(() => c.style.display = 'none', 600); } }""", html)


def move_to(page: Page, loc, pause: float = 0.75) -> None:
    loc.scroll_into_view_if_needed()
    box = loc.bounding_box()
    if box:
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.evaluate("([x,y]) => document.getElementById('demo-cursor').style.transform = `translate(${x}px,${y}px)`",
                      [x, y])
    time.sleep(pause)


def click(page: Page, loc, pause: float = 0.75) -> None:
    move_to(page, loc, pause)
    page.evaluate("""() => { const c = document.getElementById('demo-cursor');
        c.classList.remove('click'); void c.offsetWidth; c.classList.add('click'); }""")
    loc.click()


def scroll_to(page: Page, loc, block: str = "start") -> None:
    loc.evaluate(f"el => el.scrollIntoView({{behavior: 'smooth', block: '{block}'}})")
    time.sleep(1.1)


def idle(page: Page) -> None:
    """Wait until Streamlit has finished rerunning (no 'Running' status widget)."""
    time.sleep(0.4)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=60_000)
    except Exception:
        pass
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# Scene runner
# ---------------------------------------------------------------------------
class Timeline:
    def __init__(self, page: Page, durations: dict[str, float], t0: float):
        self.page, self.dur = page, durations
        self.t0 = t0                         # wall-clock time when video recording started
        self.events: list[tuple[str, float]] = []

    def now(self) -> float:
        return time.monotonic() - self.t0

    def scene(self, name: str, action=None, min_tail: float = 0.6) -> None:
        start = self.now()
        self.events.append((name, start))
        caption(self.page, NARRATION[name])
        if action:
            action()
        remaining = self.dur[name] + min_tail - (self.now() - start)
        if remaining > 0:
            time.sleep(remaining)
        self.page.screenshot(path=str(BUILD / f"scene_{len(self.events):02d}_{name}.png"))


# ---------------------------------------------------------------------------
# 1. Narration audio
# ---------------------------------------------------------------------------
def synth_narration() -> dict[str, float]:
    durations = {}
    for name, text in NARRATION.items():
        out = BUILD / f"{name}.{AUDIO_EXT}"
        if VOICE.startswith("say:"):
            subprocess.run(["say", "-v", VOICE[4:], "-r", RATE.strip("+%") or "178", "-o", str(out), text], check=True)
        else:
            edge = Path(sys.executable).with_name("edge-tts")
            subprocess.run([str(edge), "--voice", VOICE, f"--rate={RATE}", "--text", text,
                            "--write-media", str(out)], check=True, capture_output=True)
        info = subprocess.run(["afinfo", str(out)], capture_output=True, text=True).stdout
        sec = next(float(l.split()[2]) for l in info.splitlines() if "estimated duration" in l)
        durations[name] = sec
    return durations


# ---------------------------------------------------------------------------
# 2. App server
# ---------------------------------------------------------------------------
def start_app() -> subprocess.Popen:
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
           "--server.port", str(PORT), "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer",
           "--theme.base", "light", "--theme.primaryColor", "#5b3fd6",
           "--theme.backgroundColor", "#f3f4fb", "--theme.secondaryBackgroundColor", "#ffffff",
           "--theme.textColor", "#1d2143"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "ANTHROPIC_API_KEY": ""}, start_new_session=True)
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1).read() == b"ok":
                return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Streamlit did not start")


# ---------------------------------------------------------------------------
# 3. The walkthrough
# ---------------------------------------------------------------------------
INTRO_HTML = """<div class="inner"><h1>AI-Powered Clinical Data Explorer</h1>
<p>Ask questions about your patients in plain English, and get cohorts, survival analysis
and explainable risk predictions, without writing code.</p>
<ul><li>💬 Natural-language queries (Claude)</li><li>🧬 Cohort selection &amp; validation</li>
<li>📈 Kaplan-Meier &amp; log-rank</li><li>🎯 XGBoost risk + SHAP</li></ul>
<div class="foot">Hackathon demo · synthetic data</div></div>"""

OUTRO_HTML = """<div class="inner"><h1>Question → Cohort → Survival → Risk</h1>
<p>Validated natural-language queries, publication-style survival tables, and explainable ML,
in one interactive dashboard.</p>
<ul><li>Claude + rule-based fallback</li><li>Kaplan-Meier · log-rank · Table 1</li>
<li>XGBoost Cox · C-index · risk tiers</li><li>SHAP global &amp; per-patient</li></ul>
<div class="foot">Built with Claude, XGBoost, SHAP and Streamlit · For research use only, not for clinical decisions</div></div>"""

EDITED_QUERY = {"filters": [{"column": "stage", "op": "==", "value": "IV"}],
                "outcome": "overall_survival", "group_by": "pdl1", "plot_type": "km_curve"}


def walkthrough(page: Page, durations: dict[str, float], t_rec: float) -> tuple[list[tuple[str, float]], float]:
    page.goto(URL)
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
    idle(page)
    setup(page)
    tl = Timeline(page, durations, t_rec)
    card(page, INTRO_HTML)
    video_start = tl.now()                  # trim the blank loading frames before this
    sidebar = page.locator('[data-testid="stSidebar"]')
    main = page.locator('[data-testid="stMain"]')

    tl.scene("intro", lambda: time.sleep(0.5), min_tail=1.0)

    def do_upload():
        card(page, None)
        time.sleep(0.8)
        move_to(page, sidebar.locator('[data-testid="stFileUploaderDropzone"]'))
        page.locator('input[type="file"]').set_input_files(str(CSV))
        page.get_by_text("1,248 patients", exact=False).first.wait_for(timeout=60_000)
        idle(page)
        move_to(page, page.get_by_text("1,248 patients", exact=False).first, pause=1.0)
    tl.scene("upload", do_upload)

    q_box = page.get_by_placeholder("e.g. Show overall survival by treatment group for patients older than 65")

    def do_ask():
        click(page, q_box)
        q_box.press_sequentially("Show overall survival for patients older than 65, split by treatment", delay=38)
        time.sleep(0.5)
        click(page, page.get_by_role("button", name="Generate"))
        page.get_by_text("patients selected", exact=False).first.wait_for(timeout=60_000)
        idle(page)
    tl.scene("ask", do_ask, min_tail=0.3)

    def do_parsed():
        scroll_to(page, main.locator('[data-testid="stHorizontalBlock"]').nth(1), block="center")
        move_to(page, page.get_by_text("Parsed Query", exact=False).first, pause=1.2)
    tl.scene("parsed", do_parsed)

    def do_km():
        scroll_to(page, page.get_by_role("tab", name="Plot"), block="start")
        move_to(page, main.locator('[data-testid="stImage"]').first, pause=0.8)
    tl.scene("km", do_km)

    def do_summary():
        click(page, page.get_by_role("tab", name="Summary"))
        idle(page)
        time.sleep(0.6)
        move_to(page, page.get_by_text("Log-rank p", exact=False).first, pause=1.5)
        scroll_to(page, main.locator('[data-testid="stDataFrame"]').first, block="center")
    tl.scene("summary", do_summary)

    def do_table1():
        scroll_to(page, page.get_by_role("tab", name="Table 1"), block="start")
        click(page, page.get_by_role("tab", name="Table 1"))
        idle(page)
        time.sleep(0.4)
        heading = page.get_by_text("Baseline characteristics", exact=False).first
        scroll_to(page, heading, block="start")
        move_to(page, main.locator('[data-testid="stDataFrame"]:visible').first, pause=0.8)
    tl.scene("table1", do_table1)

    def do_edit():
        editor = page.get_by_label("Edit the query (JSON) and re-run")
        scroll_to(page, editor, block="center")
        click(page, editor)
        editor.fill(json.dumps(EDITED_QUERY, indent=2))
        editor.press("Meta+Enter")
        idle(page)
        time.sleep(0.8)
        click(page, page.get_by_role("button", name="Re-run edited query"))
        idle(page)
        click(page, page.get_by_role("tab", name="Plot"))
        idle(page)
        page.get_by_text("grouped by pdl1", exact=False).first.wait_for(timeout=60_000)
        scroll_to(page, page.get_by_role("tab", name="Plot"), block="start")
        move_to(page, main.locator('[data-testid="stImage"]').first, pause=0.8)
    tl.scene("edit", do_edit)

    def do_train():
        click(page, sidebar.get_by_text("Risk Prediction", exact=True))
        idle(page)
        page.get_by_role("button", name="Train model").wait_for(timeout=30_000)
        time.sleep(0.6)
        click(page, page.get_by_role("button", name="Train model"))
        page.get_by_text("C-index (validation)", exact=False).first.wait_for(timeout=120_000)
        idle(page)
        move_to(page, page.get_by_text("C-index (validation)", exact=False).first, pause=1.2)
    tl.scene("train", do_train)

    def do_patient():
        age = page.get_by_role("spinbutton", name="Age")
        scroll_to(page, page.get_by_text("Input Patient Features", exact=False).first, block="start")
        click(page, age)
        age.fill("78")
        time.sleep(0.4)
        stage = page.locator('[data-testid="stSelectbox"]').filter(has_text="Stage").first
        click(page, stage)
        time.sleep(0.5)
        page.get_by_role("option", name="IV", exact=True).click()
        time.sleep(0.6)
        click(page, page.get_by_role("button", name="Predict Risk"))
        idle(page)
        page.get_by_text("risk within", exact=False).first.wait_for(timeout=60_000)
        move_to(page, page.get_by_text("risk within", exact=False).first, pause=1.2)
    tl.scene("patient", do_patient)

    def do_shap():
        find = """() => { const t = [...document.querySelectorAll('.panel-t')].find(e => e.textContent.includes('SHAP Explanation'));
            const img = t && t.closest('[data-testid="stVerticalBlock"]').querySelector('img');
            return img && img.complete && img.naturalWidth > 0 ? img : null; }"""
        img = page.wait_for_function(find, timeout=60_000).as_element()
        img.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
        time.sleep(1.1)
        box = img.bounding_box()
        page.evaluate("([x,y]) => document.getElementById('demo-cursor').style.transform = `translate(${x}px,${y}px)`",
                      [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
        time.sleep(1.0)
    tl.scene("shap", do_shap)

    def do_global():
        exp = page.get_by_text("Model performance & global drivers", exact=False).first
        scroll_to(page, exp, block="start")
        click(page, exp)
        idle(page)
        time.sleep(0.8)
        btn = page.get_by_role("button", name="Plot KM by predicted risk group")
        scroll_to(page, main.locator('[data-testid="stImage"]').nth(-1), block="center")
        time.sleep(1.0)
        click(page, btn)
        idle(page)
        page.get_by_text("Risk groups are tertiles", exact=False).first.wait_for(timeout=60_000)
        scroll_to(page, page.get_by_text("Risk groups are tertiles", exact=False).first, block="end")
    tl.scene("global", do_global)

    def do_outro():
        caption(page, None)
        card(page, OUTRO_HTML)
    tl.scene("outro", do_outro, min_tail=1.5)
    return tl.events, video_start


# ---------------------------------------------------------------------------
# 4. Mux
# ---------------------------------------------------------------------------
def mux(video: Path, events: list[tuple[str, float]], trim: float, out: Path) -> None:
    inputs, filters, labels = ["-ss", f"{trim:.3f}", "-i", str(video)], [], []
    for i, (name, start) in enumerate(events, start=1):
        inputs += ["-i", str(BUILD / f"{name}.{AUDIO_EXT}")]
        delay = max(0, int((start - trim + 0.25) * 1000))
        filters.append(f"[{i}:a]aresample=48000,adelay={delay}|{delay}[a{i}]")
        labels.append(f"[a{i}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0[aout]")
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    print("Synthesizing narration…")
    durations = synth_narration()
    print(f"  total narration {sum(durations.values()):.0f}s")

    print("Starting app…")
    proc = start_app()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                      record_video_dir=str(BUILD), record_video_size={"width": W, "height": H})
            page = ctx.new_page()
            t_rec = time.monotonic()   # recording starts with the page
            print("Recording walkthrough…")
            events, trim = walkthrough(page, durations, t_rec)
            page.screenshot(path=str(BUILD / "last_frame.png"))
            video_path = Path(page.video.path())
            ctx.close()
            browser.close()
    finally:
        os.killpg(proc.pid, signal.SIGTERM)

    (BUILD / "timeline.json").write_text(json.dumps({"trim": trim, "events": events}, indent=2))
    out = OUT_DIR / "demo_video.mp4"
    print("Muxing…")
    mux(video_path, events, trim, out)
    print("Done:", out)


if __name__ == "__main__":
    main()
