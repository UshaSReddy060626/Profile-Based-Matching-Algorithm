import io
import json
import os
from datetime import datetime

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

try:
    from fpdf import FPDF
except ImportError:          # the app still runs; only the PDF button needs fpdf2
    FPDF = None

from matching import MatchEngine, learn_weights, DEFAULT_W, MBTI_TYPES, CITIES, FEATURE_NAMES

st.set_page_config(page_title="Profile Match Finder", layout="wide", initial_sidebar_state="expanded")

# Palette: dark green #0A3323, moss #839958, beige #F7F4D5, rosy brown #D3968C, midnight green #105666
DARK, MOSS, BEIGE, ROSY, MID, TINT = "#0A3323", "#839958", "#F7F4D5", "#D3968C", "#105666", "#B7B089"

st.markdown("""<style>
@font-face{font-family:"Playfair Display";src:url("app/static/fonts/PlayfairDisplay.ttf") format("truetype");font-weight:400 900}
@font-face{font-family:"Inter";src:url("app/static/fonts/Inter.ttf") format("truetype");font-weight:100 900}
:root{--dark:#0A3323;--moss:#839958;--beige:#F7F4D5;--rosy:#D3968C;--mid:#105666;--card:#FCFBEA;--line:#D9D5AC;
      --serif:"Playfair Display",Georgia,"Times New Roman",serif;--sans:"Inter","Segoe UI",Arial,sans-serif}
html,body,.stApp,.stApp p,.stApp label,.stApp li,.stApp button,.stApp input,.stApp textarea,.stApp td,.stApp th,
.stApp [data-baseweb="select"] *:not([data-testid="stIconMaterial"]),.stApp [data-baseweb="tab"]{font-family:var(--sans)}
.stApp{color:var(--dark);background-color:var(--beige);
  background-image:linear-gradient(rgba(10,51,35,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(10,51,35,.06) 1px,transparent 1px);
  background-size:34px 34px}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1160px;padding-top:2.2rem;padding-bottom:5rem}

/* sidebar */
section[data-testid="stSidebar"]{background:var(--dark)}
section[data-testid="stSidebar"] *{color:var(--beige)}
section[data-testid="stSidebar"] h2{font-family:var(--serif);font-weight:600;font-size:1.3rem;border-bottom:1px solid #2c5a47;padding-bottom:.5rem}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#F7F4D5!important;color:var(--dark)!important;border-radius:4px}
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:var(--dark)!important}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{background:var(--rosy);border:0;color:var(--dark)!important;font-weight:600;border-radius:4px;
  transition:transform .15s ease,box-shadow .15s ease}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *{color:var(--dark)!important}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover{transform:translateY(-2px);box-shadow:0 6px 14px rgba(0,0,0,.35)}

/* headings */
.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;color:var(--mid);font-weight:600}
h1.page-title{font-family:var(--serif);font-weight:700;font-size:2.5rem;color:var(--dark);margin:.2rem 0 .3rem;padding:0;letter-spacing:-.01em}
.lede{color:#3f5a4c;font-size:.98rem;max-width:740px;margin-bottom:1rem;line-height:1.55}
h3.section{font-family:var(--serif);font-weight:600;font-size:1.4rem;color:var(--dark);margin:1.4rem 0 .6rem}
.sub{color:#4b6557;font-size:.88rem;margin:-.3rem 0 .8rem}

/* tabs */
button[data-baseweb="tab"]{font-size:.98rem;font-weight:500;color:#3f5a4c;padding:.6rem 1.1rem}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--dark);font-weight:700}
div[data-baseweb="tab-highlight"]{background:var(--mid);height:3px}
div[data-baseweb="tab-border"]{background:var(--line)}

/* stat blocks */
.stat{border-top:3px solid var(--moss);padding-top:.5rem;background:rgba(252,251,234,.6)}
.stat .lab{text-transform:uppercase;letter-spacing:.1em;font-size:.68rem;color:#4b6557}
.stat .val{font-family:var(--serif);font-size:2rem;color:var(--mid);line-height:1.15;font-weight:600}
.stat .note{font-size:.78rem;color:#4b6557}
.kpi{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--mid);border-radius:4px;padding:.8rem 1rem;
  transition:transform .18s ease,box-shadow .18s ease}
.kpi:hover{transform:translateY(-3px);box-shadow:0 10px 22px rgba(10,51,35,.14)}
.kpi .lab{text-transform:uppercase;letter-spacing:.1em;font-size:.68rem;color:#4b6557}
.kpi .val{font-family:var(--serif);font-size:1.7rem;color:var(--dark);font-weight:600;line-height:1.2}
.kpi .note{font-size:.8rem;color:var(--mid);font-weight:600}

/* match cards */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid var(--line)!important;border-radius:6px;
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-3px);box-shadow:0 12px 26px rgba(10,51,35,.15);border-color:var(--moss)!important}
.score{font-family:var(--serif);font-size:2.6rem;font-weight:700;color:var(--mid);line-height:1}
.meta{color:#4b6557;font-size:.86rem}
.name{font-size:1.1rem;font-weight:700;color:var(--dark)}
.bar-row{display:flex;justify-content:space-between;font-size:.78rem;color:#3f5a4c;margin-top:.45rem;font-weight:500}
.bar{height:7px;background:#E6E2B8;border-radius:4px;overflow:hidden;margin-top:2px}
.bar>div{height:100%;background:var(--moss);border-radius:4px}
.tag{display:inline-block;padding:.3rem .8rem;border-radius:4px;font-size:.82rem;font-weight:600;margin:.6rem 0 .4rem}
.tag.yes{background:var(--moss);color:var(--dark)}
.tag.no{background:var(--rosy);color:var(--dark)}

/* buttons */
div[data-testid="stMain"] [data-testid="stBaseButton-primary"]{background:var(--mid);border:0;border-radius:4px;color:var(--beige);
  transition:transform .15s ease,box-shadow .15s ease,background .15s ease}
div[data-testid="stMain"] [data-testid="stBaseButton-primary"] *{color:var(--beige)!important}
div[data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover{background:var(--dark);transform:translateY(-2px);box-shadow:0 6px 14px rgba(10,51,35,.3)}
div[data-testid="stMain"] [data-testid="stBaseButton-secondary"],div[data-testid="stPopover"] button{background:transparent;border:1px solid var(--rosy);
  border-radius:4px;color:var(--dark);transition:transform .15s ease,background .15s ease}
div[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover,div[data-testid="stPopover"] button:hover{background:var(--rosy);transform:translateY(-2px);color:var(--dark)}
div[data-testid="stPopover"] button{border-color:var(--moss);font-size:.85rem}
div[data-testid="stPopover"] button:hover{background:var(--moss)}
[data-testid="stDownloadButton"] button{background:var(--dark);color:var(--beige);border:0;border-radius:4px;transition:transform .15s ease}
[data-testid="stDownloadButton"] button *{color:var(--beige)!important}
[data-testid="stDownloadButton"] button:hover{transform:translateY(-2px);background:var(--mid)}

/* popover content */
table.bk{border-collapse:collapse;font-size:.85rem;min-width:300px}
table.bk th{text-align:left;color:#4b6557;font-weight:600;border-bottom:1px solid var(--line);padding:.3rem .6rem}
table.bk td{padding:.3rem .6rem;border-bottom:1px solid #ECE8C0}
table.bk tr.tot td{font-weight:700;border-top:2px solid var(--dark);border-bottom:0}

/* floating live-weights chip */
.float-chip{position:fixed;right:26px;bottom:26px;z-index:999;background:var(--dark);color:var(--beige);padding:.7rem 1rem;border-radius:8px;
  box-shadow:0 10px 28px rgba(10,51,35,.4);font-size:.82rem;line-height:1.5;border:1px solid #2c5a47}
.float-chip .ttl{text-transform:uppercase;letter-spacing:.12em;font-size:.66rem;color:var(--rosy);font-weight:600}
.float-chip b{font-family:var(--serif);font-size:1rem}
.note{color:#4b6557;font-size:.85rem;margin-top:.8rem}
.insight{background:var(--card);border-left:4px solid var(--rosy);padding:.8rem 1rem;border-radius:4px;color:var(--dark);font-size:.95rem;line-height:1.55}
div[data-testid="stAlert"]{background:var(--card);border:1px solid var(--line);color:var(--dark)}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading matching engine...")
def load_engine():
    return MatchEngine("users.csv", "feedback.csv")


engine = load_engine()
users, feats, saved_weights = engine.users, engine.features, engine.saved_weights

S = st.session_state
S.setdefault("profile", None); S.setdefault("weights", DEFAULT_W.copy()); S.setdefault("X", [])
S.setdefault("y", []); S.setdefault("rated", {}); S.setdefault("uid", None); S.setdefault("w0", DEFAULT_W.copy())


def rate(uid, action):
    if uid in S.rated:
        return
    i = int(users.index[users.user_id == uid][0])
    S.X.append(feats(S.profile)[i]); S.y.append(action); S.rated[uid] = action
    S.weights = learn_weights(np.array(S.X), np.array(S.y))
    st.toast(("Accepted" if action else "Rejected") + ". Weights updated.")


def bar(label, value):
    return (f"<div class='bar-row'><span>{label}</span><span>{100*value:.0f}%</span></div>"
            f"<div class='bar'><div style='width:{100*value:.0f}%'></div></div>")


def breakdown(f, w, total):
    rows = "".join(f"<tr><td>{n}</td><td>{100*v:.0f}%</td><td>{wt:.2f}</td><td>{100*v*wt:.1f}</td></tr>"
                   for n, v, wt in zip(["Text similarity", "MBTI pairing", "Nearby"], f, w))
    return ("<table class='bk'><tr><th>Factor</th><th>Score</th><th>Weight</th><th>Points</th></tr>" + rows +
            f"<tr class='tot'><td colspan='3'>Match score</td><td>{100*total:.0f}%</td></tr></table>")


def kpi(label, value, note=""):
    return f"<div class='kpi'><div class='lab'>{label}</div><div class='val'>{value}</div><div class='note'>{note}</div></div>"


def style(ch, h=300):
    return (ch.properties(height=h, autosize=alt.AutoSizeParams(type="fit", contains="padding"))
              .configure(background="transparent", font="Inter, sans-serif")
              .configure_axis(labelColor=DARK, titleColor=DARK, gridColor="#D9D5AC", domainColor=DARK, tickColor=DARK)
              .configure_legend(labelColor=DARK, titleColor=DARK, orient="bottom")
              .configure_view(stroke=None))


SIGNALS = ["Text similarity", "MBTI pairing", "Nearby (location)"]


def auc_or_none(y, scores):
    """ROC-AUC, or None when the ratings do not yet contain both an Accept and a Reject."""
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, scores))


def live_metrics(X, y, w0):
    """Live evaluation on this session's ratings.

    Each match is scored with the weights the model had BEFORE you rated it (starting weights for the first one,
    then weights learned from the earlier ratings), so the model is never tested on a rating it already learned from.
    """
    X, y, w0 = np.asarray(X, float), np.asarray(y), np.asarray(w0, float)
    ws, pre = [w0], []
    for k in range(len(y)):
        pre.append(float(X[k] @ ws[-1]))
        ws.append(learn_weights(X[:k + 1], y[:k + 1]))
    pre = np.array(pre)
    return dict(ws=np.array(ws), pre=pre,
                auc_start=auc_or_none(y, X @ w0), auc_live=auc_or_none(y, pre),
                signals={n: auc_or_none(y, X[:, k]) for k, n in enumerate(SIGNALS)})


def fmt_auc(v):
    return "n/a" if v is None else f"{v:.2f}"


def pdf_text(t):
    return str(t).encode("latin-1", "replace").decode("latin-1")


def _png(draw, size=(6.6, 2.5)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=170); plt.close(fig); buf.seek(0)
    return buf


def _pdf_table(pdf, header, rows, widths):
    pdf.set_font("Helvetica", "B", 9); pdf.set_fill_color(217, 213, 172); pdf.set_text_color(10, 51, 35)
    for h, wd in zip(header, widths):
        pdf.cell(wd, 7, pdf_text(h), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for v, wd in zip(row, widths):
            pdf.cell(wd, 6.5, pdf_text(v), border=1)
        pdf.ln()
    pdf.ln(3)


def _pdf_h(pdf, text):
    pdf.ln(2); pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(10, 51, 35)
    pdf.cell(0, 8, pdf_text(text), new_x="LMARGIN", new_y="NEXT")


def _pdf_p(pdf, text):
    pdf.set_font("Helvetica", "", 10); pdf.set_text_color(40, 60, 50)
    pdf.multi_cell(0, 5.2, pdf_text(text), new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)


@st.cache_data(show_spinner=False)
def build_pdf(name, city, mbti, w0, X, y, rated_names):
    """Session report as a PDF (bytes)."""
    X, y, w0 = np.array(X, float), np.array(y), np.array(w0, float)
    m = live_metrics(X, y, w0)
    now_w = m["ws"][-1]
    acc, rej = X[y == 1], X[y == 0]

    pdf = FPDF(format="A4"); pdf.set_auto_page_break(True, 15); pdf.set_margins(15, 15, 15); pdf.add_page()
    pdf.set_fill_color(10, 51, 35); pdf.rect(0, 0, 210, 26, "F")
    pdf.set_xy(15, 8); pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(247, 244, 213)
    pdf.cell(0, 9, "Session report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, pdf_text(f"Profile: {name} ({city}, {mbti})   |   Generated {datetime.now():%d %b %Y, %H:%M}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(32)

    _pdf_h(pdf, "Summary")
    _pdf_table(pdf, ["Ratings", "Accepted", "Rejected", "Acceptance rate"],
               [[len(y), len(acc), len(rej), f"{100 * y.mean():.0f}%"]], [40, 40, 40, 50])

    _pdf_h(pdf, "How your weights moved")
    _pdf_table(pdf, ["Factor", "Starting", "Now", "Change"],
               [[n, f"{a:.2f}", f"{b:.2f}", f"{b - a:+.2f}"] for n, a, b in zip(FEATURE_NAMES, w0, now_w)],
               [60, 40, 40, 40])

    def weights_chart(ax):
        xs = np.arange(3)
        ax.bar(xs - 0.2, w0, 0.4, color=TINT, label="Starting"); ax.bar(xs + 0.2, now_w, 0.4, color=MID, label="Now")
        ax.set_xticks(xs); ax.set_xticklabels(FEATURE_NAMES); ax.set_ylim(0, 1); ax.set_ylabel("Weight"); ax.legend(frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    pdf.image(_png(weights_chart), w=150)

    def history_chart(ax):
        for k, (n, c) in enumerate(zip(FEATURE_NAMES, [MID, ROSY, MOSS])):
            ax.plot(range(len(m["ws"])), m["ws"][:, k], marker="o", color=c, label=n)
        ax.set_xlabel("After rating number"); ax.set_ylabel("Weight"); ax.set_ylim(0, 1)
        ax.set_xticks(range(len(m["ws"]))); ax.legend(frameon=False, ncol=3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    pdf.image(_png(history_chart), w=150)

    _pdf_h(pdf, "What your choices say")
    if len(acc) and len(rej):
        _pdf_table(pdf, ["Signal", "Accepted (avg)", "Rejected (avg)"],
                   [[n, f"{100 * a:.0f}%", f"{100 * r:.0f}%"] for n, a, r in zip(SIGNALS, acc.mean(0), rej.mean(0))],
                   [70, 50, 50])
        gap = acc.mean(0) - rej.mean(0); j = int(np.argmax(np.abs(gap)))
        _pdf_p(pdf, f"{SIGNALS[j]} separates your accepted and rejected matches most "
                    f"({100 * acc.mean(0)[j]:.0f}% accepted vs {100 * rej.mean(0)[j]:.0f}% rejected).")
    else:
        _pdf_p(pdf, "Give at least one Accept and one Reject to compare what you liked with what you passed on.")

    _pdf_h(pdf, "Live model performance")
    _pdf_p(pdf, "Each match is scored with the weights the model had before you rated it, so the test only uses "
                "ratings the model had not yet learned from.")
    _pdf_table(pdf, ["Measure", "Value"],
               [["Ranking AUC, starting weights", fmt_auc(m["auc_start"])],
                ["Ranking AUC, learning as you rate", fmt_auc(m["auc_live"])]] +
               [[f"Ranking AUC, {n} alone", fmt_auc(v)] for n, v in m["signals"].items()], [100, 50])
    if len(acc) and len(rej):
        _pdf_p(pdf, f"Average score the model gave before you rated: accepted {100 * m['pre'][y == 1].mean():.0f}%, "
                    f"rejected {100 * m['pre'][y == 0].mean():.0f}%.")
    _pdf_p(pdf, "AUC of 0.5 is chance and 1.0 is a perfect ranking. With few ratings these numbers are noisy.")

    _pdf_h(pdf, "Your ratings")
    _pdf_table(pdf, ["Match", "Your choice", "Text %", "MBTI %", "Nearby %"],
               [[nm, "Accepted" if a else "Rejected", f"{100 * f[0]:.0f}", f"{100 * f[1]:.0f}", f"{100 * f[2]:.0f}"]
                for nm, a, f in zip(rated_names, y, X)], [66, 30, 25, 25, 28])
    return bytes(pdf.output())


def offline_benchmark():
    """Optional: shows the offline benchmark if analysis.py has been run. Silent when the files are missing."""
    mpath = os.path.join("results", "metrics.json")
    if not os.path.exists(mpath):
        return
    M = json.load(open(mpath))
    with st.expander("Offline benchmark on the 120 simulated users (optional)"):
        st.markdown("<div class='sub'>Created by analysis.py from 1,083 simulated Accept/Reject actions. It shows the model can "
                    "learn preferences from feedback; it is not a test on real users.</div>", unsafe_allow_html=True)
        c = st.columns(3)
        c[0].markdown(kpi("Hold-out ROC-AUC", f"{M['auc_default']:.3f} → {M['auc_learned']:.3f}", "default vs learned weights"),
                      unsafe_allow_html=True)
        c[1].markdown(kpi("Top-5 acceptance, round 8", f"{100*M['sim_round8_static']:.0f}% → {100*M['sim_round8_adaptive']:.0f}%",
                          "static vs adaptive"), unsafe_allow_html=True)
        c[2].markdown(kpi("Random baseline", f"{100*M['random']:.0f}%", "acceptance of random picks"), unsafe_allow_html=True)
        cpath = os.path.join("results", "acceptance_curve.csv")
        if os.path.exists(cpath):
            cv = pd.read_csv(cpath).melt("round", var_name="Model", value_name="rate")
            cv["Model"] = cv["Model"].map({"static": "Static weights", "adaptive": "Adaptive weights"})
            base = alt.Chart(cv).encode(
                x=alt.X("round:O", title="Feedback round (5 recommendations each)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 100]), title="Top-5 acceptance (%)"),
                color=alt.Color("Model:N", scale=alt.Scale(domain=["Static weights", "Adaptive weights"], range=[ROSY, MID]), title=None),
                tooltip=["Model", "round", alt.Tooltip("rate:Q", format=".1f")])
            st.altair_chart(style(base.mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3), 280), use_container_width=True)


st.markdown("<div class='eyebrow'>Profile matching</div><h1 class='page-title'>Find people who fit you</h1>"
            "<div class='lede'>Matches are ranked from what you write about yourself, MBTI pairing and distance. "
            "Each Accept or Reject adjusts how much weight each factor carries for you.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Your profile")
    mode = st.radio("Start from", ["Create a new profile", "Use an existing user"])
    if mode == "Use an existing user":
        uid = st.selectbox("User", users.user_id)
        r = users[users.user_id == uid].iloc[0]
        st.write(f"**{r['name']}**  \n{r.job_title} · {r.city} · {r.mbti}")
        profile = r.to_dict(); start_w, start_uid = saved_weights.get(uid, DEFAULT_W), uid
    else:
        name = st.text_input("Name", "Demo User")
        city = st.selectbox("City", list(CITIES))
        mbti = st.selectbox("MBTI type", MBTI_TYPES, index=MBTI_TYPES.index("ENFP"))
        prof = st.text_area("Professional summary", "Data analyst with 3 years of experience in healthcare analytics, skilled in SQL, Python, and business reporting. Looking to grow into a machine learning role and find a mentor.", height=120)
        about = st.text_area("About me", "I enjoy solving real-world problems, mentoring juniors, and working in collaborative team environments. Outside work I enjoy hiking and cooking. I value curiosity and kindness.", height=120)
        profile = dict(name=name, city=city, mbti=mbti, latitude=CITIES[city][0], longitude=CITIES[city][1],
                       professional_summary=prof, about_me=about)
        start_w, start_uid = DEFAULT_W.copy(), None
    if st.button("Find my top 5 matches", type="primary", use_container_width=True):
        if len(profile["professional_summary"].strip()) < 20 or len(profile["about_me"].strip()) < 20:
            st.error("Write at least a sentence in both text boxes.")
        else:
            S.profile, S.weights, S.X, S.y, S.rated, S.uid = profile, start_w, [], [], {}, start_uid
            S.w0 = np.array(start_w, float).copy()

tab_m, tab_s, tab_p = st.tabs(["Top matches", "Session report", "Model performance"])

# ------------------------------------------------------------------ Top matches
with tab_m:
    if S.profile is None:
        st.info("Fill in your profile on the left, then choose **Find my top 5 matches**.")
    else:
        F = feats(S.profile); score = F @ S.weights
        if S.uid:
            score[users.index[users.user_id == S.uid][0]] = -1
        top = np.argsort(-score)[:5]
        w = S.weights
        cols = st.columns(4)
        for col, (lab, val, nt) in zip(cols, [("Text weight", f"{w[0]:.2f}", f"started at {S.w0[0]:.2f}"),
                                              ("MBTI weight", f"{w[1]:.2f}", f"started at {S.w0[1]:.2f}"),
                                              ("Location weight", f"{w[2]:.2f}", f"started at {S.w0[2]:.2f}"),
                                              ("Feedback given", str(len(S.y)), "Accept or Reject")]):
            col.markdown(f"<div class='stat'><div class='lab'>{lab}</div><div class='val'>{val}</div>"
                         f"<div class='note'>{nt}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='section'>Top 5 matches for {S.profile['name']}</h3>", unsafe_allow_html=True)
        for rank, i in enumerate(top, 1):
            m = users.iloc[i]
            with st.container(border=True):
                a, b, d = st.columns([0.16, 0.5, 0.34])
                a.markdown(f"<div class='score'>{100*score[i]:.0f}%</div><div class='meta'>match</div>", unsafe_allow_html=True)
                b.markdown(f"<div class='name'>{rank}. {m['name']} <span class='meta'>({m.user_id})</span></div>"
                           f"<div class='meta'>{m.job_title} · {m.city} · {m.mbti} · looking for {m.looking_for}</div>",
                           unsafe_allow_html=True)
                b.caption(m.professional_summary)
                d.markdown(bar("Text similarity", F[i, 0]) + bar("MBTI pairing", F[i, 1]) + bar("Nearby", F[i, 2]),
                           unsafe_allow_html=True)
                if m.user_id in S.rated:
                    cls, label = ("yes", "Accepted") if S.rated[m.user_id] else ("no", "Rejected")
                    d.markdown(f"<span class='tag {cls}'>{label}</span>", unsafe_allow_html=True)
                else:
                    d.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                    x, y = d.columns(2)
                    x.button("Accept", key=f"a{m.user_id}", type="primary", on_click=rate, args=(m.user_id, 1), use_container_width=True)
                    y.button("Reject", key=f"r{m.user_id}", type="secondary", on_click=rate, args=(m.user_id, 0), use_container_width=True)
                with d.popover("Score breakdown", use_container_width=True):
                    st.markdown(breakdown(F[i], w, score[i]), unsafe_allow_html=True)
        out = pd.DataFrame({"rank": range(1, 6), "user_id": users.user_id.values[top], "name": users.name.values[top],
                            "job_title": users.job_title.values[top], "city": users.city.values[top],
                            "mbti": users.mbti.values[top], "match_%": np.round(100 * score[top], 1),
                            "text_%": np.round(100 * F[top, 0], 0), "mbti_%": np.round(100 * F[top, 1], 0),
                            "nearby_%": np.round(100 * F[top, 2], 0)})
        st.download_button("Download these matches (CSV)", out.to_csv(index=False), "top_matches.csv", "text/csv")
        st.markdown("<div class='note'>Rate a few matches and the weights above will shift, then the list re-ranks. "
                    "Open the Session report tab to see what your choices say.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Session report
with tab_s:
    if S.profile is None or not S.y:
        st.info("Accept or Reject a few matches in the Top matches tab and your report will appear here.")
    else:
        Xr, yr = np.array(S.X), np.array(S.y)
        acc, rej = Xr[yr == 1], Xr[yr == 0]
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Ratings given", str(len(yr))), unsafe_allow_html=True)
        k2.markdown(kpi("Accepted", str(len(acc))), unsafe_allow_html=True)
        k3.markdown(kpi("Rejected", str(len(rej))), unsafe_allow_html=True)
        k4.markdown(kpi("Acceptance rate", f"{100*yr.mean():.0f}%"), unsafe_allow_html=True)

        nm_of = users.set_index("user_id").name
        rated_names = tuple(f"{nm_of[u]} ({u})" for u in S.rated)
        if FPDF is None:
            st.caption("The PDF download needs one more package. In the terminal run: pip install fpdf2")
        else:
            pdf_bytes = build_pdf(S.profile["name"], S.profile["city"], S.profile["mbti"], tuple(S.w0),
                                  tuple(tuple(r) for r in S.X), tuple(S.y), rated_names)
            st.download_button("Download report (PDF)", pdf_bytes, "session_report.pdf", "application/pdf")

        st.markdown("<h3 class='section'>How your weights moved</h3>", unsafe_allow_html=True)
        wdf = pd.DataFrame({"Factor": FEATURE_NAMES * 2, "Weights": ["Starting"] * 3 + ["Now"] * 3,
                            "Weight": list(S.w0) + list(S.weights)})
        ch = alt.Chart(wdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Factor:N", title=None, sort=FEATURE_NAMES, axis=alt.Axis(labelAngle=0)),
            xOffset="Weights:N", y=alt.Y("Weight:Q", scale=alt.Scale(domain=[0, 1]), title="Weight"),
            color=alt.Color("Weights:N", scale=alt.Scale(domain=["Starting", "Now"], range=[TINT, MID]), title=None),
            tooltip=["Factor", "Weights", alt.Tooltip("Weight:Q", format=".2f")])
        st.altair_chart(style(ch, 260), use_container_width=True)

        st.markdown("<h3 class='section'>What your choices say</h3>", unsafe_allow_html=True)
        if len(acc) and len(rej):
            gdf = pd.DataFrame({"Factor": ["Text similarity", "MBTI pairing", "Nearby"] * 2,
                                "Group": ["Accepted"] * 3 + ["Rejected"] * 3,
                                "Average score": list(100 * acc.mean(0)) + list(100 * rej.mean(0))})
            ch = alt.Chart(gdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Factor:N", title=None, sort=["Text similarity", "MBTI pairing", "Nearby"], axis=alt.Axis(labelAngle=0)), xOffset="Group:N",
                y=alt.Y("Average score:Q", scale=alt.Scale(domain=[0, 100]), title="Average score (%)"),
                color=alt.Color("Group:N", scale=alt.Scale(domain=["Accepted", "Rejected"], range=[MOSS, ROSY]), title=None),
                tooltip=["Factor", "Group", alt.Tooltip("Average score:Q", format=".0f")])
            st.altair_chart(style(ch, 260), use_container_width=True)
            gap = acc.mean(0) - rej.mean(0)
            j = int(np.argmax(np.abs(gap))); nm = ["text similarity", "MBTI pairing", "location"][j]
            mb_gap = 100 * gap[1]
            mb_txt = ("accepted and rejected matches had about the same MBTI pairing" if abs(mb_gap) < 1 else
                      f"accepted matches were {'higher' if mb_gap > 0 else 'lower'} by {abs(mb_gap):.0f} points")
            st.markdown(
                f"<div class='insight'>Your accepted matches averaged <b>{100*acc.mean(0)[j]:.0f}%</b> on {nm}, against "
                f"<b>{100*rej.mean(0)[j]:.0f}%</b> for rejected ones, so {nm} separates your choices most so far. "
                f"On MBTI pairing, {mb_txt}, and your MBTI weight went from "
                f"{S.w0[1]:.2f} to {S.weights[1]:.2f}. This is based on only {len(yr)} ratings, so treat it as an early signal.</div>",
                unsafe_allow_html=True)
        else:
            st.info("Give at least one Accept and one Reject to compare what you liked with what you passed on.")

        st.markdown("<h3 class='section'>Your ratings</h3>", unsafe_allow_html=True)
        nm_of = users.set_index("user_id").name
        tbl = pd.DataFrame([{"Match": f"{nm_of[uid]} ({uid})", "Your choice": "Accepted" if a else "Rejected",
                             "Text %": round(100 * f[0]), "MBTI %": round(100 * f[1]), "Nearby %": round(100 * f[2])}
                            for (uid, a), f in zip(S.rated.items(), S.X)])
        st.dataframe(tbl, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Model performance (live)
with tab_p:
    if S.profile is None or not S.y:
        st.info("Accept or Reject a few matches in the Top matches tab. This tab then shows how the model performed "
                "on the ratings you just gave.")
    else:
        Xr, yr = np.array(S.X), np.array(S.y)
        L = live_metrics(Xr, yr, S.w0)
        ws, pre = L["ws"], L["pre"]
        st.markdown("<div class='sub'>Live results from your own ratings in this session. Each match is scored with the weights "
                    "the model had before you rated it, so the model is never tested on something it already learned from.</div>",
                    unsafe_allow_html=True)
        shift = ws[-1] - ws[0]; jj = int(np.argmax(np.abs(shift)))
        c = st.columns(4)
        c[0].markdown(kpi("Matches rated", str(len(yr)), f"{int(yr.sum())} accepted, {int((1 - yr).sum())} rejected"),
                      unsafe_allow_html=True)
        c[1].markdown(kpi("Acceptance rate", f"{100*yr.mean():.0f}%", "of the matches shown to you"), unsafe_allow_html=True)
        if L["auc_start"] is None:
            c[2].markdown(kpi("Ranking quality (AUC)", "n/a", "needs one Accept and one Reject"), unsafe_allow_html=True)
        else:
            c[2].markdown(kpi("Ranking quality (AUC)", f"{L['auc_start']:.2f} → {L['auc_live']:.2f}",
                              "starting weights vs learning live"), unsafe_allow_html=True)
        c[3].markdown(kpi("Biggest weight change", f"{FEATURE_NAMES[jj]} {shift[jj]:+.2f}",
                          f"{ws[0][jj]:.2f} → {ws[-1][jj]:.2f}"), unsafe_allow_html=True)

        st.markdown("<h3 class='section'>How the weights changed with each rating</h3>", unsafe_allow_html=True)
        hist = pd.DataFrame([{"Rating": k, "Factor": FEATURE_NAMES[f], "Weight": float(ws[k][f])}
                             for k in range(len(ws)) for f in range(3)])
        ch = alt.Chart(hist).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3).encode(
            x=alt.X("Rating:Q", title="After rating number (0 = starting weights)", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("Weight:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Factor:N", scale=alt.Scale(domain=FEATURE_NAMES, range=[MID, ROSY, MOSS]), title=None),
            tooltip=["Rating", "Factor", alt.Tooltip("Weight:Q", format=".2f")])
        st.altair_chart(style(ch, 280), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("<h3 class='section'>Score given before you rated</h3>", unsafe_allow_html=True)
            nm_of = users.set_index("user_id").name
            names = [f"{nm_of[u]} ({u})" for u in S.rated]
            bdf = pd.DataFrame({"Rating": range(1, len(yr) + 1), "Match": names, "Your choice": np.where(yr == 1, "Accepted", "Rejected"),
                                "Score": 100 * pre})
            ch = alt.Chart(bdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Rating:O", title="Rating number", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Match score before rating (%)"),
                color=alt.Color("Your choice:N", scale=alt.Scale(domain=["Accepted", "Rejected"], range=[MOSS, ROSY]), title=None),
                tooltip=["Match", "Your choice", alt.Tooltip("Score:Q", format=".0f")])
            st.altair_chart(style(ch, 280), use_container_width=True)
        with right:
            st.markdown("<h3 class='section'>How well does each signal predict your choices?</h3>", unsafe_allow_html=True)
            sig = pd.DataFrame([{"Signal": n, "AUC": v} for n, v in L["signals"].items() if v is not None])
            if sig.empty:
                st.info("Give at least one Accept and one Reject to see this chart.")
            else:
                ch = alt.Chart(sig).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color=MID).encode(
                    y=alt.Y("Signal:N", title=None, sort="-x", axis=alt.Axis(labelLimit=200)),
                    x=alt.X("AUC:Q", scale=alt.Scale(domain=[0, 1]), title="AUC on your ratings (dashed line = chance, 0.5)"),
                    tooltip=["Signal", alt.Tooltip("AUC:Q", format=".2f")])
                chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[5, 4], color=DARK).encode(x="x:Q")
                st.altair_chart(style(ch + chance, 280), use_container_width=True)

        if (yr == 1).any() and (yr == 0).any():
            a_m, r_m = pre[yr == 1].mean(), pre[yr == 0].mean()
            verdict = ("so the ranking put what you liked above what you passed on." if a_m > r_m else
                       "so for now the model ranked your rejected matches at least as high as the accepted ones. "
                       "It needs more ratings to learn your taste.")
            st.markdown(f"<div class='insight'>Before you rated them, the model gave your accepted matches an average score of "
                        f"<b>{100*a_m:.0f}%</b> and your rejected matches <b>{100*r_m:.0f}%</b>, {verdict} "
                        f"This is based on only {len(yr)} ratings, so treat it as an early signal.</div>", unsafe_allow_html=True)
        else:
            st.info("Give at least one Accept and one Reject to compare the scores the model gave to your choices.")
        offline_benchmark()

if S.profile is not None:
    w = S.weights
    st.markdown(f"<div class='float-chip'><div class='ttl'>Live weights</div>Text <b>{w[0]:.2f}</b> · MBTI <b>{w[1]:.2f}</b> · "
                f"Location <b>{w[2]:.2f}</b><div>{len(S.y)} rating{'s' if len(S.y) != 1 else ''} given</div></div>",
                unsafe_allow_html=True)
