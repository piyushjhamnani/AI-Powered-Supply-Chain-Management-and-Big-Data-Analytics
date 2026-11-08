"""Visual helpers for the Streamlit dashboard (warehouse-signage look: Barlow type, safety-yellow accents)."""
import html
import streamlit as st

INK, STEEL, YELLOW, RED, GREEN, MUTED, PAPER = "#1F2933", "#2F5D8A", "#F2B100", "#C0392B", "#2E7D57", "#6B7780", "#FFFFFF"
RISK_COLOR = {"CRITICAL": RED, "HIGH": "#D9822B", "MEDIUM": YELLOW, "OVERSTOCK": STEEL, "LOW": GREEN}
LEVEL_COLOR = {"critical": RED, "warn": "#D9822B", "action": STEEL, "ok": GREEN, "info": MUTED}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Barlow', 'Segoe UI', Arial, sans-serif;
}}
h1, h2, h3 {{ font-family: 'Barlow Condensed', 'Arial Narrow', Arial, sans-serif !important; letter-spacing: .2px; color: {INK}; }}
h1 {{ font-weight: 700 !important; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}

/* KPI strip: one ruled band, not a grid of cards */
.kpis {{ display:flex; flex-wrap:wrap; background:{PAPER}; border:1px solid #D5D9D2; border-radius:6px; margin:.4rem 0 1.2rem; }}
.kpi {{ flex:1 1 140px; padding:.75rem 1rem; border-right:1px solid #E3E6DF; }}
.kpi:last-child {{ border-right:0; }}
.kpi .v {{ font-family:'Barlow Condensed', Arial, sans-serif; font-size:1.9rem; font-weight:700; line-height:1.05; white-space:nowrap; color:{INK}; }}
.kpi .l {{ font-size:.86rem; color:{MUTED}; }}
.kpi.bad .v {{ color:{RED}; }}

/* Alerts: critical rows get a hazard-tape edge */
.alert {{ background:{PAPER}; border:0; border-left:10px solid {YELLOW}; padding:.6rem .9rem; margin:.45rem 0; border-radius:0 6px 6px 0; }}
.alert.CRITICAL {{ border-image: repeating-linear-gradient(-45deg, {YELLOW} 0 7px, {INK} 7px 14px) 1; border-radius:0; }}
.alert.HIGH {{ border-left-color:#D9822B; }}
.alert.MEDIUM {{ border-left-color:{YELLOW}; }}
.alert.OVERSTOCK {{ border-left-color:{STEEL}; }}
.alert .t {{ font-weight:600; color:{INK}; }}
.alert .s {{ color:#4A5560; font-size:.92rem; }}
.tag {{ display:inline-block; font-size:.78rem; font-weight:600; padding:.05rem .45rem; border-radius:3px; color:#fff; margin-right:.4rem; }}

/* Agent conveyor log */
.log {{ border-left:3px solid #CBD0C8; margin-left:.6rem; padding-left:1rem; }}
.ev {{ position:relative; padding:.35rem 0 .35rem .2rem; }}
.ev:before {{ content:""; position:absolute; left:-1.42rem; top:.75rem; width:.7rem; height:.7rem; border-radius:50%; background:var(--c); border:2px solid #F2F3EF; }}
.ev .a {{ font-family:'Barlow Condensed', Arial, sans-serif; font-weight:600; color:var(--c); margin-right:.4rem; }}
.ev .n {{ color:{MUTED}; font-size:.8rem; margin-right:.4rem; }}
[data-testid="stMetricValue"] {{ font-family:'Barlow Condensed', Arial, sans-serif; font-size:1.6rem; font-weight:600; }}
.note {{ color:{MUTED}; font-size:.9rem; }}
.banner {{ background:#FFF6D6; border:1px solid {YELLOW}; padding:.5rem .8rem; border-radius:6px; margin-bottom:.8rem; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi_strip(items):
    """items: list of (value, label, is_bad)"""
    cells = "".join(f'<div class="kpi{" bad" if bad else ""}"><div class="v">{html.escape(str(v))}</div>'
                    f'<div class="l">{html.escape(l)}</div></div>' for v, l, bad in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def alert(risk, title, sub, tag=None):
    color = RISK_COLOR.get(risk, MUTED)
    st.markdown(f'<div class="alert {risk}"><span class="tag" style="background:{color}">{tag or risk.title()}</span>'
                f'<span class="t">{html.escape(title)}</span><div class="s">{html.escape(sub)}</div></div>',
                unsafe_allow_html=True)


def agent_log(events, icons):
    rows = []
    for e in events:
        c = LEVEL_COLOR.get(e.level, MUTED)
        rows.append(f'<div class="ev" style="--c:{c}"><span class="n">{e.step:02d}</span>'
                    f'<span class="a">{icons.get(e.agent, "•")} {html.escape(e.agent)}</span>'
                    f'{html.escape(e.message)}</div>')
    st.markdown(f'<div class="log">{"".join(rows)}</div>', unsafe_allow_html=True)


def inr(x):
    """Indian-style money: ₹4.91 lakh / ₹1.2 cr / ₹8,450"""
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} lakh"
    return f"₹{x:,.0f}"
