"""Streamlit dashboard for an energy and QoS-aware IoT routing simulation (simplified UI)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from routing_engine import (
    DEFAULT_SOURCE, DEFAULT_TARGET, EDGE_DATA, NODE_META, RoutingWeights, add_routing_costs,
    attach_predicted_risk, create_topology, path_metrics, route, route_edges,
    set_failures, train_failure_predictor,
)

st.set_page_config(page_title="IoT RouteSense", page_icon="🛰️", layout="wide")

BASELINE_COLOR = "#f2b84b"   # amber  - shortest-latency route
SMART_COLOR = "#3fd9c0"      # teal   - QoS/energy-aware route
FAIL_COLOR = "#ef5b6b"       # red    - failed links/nodes

st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap');
    html, body, .stApp, [data-testid="stSidebar"] {{font-family: 'Plus Jakarta Sans', sans-serif;}}
    .stApp {{background: #0b1320; color: #e8eef7;}}
    [data-testid="stSidebar"] {{background: #0f1a2b; border-right: 1px solid #1f3049;}}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1.6rem; max-width: 1250px;}}
    h1.title {{font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;}}
    p.sub {{color: #93a7c1; margin: .25rem 0 1.1rem; font-size: 1rem;}}
    .card {{border: 1px solid #22344f; border-left: 5px solid var(--accent); border-radius: 10px;
            background: #111f33; padding: 1rem 1.1rem; margin-bottom: .9rem;}}
    .card h4 {{margin: 0 0 .6rem; font-size: 1rem; font-weight: 700; color: #cfdcee;}}
    .path {{display: flex; flex-wrap: wrap; gap: .35rem; align-items: center; margin-bottom: .8rem;}}
    .chip {{background: #1a2d48; border-radius: 6px; padding: .15rem .55rem; font-weight: 700; font-size: .9rem;}}
    .arrow {{color: #5f7592;}}
    .stats {{display: flex; gap: 1.6rem;}}
    .stat b {{display: block; font-size: 1.35rem; font-weight: 800;}}
    .stat span {{color: #8aa0bd; font-size: .8rem;}}
    .verdict {{padding: .75rem 1rem; border-radius: 10px; background: #14263d; color: #d6e4f5; font-size: .95rem;}}
    .legend span {{display: inline-block; margin-right: 1.1rem; font-size: .85rem; color: #9db1cb;}}
    .dot {{display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .35rem;}}
</style>""", unsafe_allow_html=True)

# Simple priority presets replace the four separate weight sliders.
PRESETS = {
    "Balanced": RoutingWeights(0.40, 0.30, 0.20, 0.10),
    "Save energy": RoutingWeights(0.15, 0.60, 0.15, 0.10),
    "Most reliable": RoutingWeights(0.15, 0.15, 0.40, 0.30),
    "Fastest": RoutingWeights(0.70, 0.10, 0.10, 0.10),
}
DEFAULT_UTILIZATION = 0.45


@st.cache_resource
def predictor():
    return train_failure_predictor()


def make_figure(graph, baseline_path, smart_path):
    fig = go.Figure()
    baseline_edges, smart_edges = route_edges(baseline_path), route_edges(smart_path)
    for u, v, data in graph.edges(data=True):
        x0, y0 = graph.nodes[u]["pos"]
        x1, y1 = graph.nodes[v]["pos"]
        edge = frozenset((u, v))
        if not data["available"]:
            color, width, dash = FAIL_COLOR, 2, "dash"
        elif edge in smart_edges:
            color, width, dash = SMART_COLOR, 5, "solid"
        elif edge in baseline_edges:
            color, width, dash = BASELINE_COLOR, 3, "dot"
        else:
            color, width, dash = "#3b5578", 1.5, "solid"
        hover = f"{u} - {v}<br>Latency: {data['latency']:.0f} ms<br>Energy: {data['energy']:.1f} mJ"
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="text", text=[hover, hover],
                                 line=dict(color=color, width=width, dash=dash), showlegend=False))
    roles = {"Device": "#6fa8ff", "Relay": "#c9a3ff", "Gateway": "#ffcf70"}
    for role, color in roles.items():
        nodes = [n for n, d in graph.nodes(data=True) if d["role"] == role]
        fig.add_trace(go.Scatter(
            x=[graph.nodes[n]["pos"][0] for n in nodes], y=[graph.nodes[n]["pos"][1] for n in nodes],
            mode="markers+text", text=nodes, textposition="top center", name=role,
            hovertemplate="<b>%{text}</b><extra></extra>",
            marker=dict(size=26, color=[color if graph.nodes[n]["available"] else FAIL_COLOR for n in nodes],
                        line=dict(color="#0b1320", width=2)),
        ))
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#d8e4f3"),
                      legend=dict(orientation="h", y=1.04, x=0), xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


def route_card(title, path, metrics, accent):
    if not path or metrics["status"] != "Reachable":
        body = "<div style='color:#ef5b6b;font-weight:700'>Destination unreachable with the current failures</div>"
    else:
        chips = "<span class='arrow'>›</span>".join(f"<span class='chip'>{n}</span>" for n in path)
        body = (f"<div class='path'>{chips}</div><div class='stats'>"
                f"<div class='stat'><b>{metrics['latency_ms']:.0f} ms</b><span>Latency</span></div>"
                f"<div class='stat'><b>{metrics['energy_mj']:.1f} mJ</b><span>Energy</span></div>"
                f"<div class='stat'><b>{metrics['packet_delivery_pct']:.1f}%</b><span>Delivery</span></div></div>")
    st.markdown(f"<div class='card' style='--accent:{accent}'><h4>{title}</h4>{body}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛰️ Controls")
    source = st.selectbox("From", list(NODE_META), index=list(NODE_META).index(DEFAULT_SOURCE))
    target = st.selectbox("To", list(NODE_META), index=list(NODE_META).index(DEFAULT_TARGET))
    priority = st.radio("What matters most?", list(PRESETS), help="Changes how the smart route weighs latency, energy, loss and risk.")
    with st.expander("Simulate a failure"):
        failed_nodes = st.multiselect("Failed nodes", list(NODE_META))
        failed_links_text = st.multiselect("Failed links", [f"{u} - {v}" for u, v, *_ in EDGE_DATA])

# ---------------------------------------------------------------- compute
model, _ = predictor()
graph = create_topology()
failed_links = [tuple(text.split(" - ")) for text in failed_links_text]
set_failures(graph, failed_nodes, failed_links)
attach_predicted_risk(graph, model, DEFAULT_UTILIZATION)
add_routing_costs(graph, PRESETS[priority])
baseline_path = route(graph, source, target, "baseline")
smart_path = route(graph, source, target, "intelligent")
baseline_m, smart_m = path_metrics(graph, baseline_path), path_metrics(graph, smart_path)

# ---------------------------------------------------------------- page
st.markdown("<h1 class='title'>IoT RouteSense</h1>"
            "<p class='sub'>Compare the fastest route with a smarter one that also considers energy, packet loss and failure risk.</p>",
            unsafe_allow_html=True)

left, right = st.columns([1.6, 1])
with left:
    st.plotly_chart(make_figure(graph, baseline_path, smart_path), use_container_width=True)
    st.markdown(
        f"<div class='legend'><span><i class='dot' style='background:{SMART_COLOR}'></i>Smart route</span>"
        f"<span><i class='dot' style='background:{BASELINE_COLOR}'></i>Fastest route</span>"
        f"<span><i class='dot' style='background:{FAIL_COLOR}'></i>Failed</span></div>", unsafe_allow_html=True)

with right:
    route_card("Smart route", smart_path, smart_m, SMART_COLOR)
    route_card("Fastest route", baseline_path, baseline_m, BASELINE_COLOR)

    if baseline_m["status"] == "Reachable" and smart_m["status"] == "Reachable":
        if baseline_path == smart_path:
            msg = "Both policies picked the same route. Try <b>Save energy</b> or simulate a failure."
        else:
            d_energy = baseline_m["energy_mj"] - smart_m["energy_mj"]
            d_lat = smart_m["latency_ms"] - baseline_m["latency_ms"]
            msg = (f"The smart route uses <b>{abs(d_energy):.1f} mJ {'less' if d_energy >= 0 else 'more'}</b> energy "
                   f"and takes <b>{abs(d_lat):.0f} ms {'longer' if d_lat >= 0 else 'less'}</b>.")
        st.markdown(f"<div class='verdict'>{msg}</div>", unsafe_allow_html=True)

st.caption("Prototype note: link values and the Random Forest risk model use simulated data. This is not a Graph Neural Network; "
           "a GNN is future work.")