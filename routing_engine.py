"""Core simulation logic for the Intelligent IoT Routing prototype.

This module implements a weighted-graph routing baseline and a small supervised
ML model for link-failure risk.  It deliberately does *not* claim to implement
a Graph Neural Network; a GNN is documented as future work in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


DEFAULT_SOURCE = "D1"
DEFAULT_TARGET = "GW"


@dataclass(frozen=True)
class RoutingWeights:
    """Relative importance given to normalized routing objectives."""

    latency: float = 0.40
    energy: float = 0.30
    loss: float = 0.20
    failure_risk: float = 0.10


NODE_META = {
    "D1": ("Sensor A", "Device", 1.0, (0.7, 2.5)),
    "D2": ("Sensor B", "Device", 0.92, (1.1, 0.8)),
    "D3": ("Sensor C", "Device", 0.78, (2.2, 3.3)),
    "D4": ("Sensor D", "Device", 0.86, (3.2, 0.7)),
    "D5": ("Sensor E", "Device", 0.64, (4.0, 2.9)),
    "R1": ("Relay 1", "Relay", 0.90, (2.2, 2.1)),
    "R2": ("Relay 2", "Relay", 0.72, (3.6, 1.9)),
    "R3": ("Relay 3", "Relay", 0.81, (4.9, 2.3)),
    "R4": ("Relay 4", "Relay", 0.58, (3.9, 3.7)),
    "R5": ("Relay 5", "Relay", 0.89, (5.8, 3.5)),
    "GW": ("Edge Gateway", "Gateway", 1.0, (7.3, 2.5)),
}

# (from, to, latency_ms, energy_mj, loss_probability, bandwidth_mbps)
EDGE_DATA = [
    ("D1", "R1", 8, 1.8, 0.012, 12), ("D1", "D2", 5, 2.9, 0.022, 8),
    ("D2", "R1", 7, 1.5, 0.010, 12), ("D2", "R2", 13, 2.2, 0.018, 10),
    ("D3", "R1", 10, 1.4, 0.011, 12), ("D3", "R4", 10, 1.8, 0.016, 10),
    ("D4", "R2", 9, 1.5, 0.012, 12), ("D4", "R1", 14, 2.5, 0.025, 8),
    ("D5", "R4", 7, 1.6, 0.013, 10), ("D5", "R3", 12, 2.1, 0.020, 8),
    ("R1", "R2", 8, 1.4, 0.009, 18), ("R1", "R4", 15, 1.3, 0.014, 16),
    ("R2", "R3", 8, 1.6, 0.009, 18), ("R2", "R4", 11, 1.0, 0.012, 15),
    ("R3", "R4", 10, 1.1, 0.010, 16), ("R3", "R5", 7, 1.7, 0.011, 20),
    ("R4", "R5", 9, 0.9, 0.010, 18), ("R5", "GW", 8, 1.2, 0.008, 25),
    ("R3", "GW", 15, 2.8, 0.024, 14),
]


def create_topology() -> nx.Graph:
    """Return a fresh deterministic IoT/WSN topology with edge QoS metadata."""
    graph = nx.Graph()
    for node, (label, role, battery, position) in NODE_META.items():
        graph.add_node(node, label=label, role=role, battery=battery, pos=position, available=True)
    for u, v, latency, energy, loss, bandwidth in EDGE_DATA:
        graph.add_edge(
            u, v, latency=float(latency), energy=float(energy), loss=float(loss),
            bandwidth=float(bandwidth), available=True, predicted_risk=0.0,
        )
    return graph


def set_failures(graph: nx.Graph, failed_nodes: Iterable[str], failed_links: Iterable[tuple[str, str]]) -> None:
    """Apply failures in place.  Incident links of a failed node become unavailable."""
    failed_nodes = set(failed_nodes)
    failed_links = {frozenset(link) for link in failed_links}
    for node in graph.nodes:
        graph.nodes[node]["available"] = node not in failed_nodes
    for u, v, data in graph.edges(data=True):
        data["available"] = u not in failed_nodes and v not in failed_nodes and frozenset((u, v)) not in failed_links


def _synthetic_training_data(seed: int = 42, samples: int = 1600) -> tuple[pd.DataFrame, np.ndarray]:
    """Create transparently synthetic telemetry for a small demonstrator model."""
    rng = np.random.default_rng(seed)
    latency = rng.uniform(3, 55, samples)
    loss = rng.uniform(0.002, 0.20, samples)
    battery = rng.uniform(0.15, 1.0, samples)
    utilization = rng.uniform(0.05, 1.0, samples)
    # Synthetic label: degraded QoS and depleted devices increase failure likelihood.
    score = -4.0 + 0.060 * latency + 8.5 * loss + 1.8 * utilization + 2.2 * (1 - battery)
    probability = 1 / (1 + np.exp(-score))
    labels = rng.binomial(1, probability)
    return pd.DataFrame({"latency_ms": latency, "loss_rate": loss, "battery": battery, "utilization": utilization}), labels


def train_failure_predictor(seed: int = 42) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train a real, repeatable Random Forest on synthetic telemetry.

    The metrics evaluate synthetic holdout data only and are therefore not a
    claim about performance on a physical network.
    """
    features, labels = _synthetic_training_data(seed)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=seed, stratify=labels)
    model = RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=5, random_state=seed, class_weight="balanced")
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return model, {
        "holdout_accuracy": float(accuracy_score(y_test, probabilities >= 0.5)),
        "holdout_auc": float(roc_auc_score(y_test, probabilities)),
        "positive_rate": float(np.mean(labels)),
    }


def attach_predicted_risk(graph: nx.Graph, model: RandomForestClassifier, utilization: float = 0.45) -> None:
    """Predict a risk score for each link from current link and endpoint telemetry."""
    records, edges = [], []
    for u, v, data in graph.edges(data=True):
        records.append({
            "latency_ms": data["latency"], "loss_rate": data["loss"],
            "battery": min(graph.nodes[u]["battery"], graph.nodes[v]["battery"]),
            "utilization": utilization,
        })
        edges.append((u, v))
    risk = model.predict_proba(pd.DataFrame(records))[:, 1]
    for edge, value in zip(edges, risk):
        graph.edges[edge]["predicted_risk"] = float(value)


def available_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy containing only currently usable nodes and links."""
    result = nx.Graph()
    for node, data in graph.nodes(data=True):
        if data["available"]:
            result.add_node(node, **data)
    for u, v, data in graph.edges(data=True):
        if data["available"] and u in result and v in result:
            result.add_edge(u, v, **data)
    return result


def add_routing_costs(graph: nx.Graph, weights: RoutingWeights) -> None:
    """Attach normalized composite QoS/energy/risk costs to usable edges."""
    for _, _, data in graph.edges(data=True):
        data["baseline_cost"] = data["latency"]
        data["intelligent_cost"] = (
            weights.latency * (data["latency"] / 55.0)
            + weights.energy * (data["energy"] / 3.0)
            + weights.loss * (data["loss"] / 0.20)
            + weights.failure_risk * data["predicted_risk"]
        )


def route(graph: nx.Graph, source: str, target: str, strategy: str) -> list[str] | None:
    """Return a selected path or None if the destination is unreachable."""
    usable = available_graph(graph)
    if source not in usable or target not in usable:
        return None
    weight = "baseline_cost" if strategy == "baseline" else "intelligent_cost"
    try:
        return nx.shortest_path(usable, source=source, target=target, weight=weight)
    except nx.NetworkXNoPath:
        return None


def path_metrics(graph: nx.Graph, path: list[str] | None) -> dict[str, float | int | str]:
    """Compute interpretable metrics from a route's link-level values."""
    if not path:
        return {"status": "Unreachable", "hops": 0, "latency_ms": 0.0, "energy_mj": 0.0, "packet_delivery_pct": 0.0, "risk_pct": 0.0}
    links = [graph.edges[u, v] for u, v in zip(path, path[1:])]
    survival = float(np.prod([1 - edge["loss"] for edge in links]))
    no_failure = float(np.prod([1 - edge["predicted_risk"] for edge in links]))
    return {
        "status": "Reachable", "hops": len(path) - 1,
        "latency_ms": float(sum(edge["latency"] for edge in links)),
        "energy_mj": float(sum(edge["energy"] for edge in links)),
        "packet_delivery_pct": 100 * survival,
        "risk_pct": 100 * (1 - no_failure),
    }


def route_edges(path: list[str] | None) -> set[frozenset[str]]:
    return set() if not path else {frozenset((u, v)) for u, v in zip(path, path[1:])}


def ranked_link_risks(graph: nx.Graph) -> pd.DataFrame:
    """Format the current link-risk predictions for dashboard display."""
    rows = []
    for u, v, data in graph.edges(data=True):
        rows.append({"Link": f"{u} - {v}", "Failure risk (%)": round(100 * data["predicted_risk"], 1), "Latency (ms)": data["latency"], "Loss (%)": round(100 * data["loss"], 1), "Available": "Yes" if data["available"] else "No"})
    return pd.DataFrame(rows).sort_values("Failure risk (%)", ascending=False).reset_index(drop=True)
