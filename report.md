# Project Report: IoT RouteSense

**Course:** Computer Networks  
**Topic:** Graph Neural Network-Based Intelligent Network Routing and Failure Prediction  
**Research objective:** Energy and QoS-aware intelligent routing for IoT networks  
**Student:** Shourya Singh (25BCE0615)  
**Date:** 29 October 2026

## 1. Abstract

IoT networks must route data despite constrained batteries, variable latency, packet loss and failures. This project implements a runnable Python simulation that compares conventional shortest-latency routing with an energy and QoS-aware routing policy. It visualizes an IoT/WSN topology, simulates link/node failures and recomputes routes. A Random Forest estimates link-failure risk from synthetic telemetry. The project is an honest prototype: it demonstrates the routing and failure-prediction workflow, but it is not a Graph Neural Network (GNN) implementation and does not report real-network performance.

## 2. Problem statement

Shortest-path routing can select a fast route that consumes more energy or crosses unreliable links. In an IoT network, an effective route should balance latency, energy use, packet loss and the risk of failure. It must also reroute when a link or relay becomes unavailable.

## 3. Objectives

1. Model an IoT network with devices, relays and an edge gateway.
2. Compare latency-only routing with a multi-objective QoS/energy-aware policy.
3. Simulate node/link failures and automatic rerouting.
4. Display interpretable path metrics and predicted link-risk signals.
5. Clearly identify the gap between this prototype and a true GNN approach.

## 4. Methodology

The topology is an undirected weighted graph. Every edge has simulated latency (ms), energy cost (mJ), packet loss probability and bandwidth. The baseline uses Dijkstra's algorithm with latency as its edge cost. The intelligent policy uses:

```
C = wL(L/55) + wE(E/3) + wP(P/0.20) + wR(R)
```

where `L` is latency, `E` is energy, `P` is loss probability, `R` is ML-predicted link-failure risk and the `w` values are selected in the dashboard. Normalization makes the values comparable. The simulation removes failed components before routing, then recalculates both paths.

For failure prediction, a Random Forest is trained on explicitly **synthetic** telemetry: latency, loss rate, minimum endpoint battery and utilization. A synthetic holdout accuracy/AUC is shown only as a pipeline diagnostic. It must not be interpreted as a deployed network result.

## 5. Software and tools

| Component | Tool |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Topology/routing | NetworkX / Dijkstra |
| Charts | Plotly |
| Data handling | Pandas, NumPy |
| Failure-risk model | scikit-learn Random Forest |

## 6. Experiment procedure

1. Start the dashboard with `D1` as source and `GW` as destination.
2. Record the baseline and intelligent route metrics under normal conditions.
3. Adjust objective weights to represent an energy-sensitive or reliability-sensitive application.
4. Disable a route link or relay in the failure controls.
5. Observe whether an alternative path exists, and record the new path and metrics.
6. Repeat for at least three controlled scenarios: normal, congested/reliability-sensitive, and failed.

### Result-recording table

| Scenario | Baseline path | Intelligent path | Latency (ms) | Energy (mJ) | Delivery (%) | Observation |
|---|---|---|---:|---:|---:|---|
| Normal |  |  |  |  |  |  |
| Energy priority |  |  |  |  |  |  |
| Link/node failure |  |  |  |  |  |  |

Do not fill this table with invented values. Copy the values visible in the dashboard after each run.

## 7. Key observations to discuss

- The latency baseline is optimal only for its single objective.
- A multi-objective route can deliberately use an additional hop when it lowers energy or avoids a high-risk link.
- Rerouting succeeds only when a connected alternative exists; a disconnected graph correctly reports "Unreachable".
- Results are scenario-dependent because the weights express the application's policy.

## 8. Limitations and future work

This is not an NS-3 experiment, a packet-level wireless simulator, or a GNN. Real validation requires measured/NS-3 telemetry, train/validation/test splits by topology and time, calibration of risk outputs, and repeated trials with confidence intervals. A next step is to generate packet-level datasets in NS-3 and train a GNN (for example, a GCN/GAT) whose node/edge features include queue length, RSSI, residual energy, loss and topology context. Its learned risk scores could then replace the prototype's Random Forest feature model.

## 9. Conclusion

IoT RouteSense provides a runnable and explainable demonstration of QoS/energy-aware routing and failure-aware rerouting. It is suitable as a small project prototype because every displayed result is derived from the configured simulated graph, while its limits are explicit. A genuine GNN is the logical research extension rather than a claim made by this version.

## References supplied with the assignment

1. S. Stryczek and M. Natkaniec, "ARCAS: A novel solution for mitigating performance drops in dense IEEE 802.11 networks," *Computer Communications*, vol. 253, Art. 108501, 2026. DOI: 10.1016/j.comcom.2026.108501.
2. J. Bacha et al., "Deep reinforcement learning based interference optimization for coordinated beamforming in ultra-dense Wi-Fi networks," *Computer Communications*, vol. 242, Art. 108286, 2025. DOI: 10.1016/j.comcom.2025.108286.
3. R. Oliveira et al., "Optimal action-selection optimization of wireless networks based on virtual representation," *Computer Networks*, vol. 264, Art. 111258, 2025. DOI: 10.1016/j.comnet.2025.111258.
4. P. Venkatesan and N. Kumaratharan, "Reinforcement learning-based dynamic spectrum allocation for efficient cognitive radio network management," *Computer Networks*, vol. 281, Art. 112217, 2026. DOI: 10.1016/j.comnet.2026.112217.
