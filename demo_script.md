# 3-Minute Demonstration Script

## Opening (20 seconds)

"Good morning. Our project is IoT RouteSense, an energy and QoS-aware intelligent routing and failure prediction prototype for IoT networks. It compares conventional shortest-path routing with a route that also considers energy, loss and predicted failure risk."

## Topology and baseline (35 seconds)

"The blue circles are sensors, purple circles are relays, and the yellow circle is the edge gateway. Each connection has latency, energy cost and packet-loss values. The dotted amber path is the conventional Dijkstra baseline. It minimizes latency only."

## Intelligent policy (40 seconds)

"The teal path is the intelligent prototype. Its cost function combines normalized latency, energy, packet-loss risk and failure risk. These sliders express the requirement of an application: for example, a battery-sensitive IoT deployment can give energy a higher weight. The dashboard gives latency, energy, delivery probability and route-level risk for both paths."

## Failure and rerouting (45 seconds)

"Now I simulate a link failure by selecting this link. It turns red dashed and is removed from routing. Both methods immediately recompute. If a connected alternative exists, we get a new route; if not, the system correctly shows unreachable. This models failure-aware routing rather than pretending every failure can be repaired."

## Prediction module and honesty (30 seconds)

"The link-risk monitor is a Random Forest failure-risk estimator trained on synthetic telemetry: latency, loss, battery and utilization. The AUC shown is only a synthetic holdout diagnostic. Although our assigned title mentions GNNs, this submission does not claim to implement a Graph Neural Network. A GNN trained on NS-3 or real telemetry is our planned extension."

## Close (10 seconds)

"In summary, the prototype proves the end-to-end routing, comparison and rerouting workflow, with transparent assumptions and clear next steps toward a GNN-based system. Thank you."
