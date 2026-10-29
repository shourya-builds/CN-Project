# IoT RouteSense - Intelligent Routing and Failure Prediction

A polished small-project prototype for **Computer Networks Assignment 8**:
**Graph Neural Network-Based Intelligent Network Routing and Failure Prediction**.

> Important: the delivered application is a working **weighted-graph routing prototype** with a small **Random Forest failure-risk estimator** trained on synthetic data. It is **not a Graph Neural Network (GNN)** and does not make claims about real-world GNN performance. The report explains how a GNN can be added later.

## What it demonstrates

- A deterministic IoT/WSN topology with 10 devices, relays and a gateway.
- Conventional shortest-latency routing (Dijkstra baseline).
- QoS/energy-aware routing using latency, energy, packet loss and predicted failure risk.
- Interactive simulated node or link failures and automatic rerouting.
- Live comparison of latency, energy, delivery probability and cumulative failure risk.
- A real, small ML pipeline: Random Forest trained/evaluated on explicitly synthetic telemetry.

## Run it on Windows

Open PowerShell **inside this folder**, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The browser should open at `http://localhost:8501`. If PowerShell blocks activation, use this one-session command first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

To stop the app, press `Ctrl+C` in PowerShell.

**Easiest option:** double-click `start_project.bat` in File Explorer. It creates the environment on the first run, installs the required packages, and starts the dashboard.

## Fast demo route (2-3 minutes)

1. Run the app and keep source `D1` and destination `GW`.
2. Point out the amber dotted baseline and teal intelligent route.
3. Change the **Energy** slider upward: the QoS/energy-aware route may prefer a lower-energy path.
4. Select a link from **Fail links** that is currently on a route. The route recomputes immediately.
5. Show the comparison chart and risk table. State that the ML risk score is trained on synthetic data.
6. Close with the limitation: "The dashboard implements the routing/failure workflow. A GNN is planned as the next research implementation step."

## Verification

After installing requirements, run:

```powershell
python -m unittest discover -s tests -v
```

## Project layout

```
app.py                   Streamlit dashboard
routing_engine.py        topology, routing, ML risk module and metrics
tests/test_routing_engine.py
report.md                concise submission report
demo_script.md           presentation/viva speaking notes
requirements.txt
```

## Routing objective

For each usable edge, the intelligent policy minimizes:

`C = wL*(latency/55) + wE*(energy/3) + wP*(loss/0.20) + wR*(predicted failure risk)`

The UI supplies the weights. Failed nodes and links are excluded before Dijkstra's algorithm runs.

## Limitations and ethical reporting

- Link values and ML telemetry are simulated; delivery and risk values are estimates, not measured traffic.
- The ML score is not a forecast validated against field traces.
- The reported AUC is strictly a diagnostic on held-out synthetic data.
- This application does not include NS-3, packet-level radio simulation or a GNN. These are appropriate extensions, not completed features.
"# CN-Project" 
