import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time
import base64

# -----------------------------
# Model Functions
# -----------------------------

def update_node(prob, beta, sigma, gamma, inf_pressure):
    S, E, I, R = prob
    dSE = beta * S * inf_pressure
    dEI = sigma * E
    dIR = gamma * I

    S_new = S - dSE
    E_new = E + dSE - dEI
    I_new = I + dEI - dIR
    R_new = R + dIR

    vec = np.array([S_new, E_new, I_new, R_new])
    vec = np.clip(vec, 0.0, 1.0)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec

def build_network(n_nodes, k, seed, net_type):
    rng = np.random.default_rng(seed)
    if net_type == "Random":
        # Random k-nearest neighbors (no preferential attachment)
        neighbors = {i: rng.choice(n_nodes, size=k, replace=False) for i in range(n_nodes)}
        G = nx.random_regular_graph(k, n_nodes, seed=seed)
    elif net_type == "Scale-Free":
        G = nx.barabasi_albert_graph(n_nodes, max(1,k//2), seed=seed)
    elif net_type == "Small-World":
        G = nx.watts_strogatz_graph(n_nodes, k if k%2==0 else k+1, 0.1, seed=seed)
    else:
        raise ValueError("Unknown network type")
    # Build adjacency list
    neighbors = {i: list(G.neighbors(i)) for i in G.nodes()}
    return neighbors, G

def simulate_nodes(n_nodes, steps, beta, sigma, gamma, network=False, neighbors=None, policy=None, rng=None):
    nodes = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(n_nodes)]
    nodes[0] = np.array([0.0, 0.0, 1.0, 0.0])  # seed infection

    traj = []
    for t in range(steps):
        eff_beta = beta
        if policy:
            if policy["type"] == "lockdown" and t >= policy["start"]:
                eff_beta *= (1 - policy["reduction"])
            if policy["type"] == "vaccination" and t == policy["start"]:
                frac = policy["coverage"]
                for i in rng.choice(n_nodes, size=int(frac*n_nodes), replace=False):
                    nodes[i] = np.array([0,0,0,1])

        inf_pressures = []
        if network:
            for i in range(n_nodes):
                if neighbors[i]:
                    inf_pressures.append(np.mean([nodes[j][2] for j in neighbors[i]]))
                else:
                    inf_pressures.append(0.0)
        else:
            global_inf = np.mean([node[2] for node in nodes])
            inf_pressures = [global_inf]*n_nodes

        nodes = [update_node(node, eff_beta, sigma, gamma, inf_pressures[i]) for i,node in enumerate(nodes)]
        avg = np.mean(nodes, axis=0) * n_nodes
        traj.append([t, *avg])

    return pd.DataFrame(traj, columns=["Step", "S", "E", "I", "R"])

def simulate_ensemble(n_runs, n_nodes, steps, beta, sigma, gamma, network, neighbors, policy, seed):
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_runs):
        df = simulate_nodes(n_nodes, steps, beta, sigma, gamma, network, neighbors, policy, rng)
        runs.append(df[["S","E","I","R"]].values)
    runs = np.stack(runs)
    mean = runs.mean(axis=0)
    lo = np.percentile(runs, 5, axis=0)
    hi = np.percentile(runs, 95, axis=0)
    return mean, lo, hi

# -----------------------------
# Classic SEIR Network Model (for comparison)
# -----------------------------
def classic_seir_binary_network(n_nodes, steps, beta, sigma, gamma, neighbors):
    states = np.zeros((n_nodes, 4))  # S, E, I, R
    states[:, 0] = 1.0
    states[0, 0] = 0
    states[0, 2] = 1.0

    traj = []
    for t in range(steps):
        new_states = states.copy()
        for i in range(n_nodes):
            S, E, I, R = states[i]
            inf_pressure = np.mean([states[j][2] for j in neighbors[i]]) if neighbors[i] else 0
            dSE = beta * S * inf_pressure
            dEI = sigma * E
            dIR = gamma * I

            S_new = S - dSE
            E_new = E + dSE - dEI
            I_new = I + dEI - dIR
            R_new = R + dIR

            new_states[i] = np.clip([S_new, E_new, I_new, R_new], 0, 1)
            total = new_states[i].sum()
            if total > 0:
                new_states[i] /= total
        states = new_states
        avg = states.mean(axis=0) * n_nodes
        traj.append(avg)
    return np.array(traj)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Quantum-Inspired Probabilistic SEIR Model", layout="wide")

st.title("Quantum-Inspired Probabilistic SEIR Model with Extensions")

# Introduction
st.markdown("""
This interactive app showcases a **quantum-inspired probabilistic SEIR epidemiological model** that incorporates network structures and policy interventions. 

**What this project demonstrates:**

- Modeling disease spread using probabilistic node states instead of binary status.
- Network effects on transmission dynamics (Random, Scale-Free, Small-World).
- How interventions like lockdowns and vaccination affect outcomes.
- Comparison with classical SEIR on networks.
- Uncertainty quantification via ensemble simulations.
- Applicability of this modeling approach to fields like **chemical engineering** (e.g., reaction-diffusion systems, catalysis, membrane processes).

Explore parameters, visualize results, and understand the benefits of probabilistic network modeling!
""")

# Sidebar: Presets
st.sidebar.header("Scenario Presets")
preset = st.sidebar.selectbox("Select Preset Scenario", ["Custom", "COVID-like", "Seasonal Flu", "High-Contact Campus", "Chemical Engineering"])

# Default parameters based on presets
preset_params = {
    "COVID-like":     {"beta": 0.5, "sigma": 0.2, "gamma": 0.1, "network": True, "net_type": "Scale-Free", "k": 6, "policy": None},
    "Seasonal Flu":   {"beta": 0.3, "sigma": 0.3, "gamma": 0.3, "network": False, "net_type": "Random", "k": 5, "policy": None},
    "High-Contact Campus": {"beta": 0.7, "sigma": 0.25, "gamma": 0.15, "network": True, "net_type": "Small-World", "k": 10, "policy": None},
    "Chemical Engineering": {"beta": 0.4, "sigma": 0.15, "gamma": 0.12, "network": True, "net_type": "Random", "k": 4, "policy": None},
    "Custom": {}
}

# Apply presets or allow custom input
preset_choice = preset_params.get(preset, {})

# Sidebar Controls
st.sidebar.header("Simulation Controls")

n_nodes = st.sidebar.slider("Number of Nodes", 50, 2000, 500, step=50,
    help="Total number of individuals (nodes) in the simulation.")

steps = st.sidebar.slider("Number of Steps", 10, 500, 160, step=10,
    help="Number of time steps to simulate disease spread.")

beta = st.sidebar.slider("Infection Rate β", 0.0, 2.0, preset_choice.get("beta", 0.3), 0.01,
    help="Rate at which susceptible individuals become exposed due to infectious contacts.")

sigma = st.sidebar.slider("Incubation Rate σ", 0.0, 1.0, preset_choice.get("sigma", 0.2), 0.01,
    help="Rate at which exposed individuals become infectious.")

gamma = st.sidebar.slider("Recovery Rate γ", 0.0, 1.0, preset_choice.get("gamma", 0.1), 0.01,
    help="Rate at which infected individuals recover and become immune.")

seed = st.sidebar.number_input("Random Seed", 0, 9999, 42,
    help="Seed for random number generation (reproducibility).")

network = st.sidebar.checkbox("Enable Network Structure", value=preset_choice.get("network", False),
    help="Toggle to simulate disease spread on a structured network rather than a fully mixed population.")

net_type = st.sidebar.selectbox("Network Type", ["Random", "Scale-Free", "Small-World"], index=["Random", "Scale-Free", "Small-World"].index(preset_choice.get("net_type", "Random")),
    help="Choose the network topology representing how individuals are connected and interact.")

k = st.sidebar.slider("Average Neighbors", 1, 50, preset_choice.get("k", 5),
    help="Average number of connections each node has, influencing transmission paths.")

# Policy Interventions
policy_type = st.sidebar.selectbox("Policy Intervention", ["None", "Lockdown", "Vaccination"])

policy = None
if policy_type == "Lockdown":
    start_lock = st.sidebar.slider("Lockdown Start Step", 0, steps, 50,
        help="Time step at which lockdown (transmission reduction) begins.")
    reduction = st.sidebar.slider("Transmission Reduction (%)", 0, 100, 50,
        help="Percentage reduction in infection rate β during lockdown.")
    policy = {"type":"lockdown", "start": start_lock, "reduction": reduction/100}
elif policy_type == "Vaccination":
    start_vax = st.sidebar.slider("Vaccination Start Step", 0, steps, 20,
        help="Time step at which vaccination campaign starts.")
    coverage = st.sidebar.slider("Vaccination Coverage (%)", 0, 100, 20,
        help="Fraction of the population vaccinated (moved to recovered) at vaccination start.")
    policy = {"type":"vaccination", "start": start_vax, "coverage": coverage/100}

ensemble_runs = st.sidebar.slider("Ensemble Runs (Uncertainty)", 1, 50, 10,
    help="Number of simulation repeats for uncertainty quantification.")

compare_classic = st.sidebar.checkbox("Compare to Classic SEIR Network Model",
    help="Overlay the classic SEIR model for direct comparison with quantum-inspired probabilistic model.")

# -----------------------------
# Network Build and Display
# -----------------------------
if network:
    neighbors, G = build_network(n_nodes, k, seed, net_type)
    st.markdown(f"### Network Structure: **{net_type}**")
    st.markdown("Below is a sample visualization of the network structure representing connections between nodes.")

    pos = nx.spring_layout(G, seed=seed)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='blue',
            size=6,
            line_width=1),
        text=[f"Node {i}" for i in G.nodes()]
    )

    fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title="Network Visualization",
                                         showlegend=False,
                                         height=400,
                                         margin=dict(b=20,l=5,r=5,t=40)))
    st.plotly_chart(fig_net, use_container_width=True)
else:
    neighbors = None
    st.markdown("### No Network Structure Enabled (Fully Mixed Population)")

# -----------------------------
# Simulation Execution with Spinner
# -----------------------------
with st.spinner("Running simulation..."):
    start_time = time.perf_counter()
    mean, lo, hi = simulate_ensemble(ensemble_runs, n_nodes, steps, beta, sigma, gamma, network, neighbors, policy, seed)
    runtime = time.perf_counter() - start_time

# -----------------------------
# Plot Results with Uncertainty Bands
# -----------------------------
df = pd.DataFrame(mean, columns=["S","E","I","R"])
df["Step"] = np.arange(steps)
df_lo = pd.DataFrame(lo, columns=["S","E","I","R"])
df_hi = pd.DataFrame(hi, columns=["S","E","I","R"])
df_lo["Step"] = df_hi["Step"] = df["Step"]

fig = go.Figure()
colors = {"S":"blue","E":"orange","I":"red","R":"green"}
for comp in ["S","E","I","R"]:
    fig.add_trace(go.Scatter(x=df["Step"], y=df[comp], mode="lines", name=f"{comp} (mean)", line=dict(color=colors[comp])))
    fig.add_trace(go.Scatter(x=pd.concat([df["Step"], df["Step"][::-1]]),
                             y=pd.concat([df_hi[comp], df_lo[comp][::-1]]),
                             fill="toself", fillcolor=colors[comp], opacity=0.2,
                             line=dict(color="rgba(255,255,255,0)"), showlegend=False))

# Classic model comparison overlay
if compare_classic and network:
    classic_states = classic_seir_binary_network(n_nodes, steps, beta, sigma, gamma, neighbors)
    for idx, comp in enumerate(["S","E","I","R"]):
        fig.add_trace(go.Scatter(x=df["Step"], y=classic_states[:, idx], name=f"{comp} (classic)", line=dict(dash="dot")))

fig.update_layout(title="SEIR Model Dynamics with Uncertainty Bands", xaxis_title="Step", yaxis_title="Population")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Final Distribution Pie Chart
# -----------------------------
final = df.iloc[-1][["S","E","I","R"]]
pie = px.pie(values=final.values, names=final.index, title="Final Population Distribution")
st.plotly_chart(pie, use_container_width=True)

# -----------------------------
# Show Runtime
# -----------------------------
st.markdown(f"**Simulation completed in {runtime:.2f} seconds with {ensemble_runs} ensemble runs.**")

# -----------------------------
# Parameter Sensitivity Explorer
# -----------------------------
st.header("🧪 Parameter Sensitivity Explorer")
st.markdown("""
Use this tool to explore how varying one parameter affects the total number of infections at the end of the simulation.
""")
param_to_vary = st.selectbox("Select Parameter to Vary", ["Infection Rate β", "Incubation Rate σ", "Recovery Rate γ"])
param_range = st.slider("Range for Parameter", 0.0, 2.0, (0.05, 1.0), 0.01)
num_points = st.slider("Number of Steps in Sensitivity", 5, 20, 10)

fixed_params = {"beta": beta, "sigma": sigma, "gamma": gamma}

vary_vals = np.linspace(param_range[0], param_range[1], num_points)
results = []

for val in vary_vals:
    b = fixed_params["beta"]
    s = fixed_params["sigma"]
    g = fixed_params["gamma"]
    if param_to_vary == "Infection Rate β":
        b = val
    elif param_to_vary == "Incubation Rate σ":
        s = val
    elif param_to_vary == "Recovery Rate γ":
        g = val
    mean_res, _, _ = simulate_ensemble(3, n_nodes, steps, b, s, g, network, neighbors, policy, seed)
    total_infected = mean_res[-1,2]  # final infected count (I)
    results.append(total_infected)

sens_df = pd.DataFrame({"Parameter Value": vary_vals, "Final Infected (I)": results})

fig_sens = px.line(sens_df, x="Parameter Value", y="Final Infected (I)",
                   title=f"Sensitivity of Final Infected to {param_to_vary}")
st.plotly_chart(fig_sens, use_container_width=True)

# -----------------------------
# Chemical Engineering Mapping Explanation
# -----------------------------
if preset == "Chemical Engineering":
    st.header("⚗️ Chemical Engineering Applications")
    st.markdown("""
This quantum-inspired probabilistic SEIR model framework can be adapted to simulate:

- **Reaction-Diffusion Systems:** Model probabilistic states of reacting species within catalysts or porous media.
- **Membrane Processes:** Capture dynamics of particle adsorption, fouling, or selective transport on a network of membrane pores.
- **Industrial Process Safety:** Evaluate propagation of failures or contamination in interconnected process units.

The network topology can represent physical or chemical connectivity (e.g., pore networks or catalyst grain contact), and probabilistic states represent partial conversion, adsorption, or contamination levels.

By adjusting parameters analogous to reaction rates, diffusion coefficients, and recovery/removal rates, this model helps predict system-wide behaviors under interventions (catalyst regeneration, cleaning protocols, etc.).
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("© 2025 Samy Kouidri — Powered by Streamlit and NetworkX")
