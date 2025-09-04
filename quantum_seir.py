import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import base64

# -----------------------------
# Probabilistic Node Model with Network
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

def simulate_nodes(n_nodes, steps, beta, sigma, gamma, network=False, k=5, policy=None, rng=None):
    nodes = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(n_nodes)]
    nodes[0] = np.array([0.0, 0.0, 1.0, 0.0])  # seed infection

    # Build random network if enabled
    if network:
        neighbors = {i: rng.choice(n_nodes, size=k, replace=False) for i in range(n_nodes)}
    else:
        neighbors = None

    traj = []
    for t in range(steps):
        # Apply policies
        eff_beta = beta
        if policy:
            if policy["type"] == "lockdown" and t >= policy["start"]:
                eff_beta *= (1 - policy["reduction"])
            if policy["type"] == "vaccination" and t == policy["start"]:
                # Vaccinate fraction by shifting them into recovered
                frac = policy["coverage"]
                for i in rng.choice(n_nodes, size=int(frac*n_nodes), replace=False):
                    nodes[i] = np.array([0,0,0,1])

        # Infection pressure
        inf_pressures = []
        if network:
            for i in range(n_nodes):
                inf_pressures.append(np.mean([nodes[j][2] for j in neighbors[i]]))
        else:
            global_inf = np.mean([node[2] for node in nodes])
            inf_pressures = [global_inf]*n_nodes

        nodes = [update_node(node, eff_beta, sigma, gamma, inf_pressures[i]) for i,node in enumerate(nodes)]
        avg = np.mean(nodes, axis=0) * n_nodes
        traj.append([t, *avg])

    return pd.DataFrame(traj, columns=["Step", "S", "E", "I", "R"])

# -----------------------------
# Helper: Multiple runs for uncertainty bands
# -----------------------------
def simulate_ensemble(n_runs, n_nodes, steps, beta, sigma, gamma, network, k, policy, seed):
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_runs):
        df = simulate_nodes(n_nodes, steps, beta, sigma, gamma, network, k, policy, rng)
        runs.append(df[["S","E","I","R"]].values)
    runs = np.stack(runs)  # shape (n_runs, steps, 4)
    mean = runs.mean(axis=0)
    lo = np.percentile(runs, 5, axis=0)
    hi = np.percentile(runs, 95, axis=0)
    return mean, lo, hi

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Probabilistic SEIR Extensions", layout="wide")
st.title("Quantum-Inspired Probabilistic Modeling for Networked Systems")

# --------- Preset selector with dynamic defaults and compartments --------------
preset = st.sidebar.selectbox("Select Model Preset", ["Epidemiology (SEIR)", "Chemical Engineering"])

if preset == "Chemical Engineering":
    # Chemical kinetics inspired defaults
    default_beta = 0.1   # reaction/mixing rate
    default_sigma = 0.4  # conversion from intermediate to product
    default_gamma = 0.05 # removal/degradation rate
    compartments = {
        "S": "Reactant (A)",
        "E": "Intermediate (B)",
        "I": "Product (C)",
        "R": "Removed/Byproduct (D)"
    }
else:
    # Epidemiological defaults
    default_beta = 0.3
    default_sigma = 0.2
    default_gamma = 0.1
    compartments = {
        "S": "Susceptible",
        "E": "Exposed",
        "I": "Infected",
        "R": "Recovered"
    }

# Sidebar controls
st.sidebar.header("Simulation Controls")
n_nodes = st.sidebar.slider("Number of Nodes", 50, 2000, 500, step=50, help="Number of individuals or reactors in the simulation")
steps = st.sidebar.slider("Steps", 10, 500, 160, step=10, help="Number of time steps to simulate")

# Dynamic labels and tooltips based on preset
if preset == "Chemical Engineering":
    beta_label = "Reaction/Mixing Rate β"
    sigma_label = "Conversion Rate σ"
    gamma_label = "Removal Rate γ"
    beta_help = "Rate at which reactant converts to intermediate."
    sigma_help = "Rate at which intermediate converts to product."
    gamma_help = "Rate of product removal or degradation."
else:
    beta_label = "Infection Rate β"
    sigma_label = "Incubation Rate σ"
    gamma_label = "Recovery Rate γ"
    beta_help = "Rate of disease transmission."
    sigma_help = "Rate of progression from exposed to infectious."
    gamma_help = "Rate of recovery from infection."

beta = st.sidebar.slider(beta_label, 0.0, 2.0, default_beta, 0.01, help=beta_help)
sigma = st.sidebar.slider(sigma_label, 0.0, 1.0, default_sigma, 0.01, help=sigma_help)
gamma = st.sidebar.slider(gamma_label, 0.0, 1.0, default_gamma, 0.01, help=gamma_help)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42, help="Random seed for reproducibility")

network = st.sidebar.checkbox("Enable Network Structure", value=False, help="Model interaction via network connections instead of global mixing")

k = st.sidebar.slider("Avg Neighbors (if network on)", 1, 50, 5, help="Average number of neighbors each node is connected to in the network")

policy_type = st.sidebar.selectbox("Policy Intervention", ["None", "Lockdown", "Vaccination"])

policy = None
if policy_type == "Lockdown":
    start = st.sidebar.slider("Lockdown Start Step", 0, steps, 50, help="Time step at which lockdown policy starts")
    reduction = st.sidebar.slider("Transmission Reduction (%)", 0, 100, 50, help="Percent reduction in transmission rate during lockdown") / 100
    policy = {"type":"lockdown","start":start,"reduction":reduction}
elif policy_type == "Vaccination":
    start = st.sidebar.slider("Vaccination Start Step", 0, steps, 20, help="Time step when vaccination begins")
    coverage = st.sidebar.slider("Coverage (%)", 0, 100, 20, help="Fraction of population vaccinated at start") / 100
    policy = {"type":"vaccination","start":start,"coverage":coverage}

ensemble_runs = st.sidebar.slider("Ensemble Runs (for uncertainty bands)", 1, 50, 10, help="Number of repeated simulations to estimate uncertainty")

# -------------------------
# Simulation
# -------------------------
with st.spinner("Running simulations..."):
    start_time = time.perf_counter()
    mean, lo, hi = simulate_ensemble(ensemble_runs, n_nodes, steps, beta, sigma, gamma, network, k, policy, seed)
    runtime = time.perf_counter() - start_time

df = pd.DataFrame(mean, columns=["S","E","I","R"])
df["Step"] = np.arange(steps)
df_lo = pd.DataFrame(lo, columns=["S","E","I","R"])
df_hi = pd.DataFrame(hi, columns=["S","E","I","R"])
df_lo["Step"] = df_hi["Step"] = df["Step"]

# --------- Plot with uncertainty bands -----------
fig = go.Figure()
colors = {"S":"blue","E":"orange","I":"red","R":"green"}

for comp in ["S","E","I","R"]:
    # Main mean line
    fig.add_trace(go.Scatter(
        x=df["Step"],
        y=df[comp],
        mode="lines",
        name=f"{compartments[comp]} mean",
        line=dict(color=colors[comp], width=3),  
        opacity=0.9
    ))
    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=pd.concat([df["Step"], df["Step"][::-1]]),
        y=pd.concat([df_hi[comp], df_lo[comp][::-1]]),
        fill="toself",
        fillcolor=colors[comp],
        opacity=0.4,
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        hoverinfo="skip",
        name=f"{compartments[comp]} uncertainty"
    ))

fig.update_layout(
    title="Chemical Process Dynamics with Uncertainty Bands" if preset=="Chemical Engineering" else "SEIR Dynamics with Uncertainty Bands",
    xaxis_title="Time Step",
    yaxis_title="Quantity / Concentration" if preset=="Chemical Engineering" else "Population",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# -------- Final distribution pie chart ---------
final = df.iloc[-1][["S","E","I","R"]]

# Pie chart labels from compartments dict
pie_labels = [compartments[c] for c in ["S", "E", "I", "R"]]

pie = px.pie(
    values=final.values,
    names=pie_labels,
    title="Final Expected Distribution"
)
st.plotly_chart(pie, use_container_width=True)

# Insights section
st.subheader("📊 Insights")
st.markdown(f"- Peak {compartments['I']} (mean): {df['I'].max():.1f}")
st.markdown(f"- Runtime for {ensemble_runs} runs: {runtime*1000:.2f} ms")
st.markdown(f"- Random Seed: {seed}")

# Export results as CSV
csv = df.to_csv(index=False).encode()
b64 = base64.b64encode(csv).decode()
st.download_button("📥 Download Simulation Results (CSV)", data=csv, file_name="results.csv", mime="text/csv")

# -----------------------------
# Sensitivity Analysis
# -----------------------------
st.subheader("🔎 Parameter Sensitivity Analysis")
param_choice = st.selectbox("Vary Parameter", ["β","σ","γ"])

n_grid = st.slider("Grid Resolution", 3, 15, 5)

vals = []
p_range = np.linspace(0.1, 1.0, n_grid)
for val in p_range:
    kwargs = {"beta":beta,"sigma":sigma,"gamma":gamma}
    kwargs[param_choice.lower()] = val
    mean_sens, _, _ = simulate_ensemble(5, n_nodes, steps, kwargs["beta"], kwargs["sigma"], kwargs["gamma"], network, k, policy, seed)
    vals.append(mean_sens[-1,2])  # final infected/product/intermediate level

heat = pd.DataFrame({"param":p_range, "final_outcome":vals})

fig_heat = px.line(
    heat,
    x="param",
    y="final_outcome",
    markers=True,
    labels={"param":param_choice, "final_outcome":f"Final {compartments['I']}"},
    title=f"Sensitivity of Final {compartments['I']} to {param_choice}"
)
st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------------
# Explanation and context
# -----------------------------
st.markdown("---")
st.header("🔍 Model Explanation")

if preset == "Chemical Engineering":
    st.markdown("""
    This model simulates a **chemical reaction network** analogous to the SEIR epidemiological compartments:
    - **Reactant (S)**: Starting chemical species
    - **Intermediate (E)**: Transient species formed during reaction
    - **Product (I)**: Final desired product formed
    - **Removed (R)**: Byproducts or degraded species

    The reaction rates (β, σ, γ) control conversion speed, intermediate formation, and product degradation/removal respectively.

    Using a network structure models spatial or reactor-to-reactor interactions influencing reaction progress. The uncertainty bands demonstrate variability over multiple simulation runs due to stochastic effects or random network configurations.

    This approach helps chemical engineers visualize complex kinetics and the impact of spatial heterogeneity or policy-like interventions (e.g., shutdowns or catalyst additions).
    """)
else:
    st.markdown("""
    This model simulates the classical **SEIR epidemic model** with probabilistic nodes and network extensions.

    - **Susceptible (S)**: Individuals not yet exposed
    - **Exposed (E)**: Individuals exposed but not yet infectious
    - **Infected (I)**: Infectious individuals
    - **Recovered (R)**: Individuals recovered or removed

    Network structure allows modeling of transmission through specific contacts rather than assuming homogenous mixing. Policies like lockdowns or vaccinations reduce transmission or move individuals into recovered states.

    Uncertainty bands reflect stochastic variations across simulation runs, providing confidence intervals on outcomes.

    This quantum-inspired probabilistic model captures realistic transmission dynamics and the effect of interventions.
    """)

st.markdown("""
---
### Additional Notes:
- The **network structure** models localized interactions instead of uniform mixing, which is critical in many real-world systems.
- The **uncertainty bands** are key for understanding the variability and confidence in model predictions.
- Policy interventions are generalized mechanisms for controlling transmission or conversion.
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("© 2025 Samy Kouidri — Powered by Streamlit and NetworkX")
