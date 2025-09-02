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
st.title("🧮 Quantum-Inspired Probabilistic SEIR Model with Extensions")

# Sidebar controls
st.sidebar.header("Simulation Controls")
n_nodes = st.sidebar.slider("Number of Nodes", 50, 2000, 500, step=50)
steps = st.sidebar.slider("Steps", 10, 500, 160, step=10)
beta = st.sidebar.slider("Infection Rate β", 0.0, 2.0, 0.3, 0.01)
sigma = st.sidebar.slider("Incubation Rate σ", 0.0, 1.0, 0.2, 0.01)
gamma = st.sidebar.slider("Recovery Rate γ", 0.0, 1.0, 0.1, 0.01)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

network = st.sidebar.checkbox("Enable Network Structure", value=False)
k = st.sidebar.slider("Avg Neighbors (if network on)", 1, 50, 5)

policy_type = st.sidebar.selectbox("Policy Intervention", ["None", "Lockdown", "Vaccination"])
policy = None
if policy_type == "Lockdown":
    start = st.sidebar.slider("Lockdown Start Step", 0, steps, 50)
    reduction = st.sidebar.slider("Transmission Reduction (%)", 0, 100, 50)/100
    policy = {"type":"lockdown","start":start,"reduction":reduction}
elif policy_type == "Vaccination":
    start = st.sidebar.slider("Vaccination Start Step", 0, steps, 20)
    coverage = st.sidebar.slider("Coverage (%)", 0, 100, 20)/100
    policy = {"type":"vaccination","start":start,"coverage":coverage}

ensemble_runs = st.sidebar.slider("Ensemble Runs (for uncertainty bands)", 1, 50, 10)

# Simulation
start_time = time.perf_counter()
mean, lo, hi = simulate_ensemble(ensemble_runs, n_nodes, steps, beta, sigma, gamma, network, k, policy, seed)
runtime = time.perf_counter() - start_time

# Build dataframe for plotting
df = pd.DataFrame(mean, columns=["S","E","I","R"])
df["Step"] = np.arange(steps)
df_lo = pd.DataFrame(lo, columns=["S","E","I","R"])
df_hi = pd.DataFrame(hi, columns=["S","E","I","R"])
df_lo["Step"] = df_hi["Step"] = df["Step"]

# Plot with bands
fig = go.Figure()
colors = {"S":"blue","E":"orange","I":"red","R":"green"}
for comp in ["S","E","I","R"]:
    fig.add_trace(go.Scatter(x=df["Step"], y=df[comp], mode="lines", name=f"{comp} mean", line=dict(color=colors[comp])))
    fig.add_trace(go.Scatter(x=pd.concat([df["Step"], df["Step"][::-1]]),
                             y=pd.concat([df_hi[comp], df_lo[comp][::-1]]),
                             fill="toself", fillcolor=colors[comp], opacity=0.2,
                             line=dict(color="rgba(255,255,255,0)"), showlegend=False))
fig.update_layout(title="SEIR Dynamics with Uncertainty Bands", xaxis_title="Step", yaxis_title="Population")
st.plotly_chart(fig, use_container_width=True)

# Final distribution
final = df.iloc[-1][["S","E","I","R"]]
pie = px.pie(values=final.values, names=final.index, title="Final Expected Distribution")
st.plotly_chart(pie, use_container_width=True)

# Insights
st.subheader("📊 Insights")
st.markdown(f"- Peak Infected (mean): {df['I'].max():.1f}")
st.markdown(f"- Runtime for {ensemble_runs} runs: {runtime*1000:.2f} ms")
st.markdown(f"- Random Seed: {seed}")

# Export results
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
    mean, _, _ = simulate_ensemble(5, n_nodes, steps, kwargs["beta"], kwargs["sigma"], kwargs["gamma"], network, k, policy, seed)
    vals.append(mean[-1,2])  # final infected
heat = pd.DataFrame({"param":p_range, "final_infected":vals})

fig_heat = px.line(heat, x="param", y="final_infected", markers=True,
                   labels={"param":param_choice,"final_infected":"Final Infected"},
                   title=f"Sensitivity of Final Infected to {param_choice}")
st.plotly_chart(fig_heat, use_container_width=True)
