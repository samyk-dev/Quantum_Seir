# Quantum-Inspired SEIR Simulator for Networked and Chemical Systems

This project implements a **probabilistic node-network model** inspired by quantum principles to simulate **SEIR dynamics** and **chemical analogs**. Nodes exist in **superposition across compartments** (`S, E, I, R`), enabling efficient uncertainty quantification, stochastic variability, and realistic networked interactions. Originally developed as a midterm project, the model was refined and expanded into a full application with interactive simulation and analysis features.

---

## Table of Contents

1. [Quick Start Guide](#-quick-start-guide)
2. [Background](#background)
3. [Reaction Kinetics Analogy](#reaction-kinetics-analogy)
4. [Quantum-Inspired Approach](#quantum-inspired-approach)
5. [Why It Matters](#why-it-matters)
6. [How It Works](#how-it-works)
7. [Features](#features)
8. [Acknowledgements](#acknowledgements)

---

## 📋 Quick Start Guide

### 🌐 Web Version

Ready to dive right in? Access the live web application instantly:

**[Access the Project Here](https://quantumseir.streamlit.app/)**

Click above to start using the **Quantum-Inspired SEIR simulator** immediately in your browser!

---

### 💻 Local Installation

Prefer to run the program locally? Follow these steps:

**Prerequisites:**

* Python 3.7+
* pip (Python package manager)
* The following Python libraries:
    - streamlit
    - numpy
    - pandas
    - plotly
    - networkx
    - scipy

**Installation:**

1. Clone or download the quantum_seir.py repository to your local machine

2. Install required dependencies:

```bash
pip install streamlit numpy pandas plotly networkx scipy
```

**Run the application:**

```bash
streamlit run quantum_seir.py
```

Open your browser to the local URL shown in the terminal (typically `http://localhost:8501`) to start interacting with the simulation.

---

## Background

The classical **SEIR model** is a [compartmental model](https://en.wikipedia.org/wiki/Compartmental_models_%28epidemiology%29) in epidemiology, dividing populations into:

* S – Susceptible
* E – Exposed (infected but not yet infectious)
* I – Infected (actively transmitting)
* R – Recovered/Removed

Transitions follow S → E → I → R with rates β (infection), σ (incubation), and γ (recovery).

This project extends SEIR modeling by representing each node as a **probability vector** `[S, E, I, R]`, capturing uncertainty directly at the node level. Nodes interact via **network structures**, where the **infection pressure** governs transitions from S → E:

* In **networked mode**, a node’s infection probability is proportional to the mean infected fraction among its neighbors.
* In **global mode**, infection pressure is the mean infected fraction across all nodes.

This mechanism allows local structure or coupling to influence transmission dynamically. Interventions such as lockdowns (reducing β), vaccinations (shifting nodes to R), or chemical process modifications are applied directly to probabilities at each step.

---

## Reaction Kinetics Analogy

The same probabilistic SEIR framework maps naturally to chemical reaction networks:

| Epidemiology          | Chemistry             |
| --------------------- | --------------------- |
| S – Susceptible       | Reactant (A)          |
| E – Exposed           | Intermediate (B)      |
| I – Infected          | Product (C)           |
| R – Recovered/Removed | Removed/Byproduct (D) |

Transition rates correspond to chemical kinetics:

* β → reaction or mixing rate
* σ → conversion from intermediate to product
* γ → removal, degradation, or byproduct formation

Infection pressure in epidemiology is analogous to **reaction pressure**: the rate at which a reactant converts to product depends on neighboring concentrations (or catalytic influence). Networked interactions represent coupled reactors, spatially distributed zones, or catalytic surfaces, whereas global interactions simulate perfectly mixed reactors. This enables **probabilistic reaction kinetics modeling** where node concentrations exist in superposition, capturing stochasticity and localized effects.

---

## Quantum-Inspired Approach

Nodes are represented as **probabilistic superstates** across `[S, E, I, R]` rather than discrete compartments. Combined with network-based infection/reaction pressure, this approach embeds uncertainty directly, reduces ensemble size requirements, and captures realistic local and global dynamics. The method is computationally efficient and applies to epidemics, chemical reaction networks, and other complex compartmental systems with heterogeneity.

---

## Why It Matters

This framework enables:

* **Realistic uncertainty modeling** at the node level
* **Network-aware dynamics** reflecting local connectivity
* **Interactive exploration** of interventions and parameter changes
* **Decision support** using infection/reaction pressure and uncertainty bands
* **Dual applicability**: epidemiology, chemical engineering, industrial systems

---

## How It Works

1. Initialize nodes with `[S, E, I, R]` probabilities
2. Construct network of local or global interactions
3. Compute **infection/reaction pressure** for each node:
   * Networked mode: mean infected/converted fraction among neighbors
   * Global mode: mean infected/converted fraction across all nodes
4. Update states using effective rates (β, σ, γ) modified by interventions
5. Apply policies such as lockdowns, vaccinations, or chemical modifications
6. Run ensembles to generate **uncertainty bands**
7. Perform sensitivity analysis on key parameters
8. Visualize outputs with interactive plots

---

## Features

* Probabilistic **superstate node model**
* **Infection/Reaction pressure** drives local transitions dynamically
* Network-aware interactions for heterogeneous or spatially structured systems
* Policy/intervention module (lockdowns, vaccinations, catalysts)
* Ensemble simulations producing uncertainty bands
* Sensitivity analysis for β, σ, γ
* Interactive **Streamlit interface** for real-time exploration
* Mapping between epidemiological compartments and chemical species

---

## Acknowledgements

Special thanks to Prof. Mark Stoykovich for guidance, inspiration, and the midterm project idea, and to my classmates from MENG 212 for their support, collaboration, and valuable discussions.
