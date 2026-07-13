"""
EDGE NODE SCHEDULER — Interactive Simulation Dashboard

A beautiful, real-time visualization of task scheduling policies
on heterogeneous edge computing nodes under dynamic workloads.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List
import time

from environment import Environment
from models import instantiate_policy, POLICIES
from simulation import SimulationEngine, SimulationResult
from rl_agent import SchedulerAgent


# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="Edge Node Scheduler",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #00d9ff;
        --secondary: #ff006e;
        --background: #0a0e27;
        --surface: #1a1f3a;
        --text: #e0e0e0;
    }
    body {
        background-color: #0a0e27;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #252b48 100%);
        border-left: 4px solid #00d9ff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 217, 255, 0.1);
    }
    .header-title {
        background: linear-gradient(90deg, #00d9ff 0%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3em;
        letter-spacing: -1px;
    }
    [data-testid="stMetricValue"] {
        color: #00d9ff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FIXED RL HYPERPARAMETERS
# ============================================================================
# Tuned via sweep across fairness_weight x miss_penalty (see project notes).
# fairness_weight=2.0 gave the best latency AND best fairness among all
# tested RL configs; miss_penalty=25.0 was the best-performing pairing with
# it. No longer exposed as sliders since these are settled values, not
# something a user should need to retune per-session.
RL_FAIRNESS_WEIGHT = 2.0
RL_MISS_PENALTY = 25.0


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("## ⚙️ Configuration")

mode = st.sidebar.radio(
    "Select Mode",
    ["🎯 Single Policy", "📊 Policy Comparison", "🤖 RL Agent", "🔁 Multi-Run Average"],
    label_visibility="collapsed"
)

n_nodes = st.sidebar.slider("Number of Compute Nodes", min_value=4, max_value=16, value=8, step=1)
n_tasks = st.sidebar.slider("Tasks per Regime", min_value=50, max_value=500, value=200, step=50)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)


# ============================================================================
# MAIN HEADER
# ============================================================================

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="header-title">⚙️ EDGE NODE SCHEDULER</h1>', unsafe_allow_html=True)
    st.markdown("*Real-time task scheduling on heterogeneous edge compute nodes under dynamic workloads*")
with col2:
    st.metric("Active Nodes", n_nodes)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_metric_card(label: str, value: str, suffix: str = ""):
    return f"""
    <div class="metric-card">
        <p style="margin: 0; color: #888; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
            {label}
        </p>
        <p style="margin: 8px 0 0 0; color: #00d9ff; font-size: 1.8em; font-weight: 700;">
            {value} <span style="font-size: 0.8em; color: #666;">{suffix}</span>
        </p>
    </div>
    """


METRIC_DEFS = {
    "miss_rate": ("Miss Rate", "%"),
    "mean_latency": ("Mean Latency", "s"),
    "throughput": ("Throughput", "T/s"),
    "fairness": ("Fairness", "Jain's Index"),
    "utilization": ("Utilization", "%"),
}

POLICY_COLORS = {
    "RoundRobin": "#aec7e8", "EDF": "#ffbb78", "WeightedLeastLoaded": "#98df8a",
    "ShortestJobFirst": "#ff9896", "HybridScheduler": "#c5b0d5", "RL-Agent": "#d62728",
    "LeastLoaded": "#c49c94", "WeightedLeastConnection": "#f7b6d2",
    "DeadlineAwareFastestNode": "#dbdb8d", "MostIdleNode": "#9edae5",
}


def plot_metrics_comparison(all_results: Dict, regimes: List[str]):
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"<b>{title}</b>" for title, _ in METRIC_DEFS.values()],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, None]]
    )
    policy_names = list(all_results.keys())
    for idx, (metric_key, (metric_title, suffix)) in enumerate(METRIC_DEFS.items()):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        for policy in policy_names:
            values = [all_results[policy][regime].to_dict()[metric_key] for regime in regimes]
            fig.add_trace(
                go.Bar(
                    x=regimes, y=values, name=policy,
                    marker_color=POLICY_COLORS.get(policy, "#cccccc"),
                    showlegend=(row == 1 and col == 1),
                    hovertemplate=f"<b>{policy}</b><br>" + f"{metric_title}: %{{y:.2f}}{suffix}<extra></extra>"
                ),
                row=row, col=col
            )
    fig.update_layout(
        title_text="<b>Comprehensive Policy Comparison</b>", title_font_size=18, title_x=0.5,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=0.75),
        height=700, template="plotly_dark", hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def plot_metrics_comparison_avg(avg_results: Dict, std_results: Dict, regimes: List[str], n_runs: int):
    """
    Same 5-panel layout as plot_metrics_comparison, but built from pre-averaged
    plain-dict metrics (avg_results[policy][regime][metric_key] -> float) with
    optional std-dev error bars (std_results, same shape). Used by the
    Multi-Run Average tab where results are already aggregated across seeds
    rather than being single SimulationResult objects.
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"<b>{title}</b>" for title, _ in METRIC_DEFS.values()],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, None]]
    )
    policy_names = list(avg_results.keys())
    for idx, (metric_key, (metric_title, suffix)) in enumerate(METRIC_DEFS.items()):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        for policy in policy_names:
            values = [avg_results[policy][regime][metric_key] for regime in regimes]
            errors = [std_results[policy][regime][metric_key] for regime in regimes]
            fig.add_trace(
                go.Bar(
                    x=regimes, y=values, name=policy,
                    marker_color=POLICY_COLORS.get(policy, "#cccccc"),
                    showlegend=(row == 1 and col == 1),
                    error_y=dict(type="data", array=errors, visible=True, thickness=1, width=3),
                    hovertemplate=f"<b>{policy}</b><br>" + f"{metric_title}: %{{y:.2f}}{suffix} (±std)<extra></extra>"
                ),
                row=row, col=col
            )
    fig.update_layout(
        title_text=f"<b>Multi-Run Average Comparison (n={n_runs} runs/regime)</b>",
        title_font_size=18, title_x=0.5,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=0.75),
        height=700, template="plotly_dark", hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    return fig


def plot_timeline(result: SimulationResult, title: str = "Task Execution Timeline"):
    if not result.execution_timeline:
        st.warning("No execution timeline available")
        return None
    timeline_df = pd.DataFrame(result.execution_timeline)
    fig = px.timeline(
        timeline_df, x_start="start", x_end="finish", y="node_id", color="miss",
        color_discrete_map={True: "#d62728", False: "#00d9ff"},
        labels={"node_id": "Node", "miss": "Deadline Miss"}, title=f"<b>{title}</b>"
    )
    fig.update_layout(height=400, template="plotly_dark", hovermode="closest")
    return fig


def plot_latency_distribution(result: SimulationResult):
    if not result.latencies:
        return None
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=result.latencies, nbinsx=30, name="Latencies", marker_color="#00d9ff", opacity=0.7,
        hovertemplate="<b>Latency Range</b><br>%{x:.2f}s<br>Count: %{y}<extra></extra>"
    ))
    fig.add_vline(
        x=result.mean_latency, line_dash="dash", line_color="#ff006e",
        annotation_text=f"Mean: {result.mean_latency:.2f}s", annotation_position="top right"
    )
    fig.update_layout(
        title="<b>Latency Distribution</b>", xaxis_title="Latency (seconds)", yaxis_title="Frequency",
        template="plotly_dark", height=400, showlegend=True, hovermode="x"
    )
    return fig


def plot_node_utilization(result: SimulationResult, n_nodes: int):
    utilizations = result.node_loads
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Node {i}" for i in range(len(utilizations))],
            y=[u * 100 for u in utilizations], marker_color="#00d9ff",
            text=[f"{u*100:.1f}%" for u in utilizations], textposition="auto",
            hovertemplate="<b>%{x}</b><br>Utilization: %{y:.1f}%<extra></extra>"
        )
    ])
    fig.update_layout(
        title="<b>Node Utilization</b>", xaxis_title="Compute Nodes", yaxis_title="Utilization (%)",
        template="plotly_dark", height=400, showlegend=False
    )
    fig.update_yaxes(range=[0, 110])
    return fig


# ============================================================================
# MODE: SINGLE POLICY
# ============================================================================

if mode == "🎯 Single Policy":
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        policy_name = st.selectbox("Select Policy", list(POLICIES.keys()),
                                    help="Choose a scheduling policy to evaluate")
    with col2:
        regime = st.selectbox("Load Regime", ["LIGHT", "MIXED", "HEAVY"],
                               help="LIGHT: Generous deadlines | MIXED: Moderate | HEAVY: Tight deadlines")

    if st.button("▶ Run Simulation", key="single_run", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Initializing environment...")
        progress_bar.progress(20)
        env = Environment(n_nodes=n_nodes, seed=seed)

        status_text.text("📝 Generating tasks...")
        progress_bar.progress(40)
        tasks = env.generate_tasks(n_tasks, regime=regime)

        status_text.text("⚡ Running simulation...")
        progress_bar.progress(60)
        policy = instantiate_policy(policy_name, env.nodes)
        result = SimulationEngine.evaluate(policy, env, tasks, verbose=False)

        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(80)

        st.markdown("## Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(create_metric_card("Miss Rate", f"{result.miss_rate:.1f}", "%"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Mean Latency", f"{result.mean_latency:.2f}", "s"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Throughput", f"{result.throughput:.2f}", "T/s"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Fairness", f"{result.fairness:.3f}", ""), unsafe_allow_html=True)
        with col5:
            st.markdown(create_metric_card("Utilization", f"{result.utilization:.1f}", "%"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_latency_distribution(result), use_container_width=True)
        with col2:
            st.plotly_chart(plot_node_utilization(result, n_nodes), use_container_width=True)

        st.plotly_chart(plot_timeline(result, f"Task Execution Timeline — {policy_name}"), use_container_width=True)

        progress_bar.progress(100)
        status_text.text("✅ Simulation complete!")


# ============================================================================
# MODE: POLICY COMPARISON
# ============================================================================

elif mode == "📊 Policy Comparison":
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_policies = st.multiselect(
            "Select Policies to Compare", list(POLICIES.keys()),
            default=["RoundRobin", "WeightedLeastLoaded", "EDF", "HybridScheduler"],
            help="Choose multiple policies for comparison"
        )
    with col2:
        compare_regimes = st.multiselect("Load Regimes", ["LIGHT", "MIXED", "HEAVY"],
                                          default=["LIGHT", "MIXED", "HEAVY"])

    if st.button("▶ Run Comparison", key="compare_run", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = {}

        for idx, policy_name in enumerate(selected_policies):
            status_text.text(f"Running {policy_name}...")
            progress = 20 + (idx / len(selected_policies)) * 70
            progress_bar.progress(int(progress))

            regime_results = {}
            for regime in compare_regimes:
                regime_env = Environment(n_nodes=n_nodes, seed=seed)
                policy = instantiate_policy(policy_name, regime_env.nodes)
                tasks = regime_env.generate_tasks(n_tasks, regime=regime)
                regime_results[regime] = SimulationEngine.evaluate(policy, regime_env, tasks, verbose=False)
            all_results[policy_name] = regime_results

        progress_bar.progress(90)
        status_text.text("Creating comparison visualizations...")

        st.plotly_chart(plot_metrics_comparison(all_results, compare_regimes), use_container_width=True)

        st.markdown("## Detailed Results Table")
        metrics_list = []
        for policy_name in selected_policies:
            for regime in compare_regimes:
                result = all_results[policy_name][regime]
                metrics_dict = result.to_dict()
                metrics_dict["Policy"] = policy_name
                metrics_dict["Regime"] = regime
                metrics_list.append(metrics_dict)

        df_metrics = pd.DataFrame(metrics_list)
        cols_order = ["Policy", "Regime", "miss_rate", "mean_latency", "throughput", "fairness", "utilization"]
        df_metrics = df_metrics[[c for c in cols_order if c in df_metrics.columns]]
        df_metrics.columns = ["Policy", "Regime", "Miss %", "Mean Lat (s)", "Throughput", "Fairness", "Util %"]

        st.dataframe(
            df_metrics.style.format({
                "Miss %": "{:.2f}", "Mean Lat (s)": "{:.2f}", "Throughput": "{:.2f}",
                "Fairness": "{:.3f}", "Util %": "{:.1f}"
            }),
            use_container_width=True
        )

        progress_bar.progress(100)
        status_text.text("✅ Comparison complete!")


# ============================================================================
# MODE: RL AGENT
# ============================================================================

elif mode == "🤖 RL Agent":
    st.markdown("---")

    st.markdown("### Training RL Agent — Imitation Warm-Start + REINFORCE Fine-Tuning")


    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        n_pretrain = st.slider("Pretrain Episodes (imitation)", min_value=50, max_value=800, value=400, step=50)
    with col2:
        n_rl = st.slider("RL Fine-Tuning Episodes", min_value=0, max_value=800, value=400, step=50)
    with col3:
        compare_with = st.multiselect(
            "Compare Against", ["RoundRobin", "WeightedLeastLoaded", "EDF", "HybridScheduler"],
            default=["EDF", "WeightedLeastLoaded"]
        )

    st.caption(
        f"Fairness penalty weight and deadline-miss penalty weight are fixed at "
        f"**{RL_FAIRNESS_WEIGHT}** and **{RL_MISS_PENALTY}** respectively — tuned via sweep, "
        f"no longer exposed as sliders."
    )

    if st.button("▶ Train Agent & Evaluate", key="rl_train", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🤖 Initializing RL agent...")
        train_env = Environment(n_nodes=n_nodes, seed=seed)
        agent = SchedulerAgent(n_nodes, lr=1e-3)
        progress_bar.progress(5)

        status_text.text("📚 Phase 1 — Pre-training (imitation learning)...")
        with st.spinner("Pretraining in progress..."):
            agent.pretrain(train_env, episodes=n_pretrain, verbose=False, tasks_per_episode=150)
        progress_bar.progress(40)

        status_text.text("🧠 Phase 2 — RL fine-tuning (REINFORCE)...")
        with st.spinner("RL training in progress..."):
            agent.train_rl(
                train_env, episodes=n_rl, verbose=False, tasks_per_episode=150,
                lr=5e-5, fairness_weight=RL_FAIRNESS_WEIGHT, miss_penalty=RL_MISS_PENALTY
            )
        progress_bar.progress(75)

        status_text.text("⚡ Evaluating on test regimes...")

        all_results = {"RL-Agent": {}}
        for regime in ["LIGHT", "MIXED", "HEAVY"]:
            regime_env = Environment(n_nodes=n_nodes, seed=seed)
            rl_regime_policy = agent.as_policy(regime_env.nodes)
            tasks = regime_env.generate_tasks(n_tasks, regime=regime)
            all_results["RL-Agent"][regime] = SimulationEngine.evaluate(
                rl_regime_policy, regime_env, tasks, verbose=False
            )

        for policy_name in compare_with:
            all_results[policy_name] = {}
            for regime in ["LIGHT", "MIXED", "HEAVY"]:
                baseline_env = Environment(n_nodes=n_nodes, seed=seed)
                policy = instantiate_policy(policy_name, baseline_env.nodes)
                tasks = baseline_env.generate_tasks(n_tasks, regime=regime)
                all_results[policy_name][regime] = SimulationEngine.evaluate(
                    policy, baseline_env, tasks, verbose=False
                )

        progress_bar.progress(90)
        status_text.text("Creating comparison visualizations...")

        st.markdown("## RL Agent vs Baselines")
        st.plotly_chart(plot_metrics_comparison(all_results, ["LIGHT", "MIXED", "HEAVY"]), use_container_width=True)

        st.markdown("## RL Training Progress")
        if agent.training_history:
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                y=agent.training_history, mode="lines", name="Mean Reward / Loss",
                line=dict(color="#00d9ff", width=2),
            ))
            if len(agent.training_history) >= 10:
                window = max(len(agent.training_history) // 20, 5)
                smoothed = np.convolve(agent.training_history, np.ones(window) / window, mode="valid")
                fig_reward.add_trace(go.Scatter(
                    y=smoothed, mode="lines", name=f"Smoothed (w={window})",
                    line=dict(color="#ff006e", width=2, dash="dash"),
                ))
            fig_reward.update_layout(
                title="<b>Training Progress (pretrain loss, then RL mean reward)</b>",
                xaxis_title="Episode", yaxis_title="Value", template="plotly_dark",
                height=350, hovermode="x"
            )
            st.plotly_chart(fig_reward, use_container_width=True)
        else:
            st.info("No training history recorded.")

        progress_bar.progress(100)
        status_text.text("✅ Agent training and evaluation complete!")


# ============================================================================
# MODE: MULTI-RUN AVERAGE
# ============================================================================

elif mode == "🔁 Multi-Run Average":
    st.markdown("---")
    st.markdown("### Multi-Run Average Comparison")
    col1, col2 = st.columns([2, 1])
    with col1:
        avg_policies = st.multiselect(
            "Select Policies to Compare", list(POLICIES.keys()) + ["RL-Agent"],
            default=["RoundRobin", "EDF", "WeightedLeastLoaded", "RL-Agent"],
            help="Include RL-Agent to train it once, then evaluate it across all the runs below."
        )
    with col2:
        avg_regimes = st.multiselect("Load Regimes", ["LIGHT", "MIXED", "HEAVY"],
                                      default=["LIGHT", "MIXED", "HEAVY"], key="avg_regimes")

    col3, col4, col5 = st.columns(3)
    with col3:
        n_runs = st.slider("Number of Runs to Average", min_value=3, max_value=30, value=10, step=1)
    with col4:
        base_seed = st.number_input("Base Seed (run i uses base_seed + i)", value=1000, min_value=0)
    with col5:
        if "RL-Agent" in avg_policies:
            rl_episodes = st.slider("RL Train Episodes (pretrain + RL, each)", min_value=50, max_value=800,
                                     value=300, step=50)
        else:
            rl_episodes = None

    if st.button("▶ Run Multi-Run Comparison", key="avg_run", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        run_seeds = [int(base_seed) + i for i in range(n_runs)]

        # Train the RL agent once (if selected), on the sidebar's configured
        # training seed — separate from run_seeds so evaluation stays
        # held-out, matching how train_rl's own internal checkpointing works.
        agent = None
        if "RL-Agent" in avg_policies:
            status_text.text("🤖 Training RL agent once (shared across all runs)...")
            train_env = Environment(n_nodes=n_nodes, seed=seed)
            agent = SchedulerAgent(n_nodes, lr=1e-3)
            agent.pretrain(train_env, episodes=rl_episodes, verbose=False, tasks_per_episode=150)
            agent.train_rl(
                train_env, episodes=rl_episodes, verbose=False, tasks_per_episode=150,
                lr=5e-5, fairness_weight=RL_FAIRNESS_WEIGHT, miss_penalty=RL_MISS_PENALTY
            )
        progress_bar.progress(15)

        # raw_metrics[policy][regime][metric_key] -> list of per-run values
        raw_metrics = {p: {r: {m: [] for m in METRIC_DEFS} for r in avg_regimes} for p in avg_policies}

        total_steps = len(avg_policies) * len(avg_regimes) * n_runs
        step = 0
        for policy_name in avg_policies:
            for regime in avg_regimes:
                for run_seed in run_seeds:
                    status_text.text(f"Running {policy_name} — {regime} — seed {run_seed}...")
                    run_env = Environment(n_nodes=n_nodes, seed=run_seed)
                    if policy_name == "RL-Agent":
                        policy = agent.as_policy(run_env.nodes)
                    else:
                        policy = instantiate_policy(policy_name, run_env.nodes)
                    tasks = run_env.generate_tasks(n_tasks, regime=regime)
                    result = SimulationEngine.evaluate(policy, run_env, tasks, verbose=False)
                    result_dict = result.to_dict()
                    for m in METRIC_DEFS:
                        raw_metrics[policy_name][regime][m].append(result_dict[m])

                    step += 1
                    progress_bar.progress(15 + int(80 * step / total_steps))

        # Aggregate mean + std across runs
        avg_results = {p: {r: {} for r in avg_regimes} for p in avg_policies}
        std_results = {p: {r: {} for r in avg_regimes} for p in avg_policies}
        for policy_name in avg_policies:
            for regime in avg_regimes:
                for m in METRIC_DEFS:
                    vals = raw_metrics[policy_name][regime][m]
                    avg_results[policy_name][regime][m] = float(np.mean(vals))
                    std_results[policy_name][regime][m] = float(np.std(vals))

        status_text.text("📊 Creating visualizations...")
        st.plotly_chart(
            plot_metrics_comparison_avg(avg_results, std_results, avg_regimes, n_runs),
            use_container_width=True
        )

        st.markdown("## Averaged Results Table (mean ± std across runs)")
        table_rows = []
        for policy_name in avg_policies:
            for regime in avg_regimes:
                row = {"Policy": policy_name, "Regime": regime, "Runs": n_runs}
                for m, (title, suffix) in METRIC_DEFS.items():
                    mean_v = avg_results[policy_name][regime][m]
                    std_v = std_results[policy_name][regime][m]
                    row[title] = f"{mean_v:.2f} ± {std_v:.2f}"
                table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        progress_bar.progress(100)
        status_text.text(f"✅ Multi-run comparison complete! ({n_runs} runs/policy/regime)")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.85em;'>"
    "⚙️ Edge Node Scheduler • Multi-policy task scheduling optimization for heterogeneous edge computing"
    "</p>",
    unsafe_allow_html=True
)