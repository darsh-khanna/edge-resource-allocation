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

# Custom CSS for enhanced aesthetics
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
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("## ⚙️ Configuration")

mode = st.sidebar.radio(
    "Select Mode",
    ["🎯 Single Policy", "📊 Policy Comparison", "🤖 RL Agent"],
    label_visibility="collapsed"
)

n_nodes = st.sidebar.slider(
    "Number of Compute Nodes",
    min_value=4,
    max_value=16,
    value=8,
    step=1
)

n_tasks = st.sidebar.slider(
    "Tasks per Regime",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

# Random seed for reproducibility
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

def get_environment(n_nodes: int, seed: int):
    """Create a fresh environment each time (nodes must not be shared across runs)."""
    return Environment(n_nodes=n_nodes, seed=seed)


def create_metric_card(label: str, value: str, suffix: str = ""):
    """Display a metric card."""
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


def plot_metrics_comparison(all_results: Dict, regimes: List[str]):
    """Create comprehensive metrics comparison visualization."""
    metrics = {
        "miss_rate": ("Miss Rate", "%"),
        "mean_latency": ("Mean Latency", "s"),
        "throughput": ("Throughput", "T/s"),
        "fairness": ("Fairness", "Jain's Index"),
        "utilization": ("Utilization", "%"),
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"<b>{title}</b>" for title, _ in metrics.values()],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, None]]
    )
    
    colors = {
        "RoundRobin": "#aec7e8",
        "EDF": "#ffbb78",
        "WeightedLeastLoaded": "#98df8a",
        "ShortestJobFirst": "#ff9896",
        "HybridScheduler": "#c5b0d5",
        "RL-Agent": "#d62728",
    }
    
    policy_names = list(all_results.keys())
    
    for metric_key, (metric_title, suffix) in list(metrics.items())[:5]:
        row = (list(metrics.keys()).index(metric_key) // 3) + 1
        col = (list(metrics.keys()).index(metric_key) % 3) + 1
        
        for policy in policy_names:
            values = [all_results[policy][regime].to_dict()[metric_key] for regime in regimes]
            
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=values,
                    name=policy,
                    marker_color=colors.get(policy, "#cccccc"),
                    showlegend=(row == 1 and col == 1),
                    hovertemplate=f"<b>{policy}</b><br>" +
                                  f"{metric_title}: %{{y:.2f}}{suffix}<extra></extra>"
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="<b>Comprehensive Policy Comparison</b>",
        title_font_size=18,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.75
        ),
        height=700,
        template="plotly_dark",
        hovermode="x unified"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    
    return fig


def plot_timeline(result: SimulationResult, title: str = "Task Execution Timeline"):
    """Visualize task execution timeline on nodes."""
    if not result.execution_timeline:
        st.warning("No execution timeline available")
        return None
    
    timeline_df = pd.DataFrame(result.execution_timeline)
    
    fig = px.timeline(
        timeline_df,
        x_start="start",
        x_end="finish",
        y="node_id",
        color="miss",
        color_discrete_map={True: "#d62728", False: "#00d9ff"},
        labels={"node_id": "Node", "miss": "Deadline Miss"},
        title=f"<b>{title}</b>"
    )
    
    fig.update_layout(
        height=400,
        template="plotly_dark",
        hovermode="closest"
    )
    
    return fig


def plot_latency_distribution(result: SimulationResult):
    """Plot latency distribution."""
    if not result.latencies:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=result.latencies,
        nbinsx=30,
        name="Latencies",
        marker_color="#00d9ff",
        opacity=0.7,
        hovertemplate="<b>Latency Range</b><br>%{x:.2f}s<br>Count: %{y}<extra></extra>"
    ))
    
    fig.add_vline(
        x=result.mean_latency,
        line_dash="dash",
        line_color="#ff006e",
        annotation_text=f"Mean: {result.mean_latency:.2f}s",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="<b>Latency Distribution</b>",
        xaxis_title="Latency (seconds)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400,
        showlegend=True,
        hovermode="x"
    )
    
    return fig


def plot_node_utilization(result: SimulationResult, n_nodes: int):
    """Plot node utilization bar chart."""
    utilizations = result.node_loads
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Node {i}" for i in range(len(utilizations))],
            y=[u * 100 for u in utilizations],
            marker_color="#00d9ff",
            text=[f"{u*100:.1f}%" for u in utilizations],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Utilization: %{y:.1f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="<b>Node Utilization</b>",
        xaxis_title="Compute Nodes",
        yaxis_title="Utilization (%)",
        template="plotly_dark",
        height=400,
        showlegend=False
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
        policy_name = st.selectbox(
            "Select Policy",
            list(POLICIES.keys()),
            help="Choose a scheduling policy to evaluate"
        )
    
    with col2:
        regime = st.selectbox(
            "Load Regime",
            ["LIGHT", "MIXED", "HEAVY"],
            help="LIGHT: Generous deadlines | MIXED: Moderate | HEAVY: Tight deadlines"
        )
    
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
        
        # Metrics cards
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
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_latency_distribution(result), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_node_utilization(result, n_nodes), use_container_width=True)
        
        # Timeline
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
            "Select Policies to Compare",
            list(POLICIES.keys()),
            default=[
                "RoundRobin",
                "WeightedLeastLoaded",
                "EDF",
                "HybridScheduler"
            ],
            help="Choose multiple policies for comparison"
        )
    
    with col2:
        compare_regimes = st.multiselect(
            "Load Regimes",
            ["LIGHT", "MIXED", "HEAVY"],
            default=["LIGHT", "MIXED", "HEAVY"]
        )
    
    if st.button("▶ Run Comparison", key="compare_run", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run simulations — each policy+regime gets a fresh environment so
        # node state from one run never bleeds into another.
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
        
        # Main comparison chart
        st.plotly_chart(plot_metrics_comparison(all_results, compare_regimes), use_container_width=True)
        
        # Detailed metrics table
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
        
        # Reorder columns
        cols_order = ["Policy", "Regime", "miss_rate", "mean_latency", "throughput", "fairness", "utilization"]
        df_metrics = df_metrics[[c for c in cols_order if c in df_metrics.columns]]
        
        # Rename columns
        df_metrics.columns = ["Policy", "Regime", "Miss %", "Mean Lat (s)", "Throughput", "Fairness", "Util %"]
        
        st.dataframe(
            df_metrics.style.format({
                "Miss %": "{:.2f}",
                "Mean Lat (s)": "{:.2f}",
                "Throughput": "{:.2f}",
                "Fairness": "{:.3f}",
                "Util %": "{:.1f}"
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
    
    st.markdown("### Training RL Agent from Expert (EDF)")
    st.info(
        "The RL agent learns optimal scheduling policies through supervised learning from "
        "the EDF (Earliest Deadline First) expert policy. It's pre-trained on multiple regimes "
        "to develop a robust generalist scheduler."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        n_pretrain = st.slider(
            "Pretrain Episodes (imitation)",
            min_value=50,
            max_value=500,
            value=200,
            step=20
        )
    
    with col2:
        n_rl = st.slider(
            "RL Training Episodes",
            min_value=50,
            max_value=500,
            value=200,
            step=20
        )
    
    with col3:
        compare_with = st.multiselect(
            "Compare Against",
            ["RoundRobin", "WeightedLeastLoaded", "EDF", "HybridScheduler"],
            default=["EDF", "HybridScheduler"]
        )
    
    if st.button("▶ Train Agent & Evaluate", key="rl_train", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use a dedicated training environment — its nodes are owned by the agent
        # and must NOT be shared with evaluation runs (which call env.reset()).
        status_text.text("🤖 Initializing RL agent...")
        train_env = Environment(n_nodes=n_nodes, seed=seed)
        agent = SchedulerAgent(n_nodes, lr=3e-4)
        progress_bar.progress(5)
        
        # Phase 1: supervised pretraining from EDF expert
        status_text.text("📚 Phase 1 — Pre-training (imitation learning)...")
        with st.spinner("Pretraining in progress..."):
            agent.pretrain(train_env, episodes=n_pretrain, verbose=False)
        progress_bar.progress(40)
        
        # Phase 2: RL fine-tuning — this is what was missing before
        status_text.text("🧠 Phase 2 — RL fine-tuning...")
        with st.spinner("RL training in progress..."):
            agent.train_rl(train_env, episodes=n_rl, verbose=False)
        progress_bar.progress(75)
        
        # Evaluate agent — use fresh environments so node state is clean
        status_text.text("⚡ Evaluating on test regimes...")
        
        # Snapshot trained weights into an eval env with identical node layout
        eval_env = Environment(n_nodes=n_nodes, seed=seed)
        rl_policy = agent.as_policy(eval_env.nodes)
        all_results = {"RL-Agent": {}}
        
        for regime in ["LIGHT", "MIXED", "HEAVY"]:
            # generate_tasks + evaluate both call env.reset() internally,
            # so we need a separate env per evaluation to keep node refs valid.
            regime_env = Environment(n_nodes=n_nodes, seed=seed)
            rl_regime_policy = agent.as_policy(regime_env.nodes)
            tasks = regime_env.generate_tasks(n_tasks, regime=regime)
            all_results["RL-Agent"][regime] = SimulationEngine.evaluate(
                rl_regime_policy, regime_env, tasks, verbose=False
            )
        
        # Compare with baseline policies (each gets its own clean environment)
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
        
        # Show comparison
        st.markdown("## RL Agent vs Baselines")
        st.plotly_chart(plot_metrics_comparison(all_results, ["LIGHT", "MIXED", "HEAVY"]), use_container_width=True)
        
        # Training history (populated by train_rl, records per-episode total reward)
        st.markdown("## RL Training Progress")
        
        if agent.training_history:
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                y=agent.training_history,
                mode="lines",
                name="Total Reward",
                line=dict(color="#00d9ff", width=2),
            ))
            # Smoothed trend
            if len(agent.training_history) >= 10:
                window = max(len(agent.training_history) // 20, 5)
                smoothed = np.convolve(
                    agent.training_history,
                    np.ones(window) / window,
                    mode="valid"
                )
                fig_reward.add_trace(go.Scatter(
                    y=smoothed,
                    mode="lines",
                    name=f"Smoothed (w={window})",
                    line=dict(color="#ff006e", width=2, dash="dash"),
                ))
            
            fig_reward.update_layout(
                title="<b>RL Training — Total Reward per Episode</b>",
                xaxis_title="Episode",
                yaxis_title="Total Reward",
                template="plotly_dark",
                height=350,
                hovermode="x"
            )
            
            st.plotly_chart(fig_reward, use_container_width=True)
        else:
            st.info("No RL training history recorded (increase RL episodes to see the reward curve).")
        
        progress_bar.progress(100)
        status_text.text("✅ Agent training and evaluation complete!")


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