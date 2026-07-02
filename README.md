# Edge Resource Allocation Under Dynamic Workloads

## Overview

This project presents an interactive **Streamlit-based simulation platform** for optimizing resource allocation in edge computing environments under dynamic workloads. The simulator models task execution across multiple edge nodes and compares traditional scheduling strategies with a reinforcement learning-based scheduler to evaluate system performance.

The dashboard provides real-time visualization of task allocation, resource utilization, latency, and scheduling efficiency, allowing users to analyze how different scheduling policies perform under changing workloads.

---

## Features

* Interactive Streamlit dashboard
* Dynamic workload generation
* Edge node resource allocation simulation
* Reinforcement learning-based scheduling
* Comparison with traditional scheduling algorithms
* Real-time performance visualization
* Resource utilization monitoring
* Task scheduling statistics and analytics

---

## Technologies Used

* Python
* Streamlit
* PyTorch
* NumPy
* Pandas
* Plotly

---

## Project Structure

```text
edge-resource-allocation/
│
├── streamlit_app.py      # Main Streamlit application
├── simulation.py         # Simulation engine
├── environment.py        # Edge environment definition
├── rl_agent.py           # Reinforcement Learning scheduler
├── models.py             # RL model architecture
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/edge-resource-allocation.git
cd edge-resource-allocation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
python -m streamlit run streamlit_app.py
```

The dashboard will open automatically in your default web browser.

---

## Simulation Metrics

The simulator evaluates several important performance metrics, including:

* Resource utilization
* Task completion rate
* Average task latency
* Scheduler efficiency
* Node workload distribution
* Overall system performance

---

## Future Improvements

* Multi-agent reinforcement learning
* Energy-aware resource scheduling
* Kubernetes integration
* Real edge device deployment
* Additional scheduling algorithms for comparison
* Support for larger-scale distributed simulations

---

## License

This project is intended for educational and research purposes.

---

## Author

Developed as part of an academic project on edge computing and intelligent resource allocation.
