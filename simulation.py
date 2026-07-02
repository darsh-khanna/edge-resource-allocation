import numpy as np
from typing import Dict, List
from environment import Environment, Task
from models import BasePolicy


class SimulationResult:
    """Container for simulation metrics and statistics."""

    def __init__(self):
        self.tasks_completed = 0
        self.deadline_misses = 0
        self.latencies = []
        self.node_assignments = []
        self.node_loads = []
        self.total_time = 0.0
        self.execution_timeline = []

    @property
    def miss_rate(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return (self.deadline_misses / self.tasks_completed) * 100

    @property
    def mean_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0.0

    @property
    def median_latency(self) -> float:
        return np.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0.0

    @property
    def throughput(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.tasks_completed / self.total_time

    @property
    def utilization(self) -> float:
        if not self.node_loads:
            return 0.0
        return np.mean(self.node_loads) * 100

    @property
    def fairness(self) -> float:
        if not self.node_assignments:
            return 1.0

        s = sum(self.node_assignments)

        if s == 0:
            return 1.0

        return (s ** 2) / (
            len(self.node_assignments) *
            sum(x ** 2 for x in self.node_assignments)
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "tasks_completed": self.tasks_completed,
            "deadline_misses": self.deadline_misses,
            "miss_rate": self.miss_rate,
            "mean_latency": self.mean_latency,
            "median_latency": self.median_latency,
            "p95_latency": self.p95_latency,
            "throughput": self.throughput,
            "utilization": self.utilization,
            "fairness": self.fairness,
            "total_time": self.total_time,
        }


class SimulationEngine:
    """Main simulation executor."""

    @staticmethod
    def evaluate(
        policy: BasePolicy,
        env: Environment,
        tasks: List[Task],
        verbose: bool = False
    ) -> SimulationResult:
        """
        TRUE ONLINE scheduling simulation.

        Every task is:
            1. observed by the scheduler
            2. assigned using CURRENT node state
            3. executed immediately
            4. node timing updated immediately

        This preserves realistic congestion feedback.
        """

        env.reset()

        result = SimulationResult()
        result.node_assignments = [0] * len(env.nodes)

        for task in tasks:

            selected_node = policy.select(task)

            node_idx = env.nodes.index(selected_node)
            result.node_assignments[node_idx] += 1

            if policy.uses_edf_queue():
                selected_node.push_edf(task)
                task_result = selected_node.step_edf()
            else:
                selected_node.push(task)
                task_result = selected_node.step_fifo()

            if task_result:
                result.latencies.append(task_result["latency"])

                if task_result["miss"]:
                    result.deadline_misses += 1

                result.execution_timeline.append({
                    "task_id": task_result["task_id"],
                    "node_id": selected_node.id,
                    "start": task_result["start"],
                    "finish": task_result["finish"],
                    "latency": task_result["latency"],
                    "miss": task_result["miss"],
                    "deadline": task_result["deadline"]
                })

        result.tasks_completed = len(result.latencies)

        result.total_time = max(
            n.time for n in env.nodes
        ) if env.nodes else 0.0

        result.node_loads = [
            n.busy_time / max(result.total_time, 1e-6)
            for n in env.nodes
        ]

        if verbose:
            print(
                f"Tasks: {result.tasks_completed} | "
                f"Misses: {result.deadline_misses} "
                f"({result.miss_rate:.2f}%) | "
                f"Latency: {result.mean_latency:.2f}s | "
                f"Throughput: {result.throughput:.2f}"
            )

        return result

    @staticmethod
    def benchmark(
        policy: BasePolicy,
        env: Environment,
        regimes=["LIGHT", "MIXED", "HEAVY"],
        n_tasks_per_regime: int = 200,
        verbose: bool = False
    ):

        results = {}

        for regime in regimes:
            tasks = env.generate_tasks(n_tasks_per_regime, regime)

            if verbose:
                print(f"\n[{regime}] Running benchmark...")

            results[regime] = SimulationEngine.evaluate(
                policy,
                env,
                tasks,
                verbose=verbose
            )

        return results