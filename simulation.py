import numpy as np
from typing import Dict, List
from environment import Environment, Task
from models import BasePolicy


class SimulationResult:
    def __init__(self):
        self.tasks_completed = 0
        self.deadline_misses = 0
        self.latencies = []
        self.node_assignments = []
        self.node_loads = []
        self.total_time = 0.0
        self.execution_timeline = []

    @property
    def miss_rate(self):
        if self.tasks_completed == 0:
            return 0.0
        return (self.deadline_misses / self.tasks_completed) * 100

    @property
    def mean_latency(self):
        return np.mean(self.latencies) if self.latencies else 0.0

    @property
    def throughput(self):
        if self.total_time <= 0:
            return 0.0
        return self.tasks_completed / self.total_time

    @property
    def utilization(self):
        if not self.node_loads:
            return 0.0
        return np.mean(self.node_loads) * 100

    @property
    def fairness(self):
        if not self.node_assignments:
            return 1.0
        s = sum(self.node_assignments)
        if s == 0:
            return 1.0
        return (s ** 2) / (len(self.node_assignments) * sum(x ** 2 for x in self.node_assignments))

    def to_dict(self):
        return {
            "tasks_completed": self.tasks_completed, "deadline_misses": self.deadline_misses,
            "miss_rate": self.miss_rate, "mean_latency": self.mean_latency,
            "throughput": self.throughput, "utilization": self.utilization,
            "fairness": self.fairness, "total_time": self.total_time,
        }


class SimulationEngine:
    @staticmethod
    def evaluate(policy, env, tasks, verbose=False):
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
                    "task_id": task_result["task_id"], "node_id": selected_node.id,
                    "start": task_result["start"], "finish": task_result["finish"],
                    "latency": task_result["latency"], "miss": task_result["miss"],
                    "deadline": task_result["deadline"]
                })

        result.tasks_completed = len(result.latencies)
        result.total_time = max(n.time for n in env.nodes) if env.nodes else 0.0
        result.node_loads = [n.busy_time / max(result.total_time, 1e-6) for n in env.nodes]
        return result