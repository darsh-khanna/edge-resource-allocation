"""
Scheduling policies for task assignment to compute nodes.

Each policy implements a different strategy for load balancing and deadline handling.
"""

from environment import Task, Node
from typing import List
import numpy as np


class BasePolicy:
    """Base class for all scheduling policies."""
    
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
    
    def select(self, task):
        raise NotImplementedError

    def uses_edf_queue(self) -> bool:
        return False


class RoundRobin(BasePolicy):
    """
    Distributes tasks in cyclic order.
    
    ✓ Shines when: LIGHT regime, uniform task sizes
    ✓ Zero overhead — no per-task scoring
    ✓ Works well when load is balanced by nature
    """
    
    def __init__(self, nodes: List[Node]):
        super().__init__(nodes)
        self.i = 0
    
    def select(self, task: Task) -> Node:
        node = self.nodes[self.i % len(self.nodes)]
        self.i += 1
        return node


class LeastLoaded(BasePolicy):
    """
    Selects node with least work (queue depth weighted by speed).
    
    ✓ Shines when: LIGHT/MIXED regimes
    ✓ Handles heterogeneous node speeds well
    ✓ Does NOT include incoming task in estimate
    """
    
    def select(self, task: Task) -> Node:
        return min(
            self.nodes,
            key=lambda n: n.workload() / max(n.base_speed, 1e-6)
        )


class WeightedLeastLoaded(BasePolicy):
    """
    Selects node with earliest task completion (greedy).
    
    ✓ Classic greedy baseline
    ✓ Includes incoming task in finish time estimate
    ✓ Good for heterogeneous speeds and sizes
    """
    
    def select(self, task: Task) -> Node:
        return min(
            self.nodes,
            key=lambda n: max(n.time, task.arrival) + (task.size / max(n.base_speed, 1e-6))
        )


class EDF(BasePolicy):
    """
    Earliest Deadline First scheduling policy.
    
    ✓ Shines when: MIXED/HEAVY regimes with varied deadlines
    ✓ Routes to nodes that give maximum slack (deadline feasibility)
    ✓ Penalizes infeasible nodes to avoid cascading misses
    ✓ Uses EDF execution queue on nodes
    """
    
    def uses_edf_queue(self) -> bool:
        return True
    
    def select(self, task: Task) -> Node:
        def score(n):
            finish = n.estimated_finish(task)
            slack = task.deadline - (finish - task.arrival)
            
            if slack >= 0:
                return -slack  # Feasible: maximize slack
            return 1000.0 + abs(slack)  # Infeasible: penalize
        
        return min(self.nodes, key=score)


class ShortestJobFirst(BasePolicy):
    """
    Routes short tasks to nodes that complete them soonest.
    
    ✓ Shines when: MIXED regime with high size variance
    ✓ Clears queue faster by prioritizing small tasks
    ✓ Includes task.size in completion estimate (unlike LeastLoaded)
    """
    
    def select(self, task: Task) -> Node:
        return min(self.nodes, key=lambda n: n.estimated_finish(task))


class WeightedLeastConnection(BasePolicy):
    """
    Distributes based on active connection count weighted by speed.
    
    ✓ Shines when: Node speeds differ significantly
    ✓ Distributes task COUNT, not total work
    ✓ Fast nodes get proportionally more tasks
    """
    
    def select(self, task: Task) -> Node:
        return min(
            self.nodes,
            key=lambda n: (n.load() + 1) / max(n.base_speed, 1e-6)
        )


class DeadlineAwareFastestNode(BasePolicy):
    """
    EDF-aware scheduling with strong infeasibility penalty.
    
    ✓ Shines when: HEAVY regime with very tight deadlines
    ✓ Uses EDF execution queue to prioritize urgent tasks
    ✓ Applies stronger penalty to steer clear of hopeless nodes
    """
    
    def uses_edf_queue(self) -> bool:
        return True
    
    def select(self, task: Task) -> Node:
        best_node = None
        best_score = float('inf')
        
        for n in self.nodes:
            finish = n.estimated_finish(task)
            slack = task.deadline - (finish - task.arrival)
            
            if slack >= 0:
                score = -slack
            else:
                score = 2000.0 + abs(slack)
            
            if score < best_score:
                best_score = score
                best_node = n
        
        return best_node


class HybridScheduler(BasePolicy):
    """
    Adaptive scheduling that changes strategy based on system load.
    
    ✓ Light load  → optimize latency (like SJF)
    ✓ Medium load → balance latency + deadline feasibility
    ✓ Heavy load  → strongly penalize infeasible assignments
    ✓ Expected best all-rounder across regimes
    """
    
    def select(self, task: Task) -> Node:
        # Compute average system load
        avg_load = sum(n.workload() for n in self.nodes) / max(len(self.nodes), 1)
        
        def score(n):
            finish = n.estimated_finish(task)
            slack = task.deadline - (finish - task.arrival)
            miss_penalty = max(0.0, -slack)
            
            if avg_load < 3.0:
                # Light: pure latency optimization
                return finish
            elif avg_load < 10.0:
                # Medium: balance latency + deadline
                return finish + 3.0 * miss_penalty
            else:
                # Heavy: heavily penalize deadline misses
                return finish + 8.0 * miss_penalty
        
        return min(self.nodes, key=score)


class MostIdleNode(BasePolicy):
    """
    Simple strategy that picks the node with most idle capacity.
    
    ✓ Extreme opposite of LeastLoaded
    ✓ Useful for comparison/baseline
    """
    
    def select(self, task: Task) -> Node:
        return max(self.nodes, key=lambda n: n.base_speed - (n.workload() / max(n.base_speed, 1e-6)))


# Registry of all available policies
POLICIES = {
    "RoundRobin": RoundRobin,
    "LeastLoaded": LeastLoaded,
    "WeightedLeastLoaded": WeightedLeastLoaded,
    "EDF": EDF,
    "ShortestJobFirst": ShortestJobFirst,
    "WeightedLeastConnection": WeightedLeastConnection,
    "DeadlineAwareFastestNode": DeadlineAwareFastestNode,
    "HybridScheduler": HybridScheduler,
    "MostIdleNode": MostIdleNode,
}


def instantiate_policy(name: str, nodes: List[Node]) -> BasePolicy:
    """Instantiate a policy by name."""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICIES.keys())}")
    return POLICIES[name](nodes)