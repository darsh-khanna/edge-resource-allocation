from environment import Task, Node
from typing import List


class BasePolicy:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
    def select(self, task):
        raise NotImplementedError
    def uses_edf_queue(self) -> bool:
        return False


class RoundRobin(BasePolicy):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.i = 0
    def select(self, task):
        node = self.nodes[self.i % len(self.nodes)]
        self.i += 1
        return node


class LeastLoaded(BasePolicy):
    def select(self, task):
        # workload()/load() are always 0 in this online sim (the queue drains
        # the instant a task is pushed), so "current backlog" has to be read
        # from how much longer the node is still busy: max(0, n.time - arrival).
        # Deliberately excludes task.size (that's WeightedLeastLoaded's job).
        return min(
            self.nodes,
            key=lambda n: max(0.0, n.time - task.arrival) / max(n.base_speed, 1e-6)
        )


class WeightedLeastLoaded(BasePolicy):
    def select(self, task):
        return min(self.nodes, key=lambda n: max(n.time, task.arrival) + (task.size / max(n.base_speed, 1e-6)))


class EDF(BasePolicy):
    def uses_edf_queue(self):
        return True
    def select(self, task):
        def score(n):
            finish = n.estimated_finish(task)
            slack = task.deadline - (finish - task.arrival)
            if slack >= 0:
                return -slack
            return 1000.0 + abs(slack)
        return min(self.nodes, key=score)


class ShortestJobFirst(BasePolicy):
    def select(self, task):
        return min(self.nodes, key=lambda n: n.estimated_finish(task))


class WeightedLeastConnection(BasePolicy):
    def select(self, task):
        # n.load() (queue length) is always 0 for the same reason as above.
        # "Connections" here has to mean cumulative tasks routed to this node
        # so far, which total_tasks_processed tracks correctly (each push is
        # processed synchronously, so it's an accurate running count).
        return min(
            self.nodes,
            key=lambda n: (n.metrics.total_tasks_processed + 1) / max(n.base_speed, 1e-6)
        )


class DeadlineAwareFastestNode(BasePolicy):
    def uses_edf_queue(self):
        return True
    def select(self, task):
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
    def select(self, task):
        # sum(n.workload()) is always 0 in this online sim, so this branch
        # never fired. Real congestion signal: average backlog time, i.e.
        # how much longer each node is still busy right now.
        # Thresholds calibrated against actual backlog distributions:
        # LIGHT ~0.03, MIXED ~0.8, HEAVY ~50 (mean, seed=42, 200 tasks/node=8).
        avg_load = sum(max(0.0, n.time - task.arrival) for n in self.nodes) / max(len(self.nodes), 1)
        def score(n):
            finish = n.estimated_finish(task)
            slack = task.deadline - (finish - task.arrival)
            miss_penalty = max(0.0, -slack)
            if avg_load < 0.5:
                return finish
            elif avg_load < 8.0:
                return finish + 3.0 * miss_penalty
            else:
                return finish + 8.0 * miss_penalty
        return min(self.nodes, key=score)


class MostIdleNode(BasePolicy):
    def select(self, task):
        # Mirror of the fixed LeastLoaded backlog signal, maximized instead
        # of minimized, so this is a genuine opposite rather than a static
        # "always pick the fastest node" fallback.
        return max(
            self.nodes,
            key=lambda n: n.base_speed - (max(0.0, n.time - task.arrival) / max(n.base_speed, 1e-6))
        )


POLICIES = {
    "RoundRobin": RoundRobin, "LeastLoaded": LeastLoaded, "WeightedLeastLoaded": WeightedLeastLoaded,
    "EDF": EDF, "ShortestJobFirst": ShortestJobFirst, "WeightedLeastConnection": WeightedLeastConnection,
    "DeadlineAwareFastestNode": DeadlineAwareFastestNode, "HybridScheduler": HybridScheduler,
    "MostIdleNode": MostIdleNode,
}

def instantiate_policy(name, nodes):
    if name not in POLICIES:
        raise ValueError(f"Unknown policy: {name}")
    return POLICIES[name](nodes)