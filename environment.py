import numpy as np
import random
import heapq
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Task:
    """Represents a computational task with timing and deadline constraints."""
    id: int
    arrival: float
    size: float
    deadline: float
    priority: float
    remaining: float = None
    
    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.size


@dataclass
class NodeMetrics:
    """Captures historical metrics of a node's performance."""
    total_tasks_processed: int = 0
    total_deadline_misses: int = 0
    total_latency: float = 0.0
    peak_queue_length: int = 0
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0


class Node:
    """Represents a compute node in the edge network."""
    
    def __init__(self, speed: float = 1.0, node_id: int = 0):
        self.base_speed = speed
        self.id = node_id
        self.queue = []
        self.time = 0.0
        self.busy_time = 0.0
        self.metrics = NodeMetrics()
        self.current_task = None
        self.task_history = []  # For visualization
    
    def reset(self):
        """Reset node state for a new simulation."""
        self.queue = []
        self.time = 0.0
        self.busy_time = 0.0
        self.current_task = None
        self.task_history = []
        self.metrics = NodeMetrics()
    
    def current_speed(self) -> float:
        """Stochastic speed variation at execution time."""
        return max(0.3, self.base_speed * np.random.uniform(0.85, 1.15))
    
    def _extract(self, item):
        """Extract task from heap tuple or direct reference."""
        return item[2] if isinstance(item, tuple) else item
    
    def workload(self) -> float:
        """Total remaining work in queue."""
        return sum(self._extract(t).remaining for t in self.queue)
    
    def load(self) -> int:
        """Number of tasks in queue."""
        return len(self.queue)
    
    def estimated_finish(self, task: Optional[Task] = None) -> float:
        """Deterministic estimate using base_speed for planning."""
        work = self.workload()
        if task:
            work += task.size
        return self.time + work / max(self.base_speed, 1e-6)
    
    def push(self, task: Task):
        """Add task to FIFO queue."""
        self.queue.append(task)
    
    def push_edf(self, task: Task):
        """Add task to EDF (earliest deadline first) priority queue."""
        heapq.heappush(self.queue, (task.deadline, id(task), task))
    
    def step_fifo(self) -> Optional[Dict]:
        """Process one task from FIFO queue."""
        if not self.queue:
            return None
        return self._process(self.queue.pop(0))
    
    def step_edf(self) -> Optional[Dict]:
        """Process one task from EDF queue."""
        if not self.queue:
            return None
        _, _, task = heapq.heappop(self.queue)
        return self._process(task)
    
    def _process(self, task: Task) -> Dict:
        """Execute a task with stochastic speed variation."""
        speed = self.current_speed()
        start = max(self.time, task.arrival)
        finish = start + task.size / speed
        latency = finish - task.arrival
        is_miss = latency > task.deadline
        
        # Update metrics
        self.busy_time += finish - start
        self.time = finish
        self.metrics.total_tasks_processed += 1
        self.metrics.total_latency += latency
        if is_miss:
            self.metrics.total_deadline_misses += 1
        self.metrics.peak_queue_length = max(self.metrics.peak_queue_length, len(self.queue))
        
        # Record for visualization
        self.task_history.append({
            'task_id': task.id,
            'start': start,
            'finish': finish,
            'latency': latency,
            'miss': is_miss,
            'deadline': task.deadline
        })
        
        return {
            "latency": latency,
            "miss": is_miss,
            "deadline": task.deadline,
            "task_id": task.id,
            "start": start,
            "finish": finish
        }


class Environment:
    """Simulates the edge computing environment with multiple nodes."""
    
    def __init__(self, n_nodes: int = 8, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        speeds = np.random.uniform(0.5, 2.0, n_nodes)
        self.nodes = [Node(float(s), i) for i, s in enumerate(speeds)]
        self.simulation_history = []
    
    def reset(self):
        """Reset all nodes for a new simulation."""
        for n in self.nodes:
            n.reset()
        self.simulation_history = []
    
    def generate_tasks(self, n_tasks: int, regime: str = "MIXED") -> List[Task]:
        """
        Generate task workload based on regime.
        
        LIGHT  - Small tasks, generous deadlines, slow arrivals
        MIXED  - Medium tasks with variance, moderate deadlines
        HEAVY  - Large tasks, tight deadlines, fast arrivals
        """
        tasks = []
        t = 0.0
        
        for i in range(n_tasks):
            if regime == "LIGHT":
                size = np.random.exponential(1.2)
                deadline = np.random.uniform(15, 35)
                gap = np.random.exponential(1.8)
            
            elif regime == "HEAVY":
                size = np.random.exponential(5)
                if np.random.rand() < 0.30:
                    size *= np.random.uniform(2, 4)
                deadline = np.random.uniform(3, 12)
                gap = np.random.exponential(0.35)
            
            else:  # MIXED
                size = np.random.exponential(3)
                if np.random.rand() < 0.20:
                    size *= np.random.uniform(2, 4)
                deadline = np.random.uniform(5, 20)
                gap = np.random.exponential(0.9)
            
            t += gap
            priority = 1.0 / max(deadline, 1e-3)
            
            tasks.append(Task(
                id=i,
                arrival=t,
                size=max(size, 0.01),
                deadline=deadline,
                priority=priority
            ))
        
        return tasks
    
    def get_system_state(self) -> Dict:
        """Get current state of entire system."""
        return {
            'total_tasks_processed': sum(n.metrics.total_tasks_processed for n in self.nodes),
            'total_deadline_misses': sum(n.metrics.total_deadline_misses for n in self.nodes),
            'avg_latency': np.mean([n.metrics.total_latency / max(n.metrics.total_tasks_processed, 1) 
                                   for n in self.nodes]),
            'node_loads': [n.load() for n in self.nodes],
            'node_speeds': [n.base_speed for n in self.nodes],
            'current_time': max(n.time for n in self.nodes) if self.nodes else 0.0
        }