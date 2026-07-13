"""
RL Scheduler Agent
==================

WHY IMITATION ALONE CAN'T BEAT EDF (see analysis notes):
For a single task, deadline & arrival are constants shared by every node
being scored. Any greedy formula of the form `finish + w * miss_penalty`
is a monotonic function of `finish` alone, so its argmin is always
"whichever node finishes soonest" -- independent of w. This is why EDF,
ShortestJobFirst, DeadlineAwareFastestNode, and HybridScheduler all pick
the identical node, every time: they're the same ranking under different
names. Pure imitation of any of them caps out at that same ranking.

WHAT ACTUALLY BEATS IT:
Fairness and tail latency are *episode-level* properties -- they depend
on the sequence of choices across many tasks, not on any single greedy
step. A policy-gradient agent that is rewarded on the real, stochastic,
multi-task outcome (latency + miss penalty + a load-imbalance penalty)
can learn to occasionally break ties away from "always the fastest idle
node," something no single-task greedy formula can represent. That's the
lever this agent uses to outperform the greedy family on fairness while
matching it on miss-rate.

TWO-PHASE TRAINING:
  1. pretrain()  -- supervised imitation of the EDF-exact teacher, using
     the REAL stochastic simulator (push/step_fifo) to advance node clocks,
     so the training distribution of n.time matches what as_policy() sees
     at inference. (Previously this used a deterministic base_speed
     advance step, which drifted from the stochastic eval-time dynamics.)
  2. train_rl()  -- REINFORCE fine-tuning on top of the pretrained
     weights, directly optimizing latency + miss-rate + fairness on the
     actual simulator, with an entropy bonus to keep exploring.
"""

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import Environment, Node, Task


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """
    MLP policy network.
    - LayerNorm on first layer for stable training across varied input scales.
    - No Dropout: deterministic greedy inference (and clean log-probs for
      REINFORCE) are both essential.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# EDF-Exact Teacher (used only for pretraining, i.e. a warm start)
# ─────────────────────────────────────────────────────────────────────────────

class _EDFTeacher:
    """
    Replicates the EDF policy's select() logic exactly (workload() == 0 at
    decision time in this online sim, so estimated_finish reduces to
    n.time + task.size / n.base_speed).
    """

    def select(self, task: Task, nodes: List[Node]) -> Node:
        best_feasible:       Node  = None
        best_feasible_slack: float = -float('inf')
        best_infeasible:     Node  = None
        best_infeasible_lat: float =  float('inf')

        for n in nodes:
            finish = n.time + task.size / max(n.base_speed, 1e-6)
            slack  = task.deadline - (finish - task.arrival)
            lat    = finish - task.arrival

            if slack >= 0:
                if slack > best_feasible_slack:
                    best_feasible_slack = slack
                    best_feasible = n
            else:
                if lat < best_infeasible_lat:
                    best_infeasible_lat = lat
                    best_infeasible = n

        return best_feasible if best_feasible is not None else best_infeasible


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class SchedulerAgent:

    def __init__(self, n_nodes: int, lr: float = 3e-4):
        self.n_nodes    = n_nodes
        self.state_dim  = 2 + 4 * n_nodes
        self.action_dim = n_nodes

        self.device = torch.device("cpu")

        self.model     = PolicyNet(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Per-episode loss (pretrain) / mean reward (train_rl) — read by
        # Streamlit's training-history chart.
        self.training_history: List[float] = []

    # ─────────────────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────────────────

    def encode(self, task: Task, nodes: List[Node], current_time: float) -> np.ndarray:
        """
        Task features (2): size_norm, deadline_norm.
        Per-node features (4): slack_norm, feasible, wait_norm, speed_norm.
        (unchanged from the original design -- these never depended on the
        broken workload()/load() values, so no fix was needed here.)
        """
        task_feats = [
            float(np.clip(task.size     / 5.0,  0.0, 4.0)),
            float(np.clip(task.deadline / 20.0, 0.0, 4.0)),
        ]

        node_feats: List[float] = []
        for n in nodes:
            finish    = n.time + task.size / max(n.base_speed, 1e-6)
            slack     = task.deadline - (finish - current_time)
            wait      = max(0.0, n.time - current_time)
            feasible  = 1.0 if slack >= 0.0 else 0.0

            # tanh, not hard clip: whenever several nodes are idle (the
            # common case) their raw slack/deadline ratio is >=1 and a hard
            # np.clip(..., -1, 1) floors them all to an identical 1.0,
            # destroying the ranking the network needs to learn (verified:
            # this dropped held-out imitation accuracy to ~15%, barely above
            # the 12.5% random baseline for 8 nodes). tanh squashes to the
            # same (-1, 1) range but keeps relative order almost everywhere.
            node_feats.extend([
                float(np.tanh(slack / max(task.deadline, 1e-3))),
                feasible,
                float(np.clip(wait  / 10.0,                     0.0,  2.0)),
                float(np.clip(n.base_speed / 2.0,               0.0,  2.0)),
            ])

        return np.array(task_feats + node_feats, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Greedy action (inference — no randomness, no node mutation)
    # ─────────────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> int:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return int(torch.argmax(self.model(t), dim=1).item())

    # ─────────────────────────────────────────────────────────────────────
    # Pretrain — supervised imitation of _EDFTeacher, using REAL dynamics
    # ─────────────────────────────────────────────────────────────────────

    def pretrain(
        self,
        env: Environment,
        episodes: int = 300,
        verbose: bool = True,
        tasks_per_episode: int = 150,
    ) -> None:
        """
        Each episode:
          1. Reset env, pick a random regime.
          2. For every task: encode (state), ask the teacher for a label,
             then ACTUALLY execute the task on that node via push/step_fifo
             (real stochastic speed), so node.time evolves exactly the way
             it will at inference time under as_policy().
          3. One cross-entropy gradient step over all (state, label) pairs.
        """
        teacher   = _EDFTeacher()
        criterion = nn.CrossEntropyLoss()
        regimes   = ["LIGHT", "MIXED", "HEAVY"]

        if verbose:
            print(f"[RL] Pretraining for {episodes} episodes …")

        for ep in range(episodes):
            env.reset()
            regime = random.choice(regimes)
            tasks  = env.generate_tasks(tasks_per_episode, regime)

            states: List[np.ndarray] = []
            labels: List[int]        = []

            for task in tasks:
                state = self.encode(task, env.nodes, task.arrival)
                states.append(state)

                node = teacher.select(task, env.nodes)
                labels.append(env.nodes.index(node))

                # Real execution (stochastic speed) instead of a deterministic
                # base_speed clock-advance -- keeps train-time n.time in the
                # same distribution as eval-time n.time.
                node.push(task)
                node.step_fifo()

            s_t = torch.tensor(np.array(states), dtype=torch.float32,
                               device=self.device)
            a_t = torch.tensor(np.array(labels),  dtype=torch.long,
                               device=self.device)

            # Multiple gradient steps per episode batch -- a single step per
            # 150-sample batch (the original design) under-fits badly (~0.19
            # held-out accuracy, barely above the 0.125 random baseline for
            # 8 nodes). A few epochs over each episode's batch is enough to
            # actually drive the network toward the teacher's decision
            # boundary while still refreshing the data every episode.
            self.model.train()
            loss_val = 0.0
            for _ in range(4):
                self.optimizer.zero_grad()
                loss = criterion(self.model(s_t), a_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                loss_val = float(loss.item())
            self.training_history.append(loss_val)

            if verbose and (ep + 1) % 50 == 0:
                recent = self.training_history[-50:]
                print(f"  ep {ep+1:4d}/{episodes}  loss {np.mean(recent):.4f}"
                      f"  regime {regime}")

        if verbose:
            print("[RL] Pretraining complete.\n")

    # ─────────────────────────────────────────────────────────────────────
    # train_rl — REINFORCE fine-tuning on the real simulator
    # ─────────────────────────────────────────────────────────────────────

    def train_rl(
        self,
        env: Environment,
        episodes: int = 200,
        verbose: bool = False,
        tasks_per_episode: int = 150,
        lr: float = 1e-4,
        entropy_coef: float = 0.01,
        miss_penalty: float = 25.0,
        fairness_weight: float = 2.0,
    ) -> None:
        """
        On-policy REINFORCE with a moving-average baseline.

        Reward per task = -(latency) - miss_penalty*miss - fairness_weight*imbalance
        where imbalance measures how far this node's running assignment
        share is above the even split, so a run of ties toward the same
        node gets penalized even when each individual choice looked optimal.

        This reward is NOT expressible as a per-task greedy formula (it
        depends on the running counts across the whole episode), which is
        exactly the class of behavior the deterministic policies structurally
        cannot represent (see module docstring).
        """
        regimes = ["LIGHT", "MIXED", "HEAVY"]
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        reward_baseline = 0.0

        if verbose:
            print(f"[RL] Fine-tuning (REINFORCE) for {episodes} episodes …")

        for ep in range(episodes):
            env.reset()
            regime = random.choice(regimes)
            tasks = env.generate_tasks(tasks_per_episode, regime)

            log_probs = []
            entropies = []
            rewards = []
            counts = [0] * len(env.nodes)

            self.model.train()
            for task in tasks:
                state = self.encode(task, env.nodes, task.arrival)
                t = torch.tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
                logits = self.model(t)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                log_probs.append(dist.log_prob(action).squeeze(0))
                entropies.append(dist.entropy().squeeze(0))

                idx = int(action.item())
                node = env.nodes[idx]
                counts[idx] += 1

                node.push(task)
                res = node.step_fifo()

                total_assigned = sum(counts)
                share = counts[idx] / max(total_assigned, 1)
                even_share = 1.0 / len(env.nodes)
                imbalance = max(0.0, share - even_share)

                miss_pen = miss_penalty if res["miss"] else 0.0
                reward = -(res["latency"]) - miss_pen - fairness_weight * imbalance
                rewards.append(reward)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            reward_baseline = 0.9 * reward_baseline + 0.1 * float(rewards_t.mean())
            advantage = rewards_t - reward_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies).mean()

            loss = -(log_probs_t * advantage).sum() - entropy_coef * entropy_t

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            self.training_history.append(float(rewards_t.mean().item()))

            if verbose and (ep + 1) % 50 == 0:
                recent = self.training_history[-50:]
                print(f"  ep {ep+1:4d}/{episodes}  mean_reward {np.mean(recent):.3f}"
                      f"  regime {regime}")

        if verbose:
            print("[RL] Fine-tuning complete.\n")

    # ─────────────────────────────────────────────────────────────────────
    # Policy wrapper — PURE READ, no node mutation
    # ─────────────────────────────────────────────────────────────────────

    def as_policy(self, nodes: List[Node]):
        agent = self

        class RLPolicy:
            def select(self, task: Task) -> Node:
                state = agent.encode(task, nodes, task.arrival)
                return nodes[agent.act(state)]

            def uses_edf_queue(self) -> bool:
                return False

        return RLPolicy()

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.eval()