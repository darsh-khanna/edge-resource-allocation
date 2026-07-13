import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import Environment, Node, Task


class PolicyNet(nn.Module):
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


class _EDFTeacher:
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


class SchedulerAgent:

    def __init__(self, n_nodes: int, lr: float = 3e-4):
        self.n_nodes    = n_nodes
        self.state_dim  = 2 + 4 * n_nodes
        self.action_dim = n_nodes

        self.device = torch.device("cpu")

        self.model     = PolicyNet(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.training_history: List[float] = []

    def encode(self, task: Task, nodes: List[Node], current_time: float) -> np.ndarray:
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

            node_feats.extend([
                float(np.tanh(slack / max(task.deadline, 1e-3))),
                feasible,
                float(np.clip(wait  / 10.0,                     0.0,  2.0)),
                float(np.clip(n.base_speed / 2.0,               0.0,  2.0)),
            ])

        return np.array(task_feats + node_feats, dtype=np.float32)

    def act(self, state: np.ndarray) -> int:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return int(torch.argmax(self.model(t), dim=1).item())

    def pretrain(
        self,
        env: Environment,
        episodes: int = 300,
        verbose: bool = True,
        tasks_per_episode: int = 150,
    ) -> None:
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

                node.push(task)
                node.step_fifo()

            s_t = torch.tensor(np.array(states), dtype=torch.float32,
                               device=self.device)
            a_t = torch.tensor(np.array(labels),  dtype=torch.long,
                               device=self.device)

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
        regime_weights: tuple = (0.4, 0.4, 0.2),
        eval_every: int = 25,
        eval_seeds: tuple = (7, 42, 123),
    ) -> None:
        """
        regime_weights: sampling probs for (LIGHT, MIXED, HEAVY). HEAVY in this
        simulator is structurally over capacity (arrival rate ~2x total node
        throughput -- verified empirically), so ~90% of tasks miss their
        deadline *no matter which node they're routed to*. That makes HEAVY's
        per-task reward mostly noise (routing barely changes the outcome),
        while LIGHT/MIXED are under capacity and genuinely reward smart
        per-task tradeoffs. Sampling HEAVY less means fewer noisy gradient
        steps drowning out the learnable signal from the other two regimes.
        Default (0.4, 0.4, 0.2) downweights HEAVY vs. the naive uniform draw.

        eval_every / eval_seeds: every `eval_every` episodes, greedy-evaluate
        the current weights across all three regimes on `eval_seeds` (not the
        training seed) and checkpoint if the combined score improves. Without
        this, REINFORCE's variance means more episodes can make things worse,
        not better (observed empirically: full-budget training regressed
        HEAVY latency ~50% vs. a shorter run). At the end, the *best*
        checkpoint is restored, not just whatever the last episode produced.
        """
        regimes = ["LIGHT", "MIXED", "HEAVY"]
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_score = float('inf')
        best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        def _eval_score():
            """Lower is better: mean latency + a heavy penalty per miss,
            averaged over regimes and eval_seeds, using greedy (argmax) act()."""
            self.model.eval()
            total = 0.0
            n = 0
            with torch.no_grad():
                for sd in eval_seeds:
                    eval_env = Environment(n_nodes=len(env.nodes), seed=sd)
                    for regime in regimes:
                        eval_env.reset()
                        eval_tasks = eval_env.generate_tasks(tasks_per_episode, regime)
                        pol = self.as_policy(eval_env.nodes)
                        lat_sum, miss_sum, cnt = 0.0, 0, 0
                        for task in eval_tasks:
                            node = pol.select(task)
                            node.push(task)
                            res = node.step_fifo()
                            lat_sum += res["latency"]
                            miss_sum += 1 if res["miss"] else 0
                            cnt += 1
                        total += (lat_sum / max(cnt, 1)) + 5.0 * (miss_sum / max(cnt, 1))
                        n += 1
            self.model.train()
            return total / max(n, 1)

        if verbose:
            print(f"[RL] Fine-tuning (REINFORCE) for {episodes} episodes …")

        for ep in range(episodes):
            env.reset()
            regime = random.choices(regimes, weights=list(regime_weights))[0]
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

                # Continuous overrun-based penalty instead of a flat constant:
                # a miss by 0.1s and a miss by 60s used to cost the same
                # (miss_penalty), which is nearly a binary/constant signal
                # once a regime is saturated (most tasks miss regardless of
                # routing) -- the network had almost nothing to learn from.
                # Scaling by how much the deadline was blown gives a real
                # gradient even when misses are frequent.
                overrun = max(0.0, res["latency"] - res["deadline"])
                miss_pen = miss_penalty * (overrun / max(res["deadline"], 1e-3)) if res["miss"] else 0.0
                reward = -(res["latency"]) - miss_pen - fairness_weight * imbalance
                rewards.append(reward)

            # Per-episode standardization already centers and scales the
            # rewards, which makes any constant baseline subtraction
            # mathematically cancel out before it can have an effect (verified:
            # (r - b - mean(r - b))/std(r - b) == (r - mean(r))/std(r) for any
            # b) -- so the old EMA `reward_baseline` was dead code. Removed
            # rather than kept as misleading no-op.
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            advantage = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies).mean()

            loss = -(log_probs_t * advantage).sum() - entropy_coef * entropy_t

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            self.training_history.append(float(rewards_t.mean().item()))

            if (ep + 1) % eval_every == 0 or ep == episodes - 1:
                score = _eval_score()
                if score < best_score:
                    best_score = score
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if verbose:
                    print(f"  ep {ep+1:4d}/{episodes}  eval_score {score:.3f}"
                          f"  (best {best_score:.3f})")
            elif verbose and (ep + 1) % 50 == 0:
                recent = self.training_history[-50:]
                print(f"  ep {ep+1:4d}/{episodes}  mean_reward {np.mean(recent):.3f}"
                      f"  regime {regime}")

        # Restore the best-eval checkpoint rather than trusting the final
        # episode's weights -- REINFORCE variance means later episodes can
        # be worse, and this is the mechanism that actually prevents it.
        self.model.load_state_dict(best_state)
        self.model.eval()

        if verbose:
            print(f"[RL] Fine-tuning complete. Restored best checkpoint (eval_score {best_score:.3f}).\n")

    def as_policy(self, nodes: List[Node]):
        agent = self

        class RLPolicy:
            def select(self, task: Task) -> Node:
                state = agent.encode(task, nodes, task.arrival)
                return nodes[agent.act(state)]

            def uses_edf_queue(self) -> bool:
                return False

        return RLPolicy()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.eval()