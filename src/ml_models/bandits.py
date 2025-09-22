"""
Simple epsilon-greedy bandit to choose add_top_k for the lazy contingency callback.

Note:
- Reward here is heuristic (negative of constraints added). The goal is to keep the LP small
  while still progressing. In practice, other signals (gap improvement per time) are better,
  but require deeper integration with solver runtime stats.

Usage:
    policy = EpsilonGreedyTopK(K_list=[64, 128, 256], epsilon=0.1)
    # pass as config.topk_policy to attach_lazy_contingency_callback
"""

from __future__ import annotations

import random
from typing import List, Dict, Optional


class EpsilonGreedyTopK:
    def __init__(self, K_list: List[int], epsilon: float = 0.1, seed: int = 42):
        self.K_list = sorted([int(k) for k in K_list if int(k) >= 0])
        if not self.K_list:
            self.K_list = [0]
        self.epsilon = float(epsilon)
        self.rng = random.Random(int(seed))
        self._counts = {k: 1 for k in self.K_list}
        self._rewards = {k: 0.0 for k in self.K_list}

    def choose_k(self, context: Optional[Dict] = None) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.K_list)
        # Exploit: pick highest average reward
        avg = {k: (self._rewards[k] / max(1, self._counts[k])) for k in self.K_list}
        # Max reward
        best_k = max(self.K_list, key=lambda kk: avg[kk])
        return best_k

    def update(
        self, chosen_k: int, reward: float, context: Optional[Dict] = None
    ) -> None:
        if chosen_k not in self._counts:
            self._counts[chosen_k] = 0
            self._rewards[chosen_k] = 0.0
        self._counts[chosen_k] += 1
        self._rewards[chosen_k] += float(reward)
