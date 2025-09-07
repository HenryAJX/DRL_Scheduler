# utils/replay.py
import random
import numpy as np
from collections import deque
from typing import Any, Tuple

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    @staticmethod
    def _extract_obs(x: Any) -> Any:
        """Extracts the observation array from a potential (obs, info) tuple."""
        return x[0] if isinstance(x, (tuple, list)) else x

    @staticmethod
    def _normalize_done(done: Any) -> bool:
        """Converts a (terminated, truncated) tuple into a single boolean."""
        return bool(any(done)) if isinstance(done, (tuple, list)) else bool(done)

    def push(self, state: Any, action: Any, reward: Any, next_state: Any, done: Any):
        """Stores a transition, automatically sanitizing environment outputs."""
        s = self._extract_obs(state)
        ns = self._extract_obs(next_state)
        d = self._normalize_done(done)
        self.buffer.append((s, action, reward, ns, d))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.asarray, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)