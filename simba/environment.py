"""SIMBA environment: budget, curriculum knowledge, CTI marketplace.

The reward semantics live in `group_outcome` and are deliberately tiny —
see simba/config.py's module docstring for why they make selective CTI
buying (and hence DRL) necessary.
"""

from typing import Dict, List, Optional


class SimbaEnvironment:

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.budget = float(cfg.init_budget)
        self.knowns: List[str] = list(cfg.knowns)
        self.g1s: List[str] = list(cfg.g1s)
        self.g2s: List[str] = list(cfg.g2s)
        self.bought: List[str] = []
        self.steps = 0
        self.ticks = 0
        self.episode_rewards: List[float] = []
        self.buys: List[tuple] = []          # (tick, label, price)
        self.wasted_buys = 0
        self.bankrupt = False

    # ------------------------------------------------------------ CTI market
    def quote(self, label: Optional[str]) -> Optional[float]:
        """Price to buy `label`'s CTI, or None when it is not on sale
        (Knowns, G1 pseudo zero-days, hard_g2s-blocklisted, unknown
        labels)."""
        if label in self.g2s and label not in self.cfg.hard_g2s:
            return self.cfg.price_of(label)
        return None

    def buy(self, label: Optional[str]) -> float:
        """Perform the epistemic action for a group whose majority true
        class is `label`. Returns the amount charged (positive)."""
        price = self.quote(label)
        if price is None:
            self.wasted_buys += 1
            return float(self.cfg.waste_buy_penalty)
        self.g2s.remove(label)
        self.knowns.append(label)
        self.bought.append(label)
        self.buys.append((self.ticks, label, price))
        return price

    # ------------------------------------------------------------- rewards
    def group_outcome(self, action: int, labels: List[str]) -> Dict[str, float]:
        """Reward of applying `action` to a group of flows with true
        classes `labels`. Returns a breakdown; 'total' is the budget delta.
        The CTI price is NOT included here (the caller charges it via
        buy(), since it needs the majority label anyway)."""
        cfg = self.cfg
        accepted = action in (0, 2)   # a CTI buy also lets the flows through
        service = damage = blocked_benign = 0.0
        for lab in labels:
            r = cfg.reward_of(lab)
            if accepted:
                if r > 0:
                    known = lab in self.knowns
                    service += r if known else cfg.unknown_accept_discount * r
                else:
                    damage += r
            else:
                if r > 0:
                    blocked_benign -= cfg.block_benign_scale * r
        total = service + damage + blocked_benign
        return {'total': total, 'service': service, 'damage': damage,
                'blocked_benign': blocked_benign}

    def apply(self, delta: float):
        self.budget += delta
        self.steps += 1
        self.episode_rewards.append(delta)
        if self.budget < self.cfg.min_budget:
            self.bankrupt = True   # sticky flag: "ever went below the line"

    def episode_ended(self, total_ticks: int) -> bool:
        if self.cfg.bankrupt_terminates and self.bankrupt:
            return True
        max_ticks = self.cfg.max_episode_ticks or total_ticks
        return self.ticks >= min(max_ticks, total_ticks)

    # ---------------------------------------------------------- POX compat
    @property
    def current_knowledge(self) -> Dict[str, List[str]]:
        """Live view in the shape FlowLogger expects (tiger_server wires
        `controller_brain.env.current_knowledge` into the flowstats
        listener to derive flow.zda/test_zda flags)."""
        return {'Knowns': self.knowns, 'G1s': self.g1s, 'G2s': self.g2s}

    # ------------------------------------------------------------- summary
    def summary(self) -> Dict[str, float]:
        return {
            'return': float(sum(self.episode_rewards)),
            'final_budget': self.budget,
            'steps': self.steps,
            'ticks': self.ticks,
            'n_buys': len(self.buys),
            'wasted_buys': self.wasted_buys,
            'bankrupt': float(self.bankrupt),
        }
