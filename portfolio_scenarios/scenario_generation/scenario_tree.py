from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class ScenarioNode:
    def __init__(self, scenario_id, stage, branch_prob, parent=None, value=None):
        self.s = scenario_id
        self.t = stage
        self.p_n = branch_prob
        self.parent = parent
        self.children = []
        self.value = value

        self.path_prob = 1.0 if parent is None else parent.path_prob * branch_prob

    def add_child(self, node):
        self.children.append(node)
        return node

    def is_leaf(self):
        return not self.children

    def __repr__(self):
        v = f"  val={np.round(self.value[:3], 4)}..." if self.value is not None else ""
        return (
            f"Node(s={self.s}, t={self.t}, "
            f"p_n={self.p_n:.4f}, Pr_n={self.path_prob:.6f}{v})"
        )


class ScenarioTree:
    """
    Multistage stochastic scenario tree.

    Build by calling .build() with a generator and branching factors.
    All branching probabilities at each level are uniform: p_n = 1 / k.
    """

    def __init__(self):
        self.root = None
        self._count = 0

    def build(
        self,
        generator,
        branching_factors,
        fluctuation_range=(0.9, 1.1),
        initial_prices=None,
        is_tcvae=False,
        **kwargs,
    ):
        """
        Build the scenario tree.

        Args:
            generator:         Any generator exposing .sample(n, stage=s) or .sample(n).
            branching_factors: List of ints — one per stage. E.g. [100, 10].
            fluctuation_range: (low, high) used for non-TC-VAE stage ≥ 2 draws.
            initial_prices:    (n_assets,) array. Defaults to ones.
            is_tcvae:          If True, passes stage= kwarg to generator.sample() so the
                               same latent z is used across stages (preserving temporal
                               coherence from the trained decoder).
        """
        self._count = 0
        self.generator = generator
        self.historical_returns = kwargs.get("historical_returns", None)

        if initial_prices is None:
            initial_prices = np.ones(generator.n_assets)

        self.root = ScenarioNode(
            scenario_id=self._nid(), stage=0, branch_prob=1.0, value=initial_prices
        )
        current = [self.root]

        has_seq = getattr(generator, "has_sequential_sampling", False)

        for stage, k in enumerate(branching_factors, start=1):
            nxt = []
            for parent in current:
                if is_tcvae:
                    returns = generator.sample(k, stage=stage)
                    draws = parent.value * (1 + returns)
                    probs = np.full(k, 1.0 / k)

                elif has_seq:
                    # Sequential GARCH-Vine: update GARCH state from parent return
                    prev_returns = None
                    if stage >= 2 and parent.parent is not None:
                        prev_returns = parent.value / parent.parent.value - 1.0

                    if hasattr(generator, "sample_with_probs"):
                        node_returns, probs = generator.sample_with_probs(
                            k, prev_returns=prev_returns
                        )
                    else:
                        node_returns = generator.sample(k, prev_returns=prev_returns)
                        probs = np.full(k, 1.0 / k)
                    draws = parent.value * (1 + node_returns)

                else:
                    if stage >= 2:
                        low, high = fluctuation_range
                        fluctuations = np.random.uniform(
                            low, high, size=(k, generator.n_assets)
                        )
                        draws = parent.value * fluctuations
                    else:
                        returns = generator.sample(k)
                        draws = parent.value * (1 + returns)
                    probs = np.full(k, 1.0 / k)

                for i in range(k):
                    child = ScenarioNode(
                        scenario_id=self._nid(),
                        stage=stage,
                        branch_prob=float(probs[i]),
                        parent=parent,
                        value=draws[i],
                    )
                    parent.add_child(child)
                    nxt.append(child)
            current = nxt

        return self

    # ── Tree navigation ────────────────────────────────────────────────────────

    def _nid(self):
        i = self._count
        self._count += 1
        return i

    def bfs(self):
        out, q = [], [self.root]
        while q:
            n = q.pop(0)
            out.append(n)
            q.extend(n.children)
        return out

    def level(self, s):
        return [n for n in self.bfs() if n.t == s]

    def leaves(self):
        return [n for n in self.bfs() if n.is_leaf()]

    def path_to(self, node):
        path, cur = [], node
        while cur:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))

    def verify_probabilities(self):
        """Return {stage: sum_of_path_probs} — each value must equal 1.0."""
        sums = defaultdict(float)
        for n in self.bfs():
            sums[n.t] += n.path_prob
        return dict(sorted(sums.items()))

    def _get_path_returns(self, node):
        path = self.path_to(node)
        returns = []
        for i in range(1, len(path)):
            ret = (path[i].value / path[i - 1].value) - 1.0
            returns.append(ret)

        if self.historical_returns is not None and len(self.historical_returns) > 0:
            returns = list(self.historical_returns) + returns

        if not returns:
            return np.empty((0, len(node.value)))
        return np.array(returns)

    # ── Visualisation ──────────────────────────────────────────────────────────

    def plot_tree(self, asset_idx=0, title="", figsize=(16, 9)):
        all_nodes = self.bfs()
        G = nx.DiGraph()
        for n in all_nodes:
            G.add_node(id(n))
            if n.parent:
                G.add_edge(id(n.parent), id(n))

        pos = {}
        by_stage = defaultdict(list)
        for n in all_nodes:
            by_stage[n.t].append(n)
        for s, nodes in by_stage.items():
            cnt = len(nodes)
            for i, n in enumerate(nodes):
                pos[id(n)] = (s, -(i - (cnt - 1) / 2) * 1.5)

        vals = [
            n.value[asset_idx] if (n.value is not None and n.t > 0) else 0.0
            for n in all_nodes
        ]
        vmin, vmax = min(vals), max(vals)
        cmap = cm.RdYlGn
        norm_v = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(norm_v(v)) for v in vals]

        labels = {
            id(n): ("Root\nt=0" if n.t == 0 else f"t={n.t}\n{n.value[asset_idx]:+.4f}")
            for n in all_nodes
        }
        edge_labels = {
            (id(n.parent), id(n)): f"{n.p_n:.2f}" for n in all_nodes if n.parent
        }

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f5f5f5")
        ax.set_facecolor("#f5f5f5")
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=900, ax=ax, alpha=0.92)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)
        nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, arrowstyle="->", arrowsize=12,
            edge_color="#999", width=1.0,
        )
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=5.5, ax=ax,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.75),
        )
        sm = cm.ScalarMappable(cmap=cmap, norm=norm_v)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
        cb.set_label(f"Price – asset {asset_idx}", fontsize=8)

        max_s = max(n.t for n in all_nodes)
        ax.set_xticks(range(max_s + 1))
        ax.set_xticklabels([f"Stage {s}" for s in range(max_s + 1)], fontsize=9)
        ax.set_title(f"{title}  |  {len(self.leaves())} leaf scenarios", fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_fan(self, asset_idx=0, title="", figsize=(13, 6)):
        all_nodes = self.bfs()
        max_s = max(n.t for n in all_nodes)

        def compound_returns(rets):
            return np.cumprod(1.0 + np.array(rets)) - 1.0

        exp_cum = compound_returns(
            [
                sum(
                    n.path_prob
                    * (n.value[asset_idx] if n.value is not None else 0.0)
                    for n in self.level(s)
                )
                for s in range(max_s + 1)
            ]
        )

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f5f5f5")
        ax.set_facecolor("#f5f5f5")

        for leaf in self.leaves():
            path = self.path_to(leaf)
            xs = [n.t for n in path]
            rets = [n.value[asset_idx] if n.value is not None else 0.0 for n in path]
            ax.plot(xs, compound_returns(rets), color="#3a7abf", alpha=0.20, linewidth=0.7)

        ax.plot(
            range(max_s + 1), exp_cum, color="#e63946", linewidth=2.2,
            label="E[cumulative return]", zorder=5,
        )
        ax.axhline(0, color="#444", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlabel("Stage", fontsize=10)
        ax.set_ylabel("Cumulative compound return", fontsize=10)
        ax.set_title(f"Scenario fan chart  |  {title}  |  asset {asset_idx}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_stats(self, asset_idx=0, title="", figsize=(11, 5)):
        all_nodes = self.bfs()
        max_s = max(n.t for n in all_nodes)
        stages, means, stds_out = [], [], []

        for s in range(1, max_s + 1):
            lvl = self.level(s)
            vals = np.array([n.value[asset_idx] for n in lvl])
            probs = np.array([n.path_prob for n in lvl])
            probs /= probs.sum()
            mu = np.dot(probs, vals)
            sig = np.sqrt(np.dot(probs, (vals - mu) ** 2))
            stages.append(s)
            means.append(mu)
            stds_out.append(sig)

        stages = np.array(stages)
        means = np.array(means)
        stds_a = np.array(stds_out)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f5f5f5")
        ax.set_facecolor("#f5f5f5")
        ax.fill_between(
            stages, means - stds_a, means + stds_a,
            alpha=0.22, color="#3a7abf", label="+/- 1 std",
        )
        ax.plot(
            stages, means, "o-", color="#3a7abf",
            linewidth=2, markersize=7, label="Weighted mean",
        )
        ax.axhline(0, color="#555", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_xlabel("Stage", fontsize=10)
        ax.set_ylabel(f"Return (asset {asset_idx})", fontsize=10)
        ax.set_xticks(stages)
        ax.set_xticklabels([f"Stage {s}" for s in stages])
        ax.set_title(f"Return statistics per stage  |  {title}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
