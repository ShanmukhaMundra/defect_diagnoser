from __future__ import annotations

import difflib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

DATA_PATH = Path(__file__).with_name("kg_data.json")

# Confidence bands. Kept in one place so they can be calibrated against
# historical tickets (see README, "Calibrating").
HIGH_MIN_POSTERIOR = 0.75
HIGH_MIN_RATIO = 3.0          # top cause must be >= 3x the runner-up
MEDIUM_MIN_POSTERIOR = 0.50
UNMODELED_WARN = 0.20


@dataclass(frozen=True)
class Evidence:
    present: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    checks: tuple[tuple[str, bool], ...] = ()   # (check_id, positive?)

    def is_empty(self) -> bool:
        return not (self.present or self.absent or self.checks)


class DefectKG:
    def __init__(self, data: dict):
        self.data = data
        meta = data["_meta"]
        self.leak = float(meta["leak_probability"])
        um = meta["unmodeled_cause"]
        self.unmodeled_id = um["id"]
        self.unmodeled_prior = float(um["prior"])
        self.unmodeled_symptom_p = float(um["symptom_probability"])

        self.symptoms: dict = data["symptoms"]
        self.causes: dict = data["root_causes"]
        self.checks: dict = data["checks"]
        self.graph = self._build_graph()
        self._alias_index = self._build_alias_index()

    @classmethod
    def load(cls, path: Path | str = DATA_PATH) -> "DefectKG":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------ graph
    def _build_graph(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for sid, s in self.symptoms.items():
            g.add_node(sid, type="Symptom", name=s["name"])
        for cid, c in self.causes.items():
            g.add_node(cid, type="RootCause", name=c["name"], prior=c["prior"], remedy=c["remedy"])
            for sid, p in c["symptoms"].items():
                if sid not in self.symptoms:
                    raise ValueError(f"{cid} references unknown symptom {sid}")
                g.add_edge(cid, sid, key="CAUSES", rel="CAUSES", p=float(p))
            for param in c.get("controlled_by", []):
                if param not in g:
                    g.add_node(param, type="Parameter", name=param.replace("_", " "))
                g.add_edge(cid, param, key="CONTROLLED_BY", rel="CONTROLLED_BY")
        for kid, k in self.checks.items():
            g.add_node(kid, type="Check", name=k["name"], minutes=k["minutes"])
            for cid, sens in k["targets"].items():
                if cid not in self.causes:
                    raise ValueError(f"check {kid} targets unknown cause {cid}")
                g.add_edge(cid, kid, key="VERIFIED_BY", rel="VERIFIED_BY", sensitivity=float(sens))

        total = sum(c["prior"] for c in self.causes.values()) + self.unmodeled_prior
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"priors must sum to 1, got {total:.4f}")
        return g

    def _build_alias_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        for sid, s in self.symptoms.items():
            for term in [sid, sid.replace("_", " "), s["name"], *s.get("aliases", [])]:
                idx[_norm(term)] = sid
        return idx

    def resolve_symptom(self, term: str) -> str | None:
        """Map free text ('silver streaks') to a symptom id ('splay_marks')."""
        key = _norm(term)
        if key in self._alias_index:
            return self._alias_index[key]
        match = difflib.get_close_matches(key, self._alias_index.keys(), n=1, cutoff=0.82)
        return self._alias_index[match[0]] if match else None

    def resolve_check(self, term: str) -> str | None:
        key = _norm(term)
        for kid, k in self.checks.items():
            if key in (_norm(kid), _norm(k["name"])):
                return kid
        match = difflib.get_close_matches(key, [_norm(k) for k in self.checks], n=1, cutoff=0.82)
        if match:
            return next(k for k in self.checks if _norm(k) == match[0])
        return None

    def hypotheses(self) -> list[str]:
        return [*self.causes, self.unmodeled_id]

    def prior(self, h: str) -> float:
        return self.unmodeled_prior if h == self.unmodeled_id else self.causes[h]["prior"]

    # ------------------------------------------------------------ likelihoods
    def p_symptom(self, h: str, sid: str, edge_override: dict | None = None) -> float:
        if h == self.unmodeled_id:
            return self.unmodeled_symptom_p
        p = self.causes[h]["symptoms"].get(sid, 0.0)
        if edge_override is not None:
            p = edge_override.get((h, sid), p)
        return 1.0 - (1.0 - self.leak) * (1.0 - p)

    def p_check_positive(self, h: str, kid: str) -> float:
        k = self.checks[kid]
        fpr = k["false_positive_rate"]
        sens = k["targets"].get(h)
        if sens is None:
            return fpr
        return 1.0 - (1.0 - fpr) * (1.0 - sens)

    def log_likelihood(self, h: str, ev: Evidence, edge_override: dict | None = None) -> float:
        ll = 0.0
        for s in ev.present:
            ll += math.log(self.p_symptom(h, s, edge_override))
        for s in ev.absent:
            ll += math.log(1.0 - self.p_symptom(h, s, edge_override))
        for kid, pos in ev.checks:
            p = self.p_check_positive(h, kid)
            ll += math.log(p if pos else 1.0 - p)
        return ll

    def posterior(self, ev: Evidence, edge_override: dict | None = None) -> dict[str, float]:
        logs = {h: math.log(self.prior(h)) + self.log_likelihood(h, ev, edge_override)
                for h in self.hypotheses()}
        m = max(logs.values())
        w = {h: math.exp(v - m) for h, v in logs.items()}
        z = sum(w.values())
        return {h: v / z for h, v in w.items()}

    # ------------------------------------------------------- explain / decide
    def evidence_contributions(self, h: str, ev: Evidence) -> list[dict]:
        """Likelihood ratio of each piece of evidence for h vs. the prior-weighted alternatives."""
        others = [o for o in self.hypotheses() if o != h]
        z = sum(self.prior(o) for o in others)
        out = []

        def mix(fn):
            return sum(self.prior(o) * fn(o) for o in others) / z

        for s in ev.present:
            lr = self.p_symptom(h, s) / mix(lambda o: self.p_symptom(o, s))
            out.append({"evidence": f"{self.symptoms[s]['name']} observed", "likelihood_ratio": round(lr, 2)})
        for s in ev.absent:
            lr = (1 - self.p_symptom(h, s)) / mix(lambda o: 1 - self.p_symptom(o, s))
            out.append({"evidence": f"{self.symptoms[s]['name']} NOT observed", "likelihood_ratio": round(lr, 2)})
        for kid, pos in ev.checks:
            def pk(x, kid=kid, pos=pos):
                p = self.p_check_positive(x, kid)
                return p if pos else 1 - p
            lr = pk(h) / mix(pk)
            label = self.checks[kid]["positive_means"] if pos else "negative"
            out.append({"evidence": f"{self.checks[kid]['name']}: {label}", "likelihood_ratio": round(lr, 2)})
        for item in out:
            r = item["likelihood_ratio"]
            item["effect"] = "supports" if r > 1.25 else "argues against" if r < 0.8 else "neutral"
        return sorted(out, key=lambda d: -abs(math.log(max(d["likelihood_ratio"], 1e-9))))

    def stability(self, ev: Evidence, top: str, samples: int = 300, spread: float = 0.5,
                  seed: int = 7) -> float:
        """Share of runs where `top` stays the leading cause when every edge
        probability is jittered in logit space (sd=`spread`, roughly +/-25%
        relative on mid-range values). Tests whether the answer depends on the
        exact numbers someone typed into the graph."""
        rng = random.Random(seed)
        edges = [(c, s, p) for c, cd in self.causes.items() for s, p in cd["symptoms"].items()]
        hits = 0
        for _ in range(samples):
            override = {(c, s): _sigmoid(_logit(p) + rng.gauss(0, spread)) for c, s, p in edges}
            post = self.posterior(ev, override)
            if max(post, key=post.get) == top:
                hits += 1
        return hits / samples

    def next_best_actions(self, ev: Evidence, post: dict[str, float], k: int = 3) -> list[dict]:
        """Rank unobserved symptoms and un-run checks by expected information gain (bits)."""
        h0 = _entropy(post.values())
        seen_s = set(ev.present) | set(ev.absent)
        seen_k = {kid for kid, _ in ev.checks}
        options = []

        def gain(p_pos_given_h):
            p_pos = sum(post[h] * p_pos_given_h(h) for h in post)
            exp_h = 0.0
            for outcome_p, like in ((p_pos, p_pos_given_h), (1 - p_pos, lambda h: 1 - p_pos_given_h(h))):
                if outcome_p <= 1e-12:
                    continue
                cond = [post[h] * like(h) / outcome_p for h in post]
                exp_h += outcome_p * _entropy(cond)
            return h0 - exp_h

        for sid in self.symptoms:
            if sid not in seen_s:
                g = gain(lambda h, sid=sid: self.p_symptom(h, sid))
                options.append({"type": "ask_about_symptom", "id": sid, "name": self.symptoms[sid]["name"],
                                "expected_info_gain_bits": round(g, 3), "minutes": 2})
        for kid, kd in self.checks.items():
            if kid not in seen_k:
                g = gain(lambda h, kid=kid: self.p_check_positive(h, kid))
                options.append({"type": "run_check", "id": kid, "name": kd["name"], "how": kd["how"],
                                "expected_info_gain_bits": round(g, 3), "minutes": kd["minutes"]})
        options.sort(key=lambda o: -o["expected_info_gain_bits"])
        return [o for o in options if o["expected_info_gain_bits"] > 0.01][:k]

    def confidence_label(self, ranked: list[tuple[str, float]], ev: Evidence) -> tuple[str, str]:
        (top, p1), (_, p2) = ranked[0], ranked[1]
        ratio = p1 / max(p2, 1e-9)
        if ev.is_empty():
            return "none", "No evidence supplied; the ranking is just the historical base rate."
        if top == self.unmodeled_id:
            return "low", "The evidence fits none of the modeled causes well; escalate to process engineering."
        if p1 >= HIGH_MIN_POSTERIOR and ratio >= HIGH_MIN_RATIO:
            return "high", f"Top cause has {p1:.0%} posterior and is {ratio:.1f}x the runner-up."
        if p1 >= MEDIUM_MIN_POSTERIOR:
            return "medium", f"Top cause leads ({p1:.0%}) but the runner-up still has {p2:.0%}."
        return "low", f"No cause exceeds 50% (top {p1:.0%}); more evidence needed."

    def diagnose(self, ev: Evidence, top_n: int = 3) -> dict:
        post = self.posterior(ev)
        ranked = sorted(post.items(), key=lambda kv: -kv[1])
        label, reason = self.confidence_label(ranked, ev)
        top_id, top_p = ranked[0]

        candidates = []
        for h, p in ranked[:top_n]:
            if h == self.unmodeled_id:
                candidates.append({"id": h, "name": self.data["_meta"]["unmodeled_cause"]["name"],
                                   "probability": round(p, 3)})
                continue
            c = self.causes[h]
            candidates.append({
                "id": h, "name": c["name"], "probability": round(p, 3),
                "evidence": self.evidence_contributions(h, ev),
                "remedy": c["remedy"],
                "process_parameters": c.get("controlled_by", []),
                "verify_with": [self.checks[k]["name"] for _, k, d in self.graph.out_edges(h, data=True)
                                if d.get("rel") == "VERIFIED_BY"],
            })

        stab = self.stability(ev, top_id) if not ev.is_empty() else None
        if label == "high" and stab is not None and stab < 0.8:
            label, reason = "medium", reason + f" Downgraded: the lead holds in only {stab:.0%} of perturbed graphs."

        return {
            "diagnosis": candidates[0]["name"],
            "diagnosis_id": top_id,
            "probability": round(top_p, 3),
            "confidence": label,
            "confidence_reason": reason,
            "stability_under_perturbation": None if stab is None else round(stab, 2),
            "unmodeled_cause_probability": round(post[self.unmodeled_id], 3),
            "out_of_graph_warning": post[self.unmodeled_id] >= UNMODELED_WARN,
            "candidates": candidates,
            "next_best_actions": self.next_best_actions(ev, post),
            "assumptions": [
                "Single dominant root cause (single-fault model).",
                "Symptoms not mentioned are treated as unknown, not absent.",
                "Probabilities come from the knowledge graph's priors and edge weights, not from the language model.",
            ],
        }


# ---------------------------------------------------------------- helpers
def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").replace("/", " ").split())


def _entropy(ps) -> float:
    return -sum(p * math.log2(p) for p in ps if p > 0)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
