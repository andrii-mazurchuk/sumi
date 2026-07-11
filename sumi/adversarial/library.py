"""
Adversarial prompt library — loads and samples from the four JSONL attack datasets.

Attack types:
  direct_demand      — explicit requests to drop the persona (standalone prompts)
  gradual_pressure   — escalating sequences grouped by sequence_id
  roleplay_injection — inject an alternative character/role (standalone prompts)
  logical_challenge  — rhetorical arguments against the persona (standalone prompts)
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

STANDALONE_ATTACK_TYPES = ("direct_demand", "roleplay_injection", "logical_challenge")
ALL_ATTACK_TYPES = (*STANDALONE_ATTACK_TYPES, "gradual_pressure")


def load_prompts(attack_type: str) -> list[dict]:
    """Load all prompt records for a given attack type from its JSONL file."""
    path = DATA_DIR / f"{attack_type}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_sequences() -> list[list[dict]]:
    """
    Load gradual_pressure prompts grouped by sequence_id, sorted by sequence_order.
    Returns a list of sequences; each sequence is a list of turn dicts in escalation order.
    """
    prompts = load_prompts("gradual_pressure")
    groups: dict[str, list[dict]] = {}
    for p in prompts:
        groups.setdefault(p["sequence_id"], []).append(p)
    return [
        sorted(turns, key=lambda t: t["sequence_order"])
        for turns in groups.values()
    ]


def sample_prompts(attack_type: str, n: int, seed: int = 42) -> list[dict]:
    """Sample up to n prompts for a standalone attack type (reproducible via seed)."""
    prompts = load_prompts(attack_type)
    rng = random.Random(seed)
    return rng.sample(prompts, min(n, len(prompts)))


def sample_sequences(n: int, seed: int = 42) -> list[list[dict]]:
    """Sample up to n full sequences from gradual_pressure (reproducible via seed)."""
    seqs = load_sequences()
    rng = random.Random(seed)
    return rng.sample(seqs, min(n, len(seqs)))
