"""
Scenario loading and validation.

Usage:
    scenario = load_scenario("examples/scenarios/minimalist_analyst.yaml")
    print(scenario.name, len(scenario.traits), len(scenario.test_cases))
"""

from pathlib import Path
from typing import Union

import yaml

from sumi.models import (
    PassThreshold,
    TestCase,
    Trait,
    ValidationScenario,
)


def load_scenario(path: Union[str, Path]) -> ValidationScenario:
    """
    Load and validate a ValidationScenario from a YAML file.

    Raises:
        FileNotFoundError: if path doesn't exist
        yaml.YAMLError: if YAML is malformed
        pydantic.ValidationError: if schema is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return _parse_scenario(data)


def _parse_scenario(data: dict) -> ValidationScenario:
    """Parse raw dict (from YAML) into a ValidationScenario."""

    traits = [Trait(**t) for t in data.get("traits", [])]

    test_cases = []
    for tc in data.get("test_cases", []):
        test_cases.append(TestCase(**tc))

    threshold_data = data.get("pass_threshold", {})
    if isinstance(threshold_data, dict):
        threshold = PassThreshold(**threshold_data)
    else:
        threshold = PassThreshold()

    return ValidationScenario(
        name=data["name"],
        goal=data["goal"].strip() if isinstance(data["goal"], str) else data["goal"],
        traits=traits,
        test_cases=test_cases,
        pass_threshold=threshold,
        temporal_turns=data.get("temporal_turns", 20),
        metadata=data.get("metadata"),
    )
