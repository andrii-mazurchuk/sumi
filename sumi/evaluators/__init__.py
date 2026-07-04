"""Evaluator implementations."""
from sumi.evaluators.base import Evaluator
from sumi.evaluators.stylometric import StylometricEvaluator
from sumi.evaluators.pattern import PatternEvaluator

__all__ = ["Evaluator", "StylometricEvaluator", "PatternEvaluator"]
