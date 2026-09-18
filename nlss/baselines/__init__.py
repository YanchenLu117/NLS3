"""V7 baselines (EXPERIMENT_PLAN §5).

Controlled comparators A0-A4 + specialist block (GP-BO / DKL-BO / ALDE / GP-LSE)
+ proposal policies.  `registry` is the single authority for selecting an arm;
heavy/external arms are honest shims that the run-driver must supply.
"""

from .gp_bo import GPBandit
from .gp_lse import GPLSEDecision, GPLSE
from .registry import (
    available_baselines,
    create_baseline,
    get_baseline,
)

__all__ = ["GPLSE", "GPLSEDecision", "GPBandit",
           "create_baseline", "get_baseline", "available_baselines"]

# Native AI-scientist / host arms (§5.2)
from . import native_arms
from .native_arms import Host, arm_injection, host_arms, native_baselines_index, native_hosts, spec
from .native_arms import AIScientistSpec, ai_scientist_keys, ai_scientist_systems
