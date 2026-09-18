"""Core V6 error types."""

from __future__ import annotations


class NLSSError(Exception):
    """Base NLSS error."""


class GroundingError(NLSSError):
    """Compile (P_tau) failed to produce a typed scientific specification."""


class VerificationRejected(NLSSError):
    """Verification (V_tau) rejected a raw object."""


class CanonicalizationError(NLSSError):
    """Canonicalization (K_tau) failed."""


class EvaluationNotApplicable(NLSSError):
    """The benchmark has no scalar empirical evaluation (e.g. HypoSpace)."""


class RecoveryNotFitted(NLSSError):
    """A posterior was requested before the recovery backend was fitted."""


class ContractViolation(NLSSError):
    """A shared NLSS contract invariant was violated."""
