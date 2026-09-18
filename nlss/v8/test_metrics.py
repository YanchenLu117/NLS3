"""P0-f tests: common recovery metrics (Detail §6)."""

import pytest

from nlss.v8.metrics import (
    MetricsProtocolError,
    apply_failure_score,
    average_precision,
    common_universe,
    recovery_aurc,
    truncated_recovery_aurc,
    utility_auc,
    checkpoint_recovery,
)


def test_common_universe_excludes_every_arms_queries() -> None:
    legal = {"c1", "c2", "c3", "c4", "c5", "c6"}
    initial = {"c1"}
    queried = {"arm_a": {"c2"}, "arm_b": {"c3"}}
    u = common_universe(legal, initial, queried)
    assert u == {"c4", "c5", "c6"}


def test_average_precision_tie_breaks_by_id() -> None:
    # universe {c2..c6}, positives {c4, c6}; scores tie for c5/c6
    scores = {"c2": 0.9, "c3": 0.8, "c4": 0.5, "c5": 0.7, "c6": 0.7}
    ap = average_precision(scores, {"c4", "c6"}, {"c2", "c3", "c4", "c5", "c6"})
    # ranking: c2(.9), c3(.9)? c3=0.9 too -> order by score desc then id: c2(.9), c3(.9), c5(.7), c6(.7), c4(.5)
    # hits at ranks 3 (c4? no — c4 not in ranking... c4 IS in universe with score .5
    # ranking: c2(.9), c3(.9), c5(.7), c6(.7), c4(.5) -> positives c4@5, c6@4
    # AP = (1/4 at rank4 for c6 + 2/5 at rank5 for c4) / 2 = (0.25+0.4)/2 = 0.325
    assert ap == pytest.approx((1 / 4 + 2 / 5) / 2)


def test_average_precision_predict_everything_scores_mid() -> None:
    # equal scores -> id order; positives spread out -> AP well below 1
    ids = [f"c{i}" for i in range(1, 7)]
    scores = {c: 0.5 for c in ids}
    ap = average_precision(scores, {"c5", "c6"}, set(ids))
    assert ap < 0.5


def test_checkpoint_recovery_depleted_convention() -> None:
    # common universe empty -> R=1, depleted flag
    r, depleted = checkpoint_recovery(universe=set(), solution_ids={"c1"}, scores={"c1": 1.0})
    assert r == 1.0 and depleted
    # solution empty in universe -> R=1, depleted
    r, depleted = checkpoint_recovery(universe={"c2"}, solution_ids={"c9"}, scores={"c2": 0.5})
    assert r == 1.0 and depleted
    # normal AP path
    r, depleted = checkpoint_recovery(
        universe={"c2", "c3"}, solution_ids={"c3"}, scores={"c2": 0.2, "c3": 0.9}
    )
    assert r == 1.0 and not depleted  # the single positive ranked first


def test_recovery_aurc_trapezoid_hand_computed() -> None:
    # budgets 40..240 -> u = 0, .2, .4, .6, .8, 1 ; R = [0, .1, .3, .5, .7, .9]
    budgets = [40, 80, 120, 160, 200, 240]
    recs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    # trapezoid over normalized steps of 0.2
    expected = sum((recs[j - 1] + recs[j]) / 2 * 0.2 for j in range(1, 6))
    assert recovery_aurc(budgets, recs) == pytest.approx(expected)
    assert recovery_aurc(budgets, recs) == pytest.approx(0.41)


def test_recovery_aurc_validation() -> None:
    with pytest.raises(MetricsProtocolError):
        recovery_aurc([4, 8], [0.1])
    with pytest.raises(MetricsProtocolError):
        recovery_aurc([8, 4], [0.1, 0.2])
    with pytest.raises(MetricsProtocolError):
        recovery_aurc([4, 4], [0.0, 0.0])


def test_truncated_aurc_excludes_depleted_checkpoint() -> None:
    budgets = [4, 6, 8, 10]
    recs = [0.2, 0.4, 1.0, 1.0]  # checkpoint 3 depleted (R=1)
    full = recovery_aurc(budgets, recs)
    truncated = truncated_recovery_aurc(budgets, recs, upto_index=2)
    assert truncated < full


def test_apply_failure_score_zeroes_from_failure_onward() -> None:
    recs = [0.2, 0.4, 0.6, 0.8]
    utils = [5.0, 6.0, 7.0, 8.0]
    r, u = apply_failure_score(recs, utils, failure_checkpoint_index=2, worst_legal_utility=0.0)
    assert r == [0.2, 0.4, 0.0, 0.0]
    assert u == [5.0, 6.0, 0.0, 0.0]
    with pytest.raises(MetricsProtocolError):
        apply_failure_score(recs, utils, failure_checkpoint_index=9, worst_legal_utility=0.0)


def test_utility_auc_same_scalarization() -> None:
    budgets = [40, 80, 120]
    assert utility_auc(budgets, [0.0, 0.5, 1.0]) == pytest.approx(recovery_aurc(budgets, [0.0, 0.5, 1.0]))


def test_checkpoint_recovery_submitted_set() -> None:
    r, depleted = checkpoint_recovery(
        universe={"c2", "c3", "c4"}, solution_ids={"c4", "c5"}, submitted_set={"c2", "c4", "c9"}
    )
    # positives in universe: {c4}; submitted ∩ universe positives: {c4} -> R=1
    assert r == 1.0 and not depleted
