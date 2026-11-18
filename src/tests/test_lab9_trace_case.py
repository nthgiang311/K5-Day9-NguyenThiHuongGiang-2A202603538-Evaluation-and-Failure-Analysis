import pytest

pytestmark = pytest.mark.student


def test_swarm_ticket_override_trace_is_detected():
    from lab9.trace_case import investigate

    case = investigate("swarm", "C-002", 55)
    trial = case["trial"]
    assert trial["lab8_pass"] is False
    assert any(event.get("kind") == "model_override" for event in trial["events"])
    assert trial["snapshot"]["refund_vnd"] == 20000
    assert case["label"]["refund_vnd"] == 0
