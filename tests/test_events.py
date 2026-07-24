import numpy as np

from dyn_evt_pdm.evaluation.events import flags_to_events
from dyn_evt_pdm.evaluation.metrics import event_metrics
from dyn_evt_pdm.types import EventInterval


def test_flags_to_events() -> None:
    flags = np.array([False, True, True, False, True])
    events = flags_to_events(flags)
    assert [(event.start, event.end) for event in events] == [(1, 2), (4, 4)]


def test_event_metrics_penalise_duplicate_alarms() -> None:
    observed = [EventInterval(10, 20)]
    predicted = [EventInterval(10, 12), EventInterval(16, 18)]
    metrics = event_metrics(predicted, observed)
    assert metrics.recall == 1.0
    assert metrics.duplicate_alarm_rate == 1.0
