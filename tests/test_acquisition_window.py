import numpy as np
import pytest
from examples.libero.speed_baselines.acquisition_window import ClosePreviewSchedule, validate_window, window_speeds

CONFIG = {'kind': 'close_preview_v1', 'lead_source_steps': 4}


def test_protects_preclose_and_closed_suffix_without_changing_actions():
    raw = np.zeros((10, 7)); raw[:, -1] = -1; raw[7:, -1] = 1
    before = raw.copy()
    speeds, near = window_speeds(raw, ['acquire'] * 10, [1] * 10, [], CONFIG)
    assert speeds == (2, 2, 2, 1, 1, 1, 1, 1, 1, 1)
    assert near == (False, False, False, True, True, True, True, True, True, True)
    np.testing.assert_array_equal(raw, before)


def test_preserves_other_phases_and_fallback():
    raw = -np.ones((10, 7))
    labels = ['approach', 'acquire', 'acquire', None, 'transport', 'place', 'recover', 'acquire', 'acquire', 'acquire']
    speeds, _ = window_speeds(raw, labels, [2, 1, 1, 1, 2, 2, 2, 1, 1, 1], [2, 3], CONFIG)
    assert speeds == (2, 2, 1, 1, 2, 2, 2, 2, 2, 2)


def test_no_lookahead_into_a_future_chunk():
    raw = -np.ones((10, 7))
    assert window_speeds(raw, ['acquire'] * 10, [1] * 10, [], CONFIG)[0] == (2,) * 10
    raw[:, -1] = 1
    assert window_speeds(raw, ['acquire'] * 10, [1] * 10, [], CONFIG)[0] == (1,) * 10


@pytest.mark.parametrize('lead', [-1, 10, True, 4.0])
def test_invalid_lead(lead):
    with pytest.raises(ValueError): validate_window({**CONFIG, 'lead_source_steps': lead})


def test_schedule_integrates_with_real_retimer_and_keeps_every_source_row():
    from strider import SubtaskSpeedSchedule, BoundaryAwareStrideRetimer, PlanPipeline, StriderRuntime
    from strider.subtask_prediction import SubtaskPrediction
    class Predictor:
        horizon = 10
        class_names = ('acquire',)
        def predict(self, observations):
            return SubtaskPrediction(probabilities=((1.0,),) * 10, valid=(True,) * 10, class_names=('acquire',))
    class Composer:
        def compose(self, actions): return np.sum(actions, axis=0)
    raw = -np.ones((10, 7)); raw[7:, -1] = 1
    gate = ClosePreviewSchedule(SubtaskSpeedSchedule({'acquire': 1}), CONFIG);gate.actions = raw
    runtime = StriderRuntime(predictor=Predictor(), schedule=gate,
        pipeline=PlanPipeline(retimer=BoundaryAwareStrideRetimer(Composer(), allowed_speeds=(1, 2, 3))))
    result = runtime.plan([{}], raw)
    groups = [tuple(p['source_indices']) for p in result.transformed.scheduled.provenance]
    assert groups == [(0, 1), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
    assert [i for g in groups for i in g] == list(range(10))
    assert result.metadata['acquisition_window']['config'] == CONFIG
