from pathlib import Path

import numpy as np
import pytest

from examples.libero.speed_baselines.source50_selector import selector_class, source50_observation
from examples.libero.speed_baselines.strider_client import StriderPhaseSelector


def observation():
    return {
        'agentview_image': np.arange(128*128*3, dtype=np.uint8).reshape(128, 128, 3),
        'robot0_joint_pos': np.arange(7, dtype=np.float32),
        'robot0_gripper_qpos': np.array([0.02, -0.02], dtype=np.float32),
    }


def test_camera_matches_source_path_without_using_policy_image():
    import cv2
    from PIL import Image

    obs = observation()
    expected = np.asarray(Image.fromarray(cv2.resize(obs['agentview_image'][::-1, ::-1].copy(),
        (224, 224), interpolation=cv2.INTER_LINEAR)).resize((84, 84), Image.Resampling.BILINEAR))
    selector = object.__new__(selector_class(StriderPhaseSelector))
    selector._observations = []
    selector.predictor = type('Predictor', (), {'history_length': 4})()
    selector.observe(np.zeros((224, 224, 3), dtype=np.uint8), obs)
    assert np.array_equal(selector._observations[0]['rgb'], expected)
    assert np.array_equal(selector._observations[0]['proprio'],
        np.concatenate((obs['robot0_joint_pos'], obs['robot0_gripper_qpos'])))


@pytest.mark.parametrize('key,value', [
    ('agentview_image', np.zeros((84,84,3),dtype=float)),
    ('robot0_joint_pos', np.zeros(8)),
    ('robot0_gripper_qpos', np.array([np.nan,0.])),
])
def test_rejects_invalid_runtime_observations(key, value):
    obs = observation()
    obs[key] = value
    with pytest.raises(ValueError): source50_observation(obs)


def test_plan_is_exact_inherited_horizon_path():
    cls = selector_class(StriderPhaseSelector)
    assert cls.plan is StriderPhaseSelector.plan
    assert cls.reset_episode is StriderPhaseSelector.reset_episode


def test_requires_checked_source50_binding():
    cls = selector_class(StriderPhaseSelector)
    with pytest.raises(ValueError, match='checked campaign'):
        cls(phase_input_binding=None)
