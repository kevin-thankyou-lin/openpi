"""Opt-in source50 camera adapter; preserve the campaign selector's complete plan path.

Install before main imports StriderPhaseSelector, or assign main.StriderPhaseSelector
explicitly. This module never starts policy, physics, or rollout processes.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

CODE_COMMIT = '6cbb1e20ea488a8b8f27af89a1129e1c8b80baa2'
SUITE_PREFIX = {'libero_10': 'long', 'libero_spatial': 'spatial',
                'libero_object': 'object', 'libero_goal': 'goal'}
SOURCE_CAMERA = {'intermediate_size': [224, 224],
                 'interpolation': 'OpenCV linear then Pillow bilinear',
                 'original_key': 'obs/agentview_rgb', 'output_size': [84, 84],
                 'rotate_180': True}


def source50_observation(obs):
    """Only observed RGB and joint/gripper state; no policy or privileged inputs."""
    import cv2
    image = np.asarray(obs['agentview_image'])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('source50 requires uint8 HWC agentview RGB')
    image = cv2.resize(np.ascontiguousarray(image[::-1, ::-1]), (224, 224),
                       interpolation=cv2.INTER_LINEAR)
    image = np.asarray(Image.fromarray(image).resize((84, 84), Image.Resampling.BILINEAR)).copy()
    joint = np.asarray(obs['robot0_joint_pos'], dtype=np.float32)
    grip = np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32)
    if joint.shape != (7,) or grip.shape != (2,):
        raise ValueError('source50 requires observed joint7 and gripper2')
    state = np.concatenate((joint, grip))
    if not np.isfinite(state).all():
        raise ValueError('source50 state must be finite')
    return {'rgb': image, 'proprio': state}


def selector_class(base):
    """Extend only input adaptation and selected-identity checks."""
    class Source50PhaseSelector(base):
        def __init__(self, **kwargs):
            import subprocess
            binding = kwargs.get('phase_input_binding')
            if binding is None or kwargs.get('legacy_input_mode', False):
                raise ValueError('source50 requires checked campaign phase-input binding')
            repo = Path(kwargs['phase_repo']).resolve()
            head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            if head != CODE_COMMIT:
                raise ValueError('source50 code commit mismatch')
            # Base handles pinned import root; do not import STRIDER prematurely.
            super().__init__(**kwargs)
            from strider.subtask_recipes import load_recipe, validate_recipe_manifest, validate_recipe_model
            manifest_raw = Path(binding['manifest_path']).read_bytes()
            if hashlib.sha256(manifest_raw).hexdigest() != binding['expected_manifest_sha256']:
                raise ValueError('source50 manifest changed during initialization')
            manifest = json.loads(manifest_raw)
            identity = binding['identity']
            task = SUITE_PREFIX[identity['suite']] + str(identity['task_id'])
            registry = json.loads((repo/'src/strider/recipes/libero_source50_v1_checkpoints.json').read_text())['tasks'][task]
            if binding['expected_checkpoint_sha256'] != registry['checkpoint_sha256']:
                raise ValueError('checkpoint is not the selected source50 task')
            if binding['expected_manifest_sha256'] != registry['dataset_manifest_sha256']:
                raise ValueError('manifest is not the selected source50 task')
            recipe = load_recipe()
            validate_recipe_manifest(recipe, manifest)
            validate_recipe_model(recipe, vars(self.predictor._model.config))
            if manifest['provenance']['source_camera'] != SOURCE_CAMERA:
                raise ValueError('source50 source camera provenance mismatch')
            if tuple(self.predictor.class_names) != tuple(registry['class_names']):
                raise ValueError('source50 phase ontology mismatch')
            self.source50_adapter_proof = {
                'code_commit': CODE_COMMIT, 'task': task,
                'runtime_rgb': 'raw observed agentview rotate180 cv2-linear224 Pillow-bilinear84',
                'observed_state': 'joint7+gripper2', 'history': 4, 'horizon': 10,
                'future_phases_used': True,
            }

        def observe(self, image, obs):
            self._observations.append(source50_observation(obs))
            self._observations = self._observations[-int(self.predictor.history_length):]

    return Source50PhaseSelector


def install(strider_client_module):
    cls = selector_class(strider_client_module.StriderPhaseSelector)
    strider_client_module.StriderPhaseSelector = cls
    return cls
