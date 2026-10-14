"""Matched SuP admission and applied-action audit; no simulator or policy inference."""
import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_campaign_session import session_spec, asset
from test_campaign_audit import evidence
from examples.libero.speed_baselines import campaign_routes as cr
from examples.libero.speed_baselines import campaign_policy as cp
from examples.libero.speed_baselines import campaign_session as cs
from examples.libero.speed_baselines import sup_identity as si
from examples.libero.speed_baselines.campaign_audit import audit_episode

AUDIT_ROOT = Path('/home/linke/strider-spatial-all10-search20-20261014/audits/sup')

@pytest.fixture
def sup_session(session_spec, monkeypatch):
    spec=json.loads(session_spec.read_text())
    policy_path=Path(spec['policy_spec']['path']);policy=json.loads(policy_path.read_text())
    registry_path=Path(policy['route_registry']['path']);registry=json.loads(registry_path.read_text())
    route=registry['routes'][0];route.pop('uniform_stride');route.update(method='sup',sup_epsilon=0.0175,
        sup_checkpoint=asset(AUDIT_ROOT/'scheduler-eps0175.pt'))
    registry_path.write_text(json.dumps(registry))
    policy.update(route_registry=asset(registry_path),arm='sup_eps0p0175',ordered_episode_indices=list(range(40,50)))
    policy_path.write_text(json.dumps(policy));spec.update(policy_spec=asset(policy_path),selection_closed=True,
        baseline_cap=10,scene_mode='restore',candidate_id='sup_eps0p0175')
    spec['initial_states']={str(ep):{'sha256':cp.array_hash(np.array([ep],dtype=float)),'dtype':'float64','shape':[1]} for ep in range(40,50)}
    scene=session_spec.parent/'scene.json';scene.write_text('{}')
    spec['scenes']={str(ep):asset(scene) for ep in range(40,50)}
    spec['reference_initials']={str(ep):{} for ep in range(40,50)}
    source=Path(cs.__file__).parent
    spec['implementation_sources'].update(selector_client=asset(source/'selector_client.py'),
        sup_identity=asset(source/'sup_identity.py'),selector_service=asset(source.parents[2]/'scripts/serve_sup_selector.py'))
    session_spec.write_text(json.dumps(spec));monkeypatch.setattr(cs,'load_scene',lambda *a,**kw:SimpleNamespace())
    return session_spec

def test_matched_sup_session_admitted_and_policy_preserves_base(sup_session):
    session=cs.CampaignSession(sup_session,repo=Path(cs.__file__).parents[3])
    assert session.bank==tuple(range(40,50))
    metadata=cp.expected_metadata(session.policy_spec)
    assert metadata['method']=='sup_comparator' and metadata['base_policy']=='pi05_libero'
    assert metadata['sup_selector']['scheduler_sha256']==si.WEIGHTS[0.0175]
    assert metadata['campaign']['policy_seed']==0 and metadata['campaign']['controller_contract']==cp.CONTROLLER
    assert not (sup_session.parent/'budget.jsonl').exists()

@pytest.mark.parametrize('fault',['selection_open','capture','wrong_bank','challenger_purpose','missing_service_pin','wrong_epsilon'])
def test_sup_refuses_unmatched_or_unfrozen_session(sup_session,fault):
    spec=json.loads(sup_session.read_text())
    if fault=='selection_open':spec['selection_closed']=False
    elif fault=='capture':spec['scene_mode']='capture'
    elif fault=='challenger_purpose':spec['purpose']='candidate_validation'
    elif fault=='missing_service_pin':del spec['implementation_sources']['selector_service']
    elif fault=='wrong_bank':
        pp=Path(spec['policy_spec']['path']);p=json.loads(pp.read_text());p['ordered_episode_indices']=list(range(30,40));pp.write_text(json.dumps(p));spec['policy_spec']=asset(pp)
    else:
        pp=Path(spec['policy_spec']['path']);p=json.loads(pp.read_text());rp=Path(p['route_registry']['path']);r=json.loads(rp.read_text());r['routes'][0]['sup_epsilon']=0.02;rp.write_text(json.dumps(r));p['route_registry']=asset(rp);pp.write_text(json.dumps(p));spec['policy_spec']=asset(pp)
    sup_session.write_text(json.dumps(spec))
    with pytest.raises(ValueError):cs.CampaignSession(sup_session,repo=Path(cs.__file__).parents[3])
    assert not (sup_session.parent/'budget.jsonl').exists()

@pytest.mark.parametrize('key,bad',[('scheduler_sha256','0'*64),('epsilon',0.02),('action_protocol','legacy_v1'),('chunk_size',50),('state_dim',9),('source_sha256',{})])
def test_selector_handshake_refuses_mismatch(sup_session,key,bad):
    session=cs.CampaignSession(sup_session,repo=Path(cs.__file__).parents[3]);route=session.policy_spec.route
    correct=si.expected_selector_metadata(route);si.validate_selector_metadata(correct,route)
    correct[key]=bad
    with pytest.raises(ValueError):si.validate_selector_metadata(correct,route)

@pytest.mark.parametrize('fault',[None,'different_selected_stride','invalid_stride','phase_trace','wrong_composition'])
def test_sup_audit_terminal_mid_chunk_and_corruption(evidence,fault):
    rows,result,route,composer=copy.deepcopy(evidence);route.method='sup';route.uniform_stride=None
    rows[2]['decision']['selected_k']=2
    if fault=='different_selected_stride':rows[2]['decision']['selected_k']=1
    elif fault=='invalid_stride':rows[2]['decision']['selected_k']=3
    elif fault=='phase_trace':result['phase_decisions']=[{}]
    elif fault=='wrong_composition':rows[3]['action'][0]+=0.01
    if fault is None:
        assert audit_episode(rows,result,route=route,composer=composer)['passed']
    else:
        with pytest.raises(ValueError):audit_episode(rows,result,route=route,composer=composer)

@pytest.mark.parametrize('dtype',['float32','float64'])
def test_selector_state_and_chunk_match_july_server_without_normalization(dtype):
    import ast,hashlib,importlib.util
    july=AUDIT_ROOT/'remote-protocol-v2/scripts/serve_openpi_sup.py'
    tree=ast.parse(july.read_text());nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('_sha256','SuPPolicy')]
    ns={'np':np,'hashlib':hashlib,'IQLScheduler':object};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(july),'exec'),ns)
    service_path=Path(cs.__file__).parents[3]/'scripts/serve_sup_selector.py'
    loader=importlib.util.spec_from_file_location('selector_test_service',service_path);service=importlib.util.module_from_spec(loader);loader.loader.exec_module(service)
    class Selector:
        config=SimpleNamespace(k_values=(1,2),action_protocol='protocol_v2')
        def select_k(self,state,actions):
            self.seen=(state.copy(),actions.copy());return 2,{1:0.1,2:0.2}
    original=np.linspace(-.9,.7,70,dtype=dtype).reshape(10,7)
    raw_state=np.array([.31,-.62,.43,.14,.25,-.36,.017,.021],dtype=dtype)
    base=SimpleNamespace(metadata={},infer=lambda obs:{'actions':original})
    old_selector=Selector();old=ns['SuPPolicy'](base,old_selector,1,'',client_downsample=True,
        scheduler_path=str(AUDIT_ROOT/'scheduler-eps0175.pt'),scheduler_epsilon=.0175)
    response=old.infer({'observation/state':raw_state})
    new_selector=Selector();new=service.predict(new_selector,{'type':'predict_k','s_env':raw_state,'A':original})
    assert response['sup']['selected_k']==new['predicted_k']==2
    assert all(np.array_equal(a,b) for a,b in zip(old_selector.seen,new_selector.seen))
    assert np.array_equal(new_selector.seen[0],raw_state.astype(np.float32))
    assert np.array_equal(response['actions'],original)
    from examples.libero.speed_baselines.actions import uniform_slices
    scale=np.array([.05]*3+[.5]*3)
    from examples.libero.speed_baselines.actions import SupActionComposer
    composer=SupActionComposer(input_min=-np.ones(6),input_max=np.ones(6),output_min=-scale,output_max=scale)
    groups=uniform_slices(original,stride=new['predicted_k'],composer=composer)
    assert [i for group in groups for i in group.source_indices]==list(range(10))
    assert len(groups)==5 and all(group.stride==2 for group in groups)
