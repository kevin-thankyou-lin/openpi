from types import SimpleNamespace
import numpy as np
import pytest
from examples.libero.speed_baselines.predictor_refresh import PrefixPredictor, RefreshableActionQueue, refresh_interval

class Selector:
    def __init__(self, speed=1):
        self.speed=speed
        self.calls=[]
    def plan(self, actions, *, task_id):
        return self.plan_suffix(actions,task_id=task_id,source_offset=0)
    def plan_suffix(self, actions, *, task_id, source_offset):
        self.calls.append((source_offset,actions.copy()))
        slices=[]
        for i in range(0,len(actions),self.speed):
            end=min(i+self.speed,len(actions))
            slices.append(SimpleNamespace(source_indices=tuple(range(i+source_offset,end+source_offset)),stride=end-i,action=actions[i:end].sum(axis=0)))
        return slices, {'source_offset':source_offset}

@pytest.mark.parametrize('hz,want',[(0,None),(4,5),(6,3),(20,1)])
def test_interval(hz,want): assert refresh_interval(hz)==want

@pytest.mark.parametrize('hz',[-1,21,float('nan'),float('inf')])
def test_invalid_interval(hz):
    with pytest.raises(ValueError): refresh_interval(hz)

def test_slow_phase_refresh_changes_only_unconsumed_actions():
    q=RefreshableActionQueue(4);s=Selector();a=np.arange(10)[:,None]
    q.start(a,s,task_id=8,env_step=0)
    consumed=[]
    for tick in range(5):
        assert q.refresh(s,task_id=8,env_step=tick) is None
        consumed.extend(q.popleft().source_indices)
    s.speed=2
    assert q.refresh(s,task_id=8,env_step=5)['source_offset']==5
    assert np.array_equal(s.calls[-1][1],a[5:])
    while q: consumed.extend(q.popleft().source_indices)
    assert consumed==list(range(10))
    assert len(s.calls)==2

def test_uniform2_has_no_extra_predictions_and_same_actions():
    s=Selector(2);q=RefreshableActionQueue(4);a=np.arange(10)[:,None]
    q.start(a,s,task_id=8,env_step=0)
    executed=[]
    for t in range(5):
        assert q.refresh(s,task_id=8,env_step=t) is None
        executed.append(q.popleft().action.item())
    assert not q and executed==[1,5,9,13,17] and len(s.calls)==1

def test_disabled_keeps_full_slow_chunk():
    s=Selector();q=RefreshableActionQueue(0);q.start(np.arange(10),s,task_id=8,env_step=0)
    for t in range(10):
        assert q.refresh(s,task_id=8,env_step=t) is None
        q.popleft()
    assert len(s.calls)==1

def test_fresh_prefix_starts_at_row_zero_not_old_offset():
    from strider.subtask_prediction import SubtaskPrediction
    class P:
        horizon=10
        class_names=('a','b')
        def predict(self, obs):
            assert obs==['latest']
            return SubtaskPrediction(tuple([(1.,0.)]*3+[(0.,1.)]*7),self.class_names,(True,)*10)
    p=PrefixPredictor(P(),3).predict(['latest'])
    assert p.horizon==3 and p.labels==('a','a','a')

def test_rejects_skipped_suffix():
    s=Selector();q=RefreshableActionQueue(4);q.start(np.arange(10),s,task_id=8,env_step=0)
    q.pending[0].source_indices=(1,)
    with pytest.raises(ValueError,match='skipped or repeated'): q.popleft()


def test_selector_refresh_uses_latest_prediction_and_absolute_source_indices():
    import strider
    from examples.libero.speed_baselines.strider_client import StriderPhaseSelector
    from strider.subtask_prediction import SubtaskPrediction
    class P:
        horizon=10
        class_names=('slow','fast')
        def predict(self, obs):
            assert obs==['fresh']
            return SubtaskPrediction(((0.,1.),)*10,self.class_names,(True,)*10)
    class Composer:
        def compose(self, actions): return np.sum(actions,axis=0)
    s=object.__new__(StriderPhaseSelector)
    s.task_id=8;s.predictor=P();s._observations=['fresh']
    s.phase_speeds={'task_08:slow':1,'task_08:fast':2}
    s._runtime_schedule=strider.SubtaskSpeedSchedule({'slow':1,'fast':2})
    s._runtime_pipeline=strider.PlanPipeline(retimer=strider.BoundaryAwareStrideRetimer(Composer(),allowed_speeds=(1,2,3)))
    a=np.arange(35).reshape(5,7)
    slices,decision=s.plan_suffix(a,task_id=8,source_offset=5)
    assert [x.source_indices for x in slices]==[(5,6),(7,8),(9,)]
    assert np.array_equal(slices[0].action,a[:2].sum(axis=0))
    assert decision['speed_factors']==[2.]*5
    assert slices[0].metadata['phases']==('task_08:fast','task_08:fast')
