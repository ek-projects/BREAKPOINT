import json
from pathlib import Path
import pytest

from engine import DEMO, Experiment, Scenario, entropy, posterior, information_gain, validate, benchmark, run_trial
from active_probe_core import parse_cases, positive_median, evaluate
from http_lab import FAULTS, PROBES, RESPONSES, local_lab, probe
from demo_runtime import run_demo
from nebius_reasoner import NemotronClient, NebiusError, ProbeArchitect, ProbeDefinition


def test_engine_math_and_reproducibility():
    assert entropy((.5, .5)) == pytest.approx(1)
    assert sum(posterior(DEMO.experiments[0], DEMO.prior, True)) == pytest.approx(1)
    assert all(information_gain(e, DEMO.prior) >= 0 for e in DEMO.experiments)
    assert information_gain(Experiment('u', 1, (.3, .3, .3)), DEMO.prior) == pytest.approx(0)
    assert benchmark(DEMO, trials=100, seed=7) == benchmark(DEMO, trials=100, seed=7)


def test_engine_validation_and_budget():
    with pytest.raises(ValueError):
        validate(Scenario((.4, .4), (Experiment('x', 1, (.5, .5)),)))
    for policy in ('information', 'random', 'cheapest'):
        assert run_trial(DEMO, 1, policy, 2, 12)['spent'] <= 2


def _row(case,fault,service,probe,impaired,mag=0):
    return f"{case},{fault},{service},{probe},{probe},{probe},1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,{mag},False,False,1,{str(impaired)}"


def _fixture():
    probes=('p1','p2','p3','p4','p5')
    head='case,fault_type,target_service,journey_type,journey_label,signature,n_baseline,n_fault,baseline_p50,baseline_p95,baseline_p99,fault_p50,fault_p95,fault_p99,p95_ratio,baseline_throughput,fault_throughput,baseline_degraded_rate,fault_degraded_rate,degraded_rate_delta,baseline_fail_rate,fault_fail_rate,fail_rate_delta,p_value_raw,effect_size,impairment_magnitude,insufficient_data,short_window,p_value,impaired'
    lines=[head]
    for fault in ('cpu','mem'):
        for p in probes:
            lines += [_row(f'a_{fault}',fault,'a',p,p=='p1',2 if p=='p1' else 0), _row(f'b_{fault}',fault,'b',p,p=='p2',2 if p=='p2' else 0)]
    return probes, '\n'.join(lines)+'\n'


def test_active_probe_core_fixture():
    probes,text=_fixture(); train={('a','cpu'),('b','cpu')}; test={('a','mem'),('b','mem')}
    cases=parse_cases(text,probes,train|test)
    assert len(cases)==4
    assert positive_median([c for c in cases.values() if (c.service,c.fault) in train])==2
    out=evaluate(cases,train,test,probes,alpha=.5,budget=3)
    assert out['n_test_cases']==2 and 0 <= out['active_accuracy'] <= 1


def test_http_lab_is_loopback_and_deterministic():
    for fault in FAULTS:
        with local_lab(fault) as url:
            for name in PROBES:
                assert probe(url,name) == RESPONSES[fault][name]
    with pytest.raises(ValueError):
        probe('https://example.org','db_probe')


def test_demo_deterministic_mode():
    for fault in FAULTS:
        out=run_demo(fault, client=None)
        assert out['correct'] is True and out['architect_attempts']==0
        assert all(not step['fallback_used'] for step in out['steps'])


def _transport(content=None, status=200):
    content = content or {'hypotheses':[],'shortlist':['probe-a'],'rationale':'test'}
    def call(url, body, headers, timeout):
        req=json.loads(body)
        assert req['max_tokens']==192
        return status, json.dumps({'model':req['model'],'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],'usage':{}})
    return call


def test_nebius_client_and_whitelist():
    with pytest.raises(NebiusError):
        NemotronClient(api_key='', transport=_transport()).chat_json('x',{})
    payload={'hypotheses':[],'shortlist':['evil','probe-b','probe-a'],'rationale':'x'}
    p=ProbeArchitect(NemotronClient(api_key='secret',transport=_transport(payload))).propose({}, {'svc':1.0}, {}, [ProbeDefinition('probe-a','a'),ProbeDefinition('probe-b','b')], 2)
    assert p.shortlist == ('probe-b','probe-a')


def test_confirmatory_freeze_and_result():
    freeze=json.loads(Path('CONFIRMATORY_FREEZE_2026-10-13.json').read_text())
    result=json.loads(Path('external_tt_confirmatory_results.json').read_text())
    assert freeze['frozen_before_train_ticket_outcomes'] is True and freeze['no_retuning_after_result'] is True
    assert freeze['probe_budget']==3 and result['n_test_cases']==30
    assert result['active_accuracy']==22/30 and result['static_accuracy']==18/30
