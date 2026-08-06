from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from paper1_qc_reviewed.qrev_v400 import (
    ANALYSIS_FEATURES, DEFAULT_PARAMETERS, MEASUREMENT_VERSION,
    SRMR_PINNED_REGRESSION_VALUE, SpeechInterval, ac_rms_dbfs,
    apply_gain_db, compute_srmr_norm, extract_qrev,
    feature_registry_frame, internal_pause_boundaries, remove_global_dc,
    validate_speech_intervals,
)
FS=16000

def task(speech_count=5,speech_sec=.8,pause_sec=1.2,tail_level=.0,tau=.12,plateau=None,floor_level=2e-5,seed=12):
    rng=np.random.default_rng(seed); total=speech_count*speech_sec+(speech_count-1)*pause_sec
    y=floor_level*rng.standard_normal(round(total*FS)); primary=[]; strict=[]; cursor=0.0
    for i in range(speech_count):
        start=cursor; end=start+speech_sec
        primary.append(SpeechInterval(start,end,f"p{i}",i,"primary_speech","primary"))
        strict.append(SpeechInterval(start+.05,end-.05,f"s{i}",i,"strict_speech","primary"))
        l=round(start*FS); r=round(end*FS); t=np.arange(r-l)/FS
        y[l:r]+=0.04*(np.sin(2*np.pi*173*t)+.5*np.sin(2*np.pi*421*t+.3))
        if i<speech_count-1 and tail_level>0:
            pr=round((end+pause_sec)*FS); rel=np.arange(pr-r)/FS; carrier=rng.standard_normal(len(rel))
            env=tail_level*(rel<plateau) if plateau is not None else tail_level*np.exp(-rel/tau)
            y[r:pr]+=env*carrier
        cursor=end+(pause_sec if i<speech_count-1 else 0)
    return y,primary,strict

def extract(y,p,s,**kw): return extract_qrev(y,FS,primary_speech=p,strict_speech=s,logical_recording_id="test",compute_srmr=False,**kw)

def test_version_registry_and_no_scalar():
    assert MEASUREMENT_VERSION=="qrev-v4.0.0-candidate"
    r=feature_registry_frame(); assert tuple(r.feature)==ANALYSIS_FEATURES and len(r)==4
    assert r.family_scalar_prohibited.all() and r.standalone_gate_prohibited.all()

def test_primary_boundaries_not_strict_boundaries():
    y,p,s=task(); b=internal_pause_boundaries(p,len(y)/FS)
    assert b[0]["speech_offset_sec"]==pytest.approx(p[0].end_sec)
    assert b[0]["speech_offset_sec"]-s[0].end_sec==pytest.approx(.05)
    assert b[0]["pause_duration_sec"]==pytest.approx(1.2)
    assert b[0]["left_interval_id"]=="p0" and b[0]["right_interval_id"]=="p1"

def test_wrong_view_is_rejected():
    y,p,s=task()
    with pytest.raises(ValueError,match="primary_speech"):
        extract_qrev(y,FS,primary_speech=s,strict_speech=s,compute_srmr=False)

def test_overlap_and_duplicate_identity_rejected():
    with pytest.raises(ValueError,match="overlap"):
        validate_speech_intervals([SpeechInterval(0,1,"a",view="primary_speech"),SpeechInterval(.9,2,"b",view="primary_speech")],3,required_view="primary_speech")
    with pytest.raises(ValueError,match="Duplicate"):
        validate_speech_intervals([SpeechInterval(0,1,"a",view="primary_speech"),SpeechInterval(2,3,"a",view="primary_speech")],4,required_view="primary_speech")

def test_ac_rms_dc_invariance_and_floor():
    t=np.arange(FS)/FS; x=.1*np.sin(2*np.pi*200*t)
    a,_=ac_rms_dbfs(x); b,_=ac_rms_dbfs(x+.5); assert b==pytest.approx(a,abs=1e-12)
    z,f=ac_rms_dbfs(np.zeros(100)); assert f and z==-120

def test_signed_tail_not_clipped():
    y,p,s=task(floor_level=1e-5); rng=np.random.default_rng(4)
    for b in internal_pause_boundaries(p,len(y)/FS):
        l=round((b['speech_offset_sec']+.7)*FS); r=round((b['speech_offset_sec']+1.0)*FS); y[l:r]+=.003*rng.standard_normal(r-l)
    assert extract(y,p,s).recording[ANALYSIS_FEATURES[0]]<0

def test_gain_polarity_and_dc_invariance():
    y,p,s=task(tail_level=.01,plateau=.18); base=extract(y,p,s).recording
    variants=[apply_gain_db(y,9),-y,y+.25]
    for v in variants:
        got=extract(v,p,s).recording
        for f in ANALYSIS_FEATURES[:3]:
            if np.isnan(base[f]): assert np.isnan(got[f])
            else: assert got[f]==pytest.approx(base[f],abs=1e-8)

def test_persistence_dose_order_and_recovery():
    doses=[.08,.16,.24]; measured=[]
    for d in doses:
        y,p,s=task(tail_level=.01,plateau=d,seed=33); measured.append(extract(y,p,s).recording[ANALYSIS_FEATURES[1]])
    assert measured==sorted(measured); assert np.max(np.abs(np.array(measured)-doses))<=.04

def test_decay_slope_recovery():
    tau=.12; y,p,s=task(tail_level=.02,tau=tau,floor_level=2e-6,seed=50)
    got=extract(y,p,s).recording[ANALYSIS_FEATURES[2]]; expected=20/np.log(10)/tau
    assert got==pytest.approx(expected,rel=.16)

def test_nondecaying_is_missing_not_zero():
    y,p,s=task(tail_level=.01,plateau=1.0); r=extract(y,p,s).recording
    assert np.isnan(r[ANALYSIS_FEATURES[2]]) and r[f"{ANALYSIS_FEATURES[2]}_status"]=="no_valid_downward_decay"

def test_provisional_two_boundary_support_and_raw_values():
    y,p,s=task(speech_count=3,tail_level=.01,plateau=.15); r=extract(y,p,s).recording
    assert r['qrev_tail_valid_boundary_count']==2 and np.isfinite(r[ANALYSIS_FEATURES[0]])
    strict=replace(DEFAULT_PARAMETERS,minimum_tail_boundary_count=4,minimum_persistence_boundary_count=4,minimum_decay_boundary_count=4)
    r2=extract(y,p,s,parameters=strict).recording
    assert np.isnan(r2[ANALYSIS_FEATURES[0]]) and np.isfinite(r2[f"{ANALYSIS_FEATURES[0]}_raw_estimate"])

def test_boundary_ledger_reconstructs():
    y,p,s=task(tail_level=.012,plateau=.18); out=extract(y,p,s); led=out.boundary_ledger
    assert np.median(led.loc[led.tail_eligible,'tail_excess_100ms_db'])==pytest.approx(out.recording[ANALYSIS_FEATURES[0]])
    assert set(led.boundary_source_view)=={'primary_speech'}

def test_deterministic():
    y,p,s=task(tail_level=.01,plateau=.15); a=extract(y,p,s); b=extract(y,p,s)
    assert a.recording==b.recording; pd.testing.assert_frame_equal(a.boundary_ledger,b.boundary_ledger)

def test_sample_rate_mono_finite_contracts():
    y,p,s=task()
    with pytest.raises(ValueError,match="16000"): extract_qrev(y[::2],8000,primary_speech=p,strict_speech=s,compute_srmr=False)
    with pytest.raises(ValueError,match="mono"): extract_qrev(np.zeros((10,2)),FS,primary_speech=[],strict_speech=[],compute_srmr=False)
    with pytest.raises(ValueError,match="non-finite"): extract_qrev(np.array([0.,np.nan]),FS,primary_speech=[],strict_speech=[],compute_srmr=False)

def test_srmr_pinned_runtime_when_dependency_available():
    pytest.importorskip("gammatone")
    fixture=Path(__file__).resolve().parents[1]/"tests"/"fixtures"/"srmrpy"/"test.mat"
    if not fixture.exists(): pytest.skip("repository fixture unavailable in isolated patch test")
    sample=loadmat(fixture)['s'][:,0]; got=compute_srmr_norm(sample,FS)
    assert got==pytest.approx(SRMR_PINNED_REGRESSION_VALUE,rel=1e-10,abs=1e-12)


def deterministic_task_at_fs(fs: int, speech_count: int = 5, speech_sec: float = .8, pause_sec: float = 1.2, tau: float = .15):
    total=speech_count*speech_sec+(speech_count-1)*pause_sec
    y=np.zeros(round(total*fs),dtype=float); primary=[]; strict=[]; cursor=0.0
    for i in range(speech_count):
        start=cursor; end=start+speech_sec
        primary.append(SpeechInterval(start,end,f"p{i}",i,"primary_speech","primary"))
        strict.append(SpeechInterval(start+.05,end-.05,f"s{i}",i,"strict_speech","primary"))
        l=round(start*fs); r=round(end*fs); t=np.arange(r-l)/fs
        y[l:r]+=.04*(np.sin(2*np.pi*173*t)+.4*np.sin(2*np.pi*421*t+.3))
        if i<speech_count-1:
            pr=round((end+pause_sec)*fs); rel=np.arange(pr-r)/fs
            y[r:pr]+=.012*np.exp(-rel/tau)*(np.sin(2*np.pi*257*rel)+.35*np.sin(2*np.pi*619*rel+.2))
        cursor=end+(pause_sec if i<speech_count-1 else 0)
    # deterministic low floor remains well below the tail but avoids all-zero late windows
    t=np.arange(len(y))/fs
    y+=2e-6*(np.sin(2*np.pi*997*t)+.5*np.sin(2*np.pi*1301*t+.1))
    return y,primary,strict


def test_common_time_shift_invariance():
    from paper1_qc_reviewed.qrev_v400 import shift_intervals
    y,p,s=task(tail_level=.01,plateau=.18)
    base=extract(y,p,s).recording
    shift=.37; n=round(shift*FS)
    shifted_wave=np.pad(y,(n,0))
    got=extract_qrev(
        shifted_wave,FS,
        primary_speech=shift_intervals(p,shift),
        strict_speech=shift_intervals(s,shift),
        logical_recording_id="shifted",compute_srmr=False,
    ).recording
    for f in ANALYSIS_FEATURES[:3]:
        assert got[f]==pytest.approx(base[f],abs=1e-8)


def test_source_rate_to_16k_resampling_contract():
    from scipy import signal
    y16,p,s=deterministic_task_at_fs(16000)
    y48,_,_=deterministic_task_at_fs(48000)
    down=signal.resample_poly(y48,1,3)
    if len(down)<len(y16): down=np.pad(down,(0,len(y16)-len(down)))
    down=down[:len(y16)]
    a=extract(y16,p,s).recording; b=extract(down,p,s).recording
    assert abs(a[ANALYSIS_FEATURES[0]]-b[ANALYSIS_FEATURES[0]])<=.15
    assert abs(a[ANALYSIS_FEATURES[1]]-b[ANALYSIS_FEATURES[1]])<=.02
    # Decay is a conditional frame-envelope slope; characterize but require the same eligibility state.
    assert np.isfinite(a[ANALYSIS_FEATURES[2]])==np.isfinite(b[ANALYSIS_FEATURES[2]])
    if np.isfinite(a[ANALYSIS_FEATURES[2]]):
        assert abs(a[ANALYSIS_FEATURES[2]]-b[ANALYSIS_FEATURES[2]])/a[ANALYSIS_FEATURES[2]]<=.12


def test_support_amounts_and_missingness_are_explicit():
    y,p,s=task(speech_count=3,tail_level=.01,plateau=.15)
    r=extract(y,p,s).recording
    assert r["qrev_tail_valid_pause_support_sec"]>=2.0
    assert r["qrev_persistence_valid_pause_support_sec"]>=2.0
    assert r["qrev_persistence_observed_duration_support_sec"]>0
    assert r["qrev_decay_valid_pause_support_sec"]>=0


def test_persistence_horizon_precedes_independent_floor_and_censors():
    assert DEFAULT_PARAMETERS.persistence_horizon_ms < DEFAULT_PARAMETERS.floor_start_ms
    y,p,s=task(tail_level=.01,plateau=.65)
    r=extract(y,p,s).recording
    assert r[ANALYSIS_FEATURES[1]]==pytest.approx(.6,abs=.011)
    assert r['qrev_persistence_recording_median_censored']
    assert r[f'{ANALYSIS_FEATURES[1]}_status']=='right_censored_at_horizon'


def test_global_dc_removal_contract():
    t=np.arange(FS)/FS
    x=.1*np.sin(2*np.pi*200*t)+.03*np.sin(2*np.pi*421*t+.2)
    centered=remove_global_dc(x+.25)
    reference=remove_global_dc(x)
    assert np.mean(centered)==pytest.approx(0.0,abs=1e-14)
    assert centered==pytest.approx(reference,abs=1e-14)


def test_extract_qrev_applies_global_dc_removal_before_srmr(monkeypatch):
    duration=4.0
    t=np.arange(round(duration*FS))/FS
    x=.04*np.sin(2*np.pi*173*t)+.01
    primary=[SpeechInterval(0.0,duration,"p0",0,"primary_speech","primary")]
    strict=[SpeechInterval(.05,duration-.05,"s0",0,"strict_speech","primary")]
    captured={}
    def fake_srmr(values, fs, *, parameters=DEFAULT_PARAMETERS):
        captured["mean"]=float(np.mean(values))
        captured["fs"]=int(fs)
        return 2.5
    monkeypatch.setattr("paper1_qc_reviewed.qrev_v400.compute_srmr_norm",fake_srmr)
    result=extract_qrev(
        x+.25,FS,
        primary_speech=primary,
        strict_speech=strict,
        logical_recording_id="dc_contract",
        compute_srmr=True,
    ).recording
    assert captured["fs"]==FS
    assert captured["mean"]==pytest.approx(0.0,abs=1e-14)
    assert result["qrev_global_dc_removal_applied"] is True
    assert result["qrev_input_mean_before_dc_removal"]==pytest.approx(.26,abs=1e-4)
    assert result["qrev_input_mean_after_dc_removal"]==pytest.approx(0.0,abs=1e-14)
    assert result[ANALYSIS_FEATURES[3]]==pytest.approx(2.5)
