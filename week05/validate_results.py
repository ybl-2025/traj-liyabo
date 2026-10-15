"""Independent checks for the fixed 21/13 quality-screened experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from prepare_data import ROOT,sha256


def validate(out):
    out=Path(out);e=pd.read_csv(out/'episodes.csv');d=pd.read_csv(out/'following_observations.csv')
    p=pd.read_csv(out/'ngsim_predictions.csv');m=pd.read_csv(out/'ngsim_episode_metrics.csv')
    cfg=json.loads((ROOT/'data/selection_config.json').read_text());checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:raise AssertionError(name)
    check('21_train_13_test',e.split.value_counts().to_dict()=={'train':21,'test':13})
    check('34_distinct_followers',len(e)==34 and e.follower_id.is_unique)
    check('301_points_per_episode',len(d)==10234 and d.groupby('episode_id').size().eq(301).all())
    check('30_second_duration',np.allclose(e.end_s-e.start_s,30))
    check('continuous_sampling',all(np.allclose(np.diff(w.time_s),.1) for _,w in d.groupby('episode_id')))
    check('5_second_guards',np.allclose(e.start_s-e.guard_start_s,5) and np.allclose(e.guard_end_s-e.end_s,5))
    train=e[e.split=='train'];test=e[e.split=='test']
    check('vehicle_disjoint_splits',not ((set(train.follower_id)|set(train.leader_id))&(set(test.follower_id)|set(test.leader_id))))
    check('time_separation_including_guards',train.guard_end_s.max()<=250 and test.guard_start_s.min()>=300 and test.guard_end_s.max()<=500)
    check('raw_acceleration_quality',e.raw_accel_max_m_s2.le(cfg['max_raw_accel_m_s2']).all())
    check('raw_kinematics_quality',e.raw_kinematic_rmse_m_s.le(cfg['max_raw_kinematic_rmse_m_s']).all())
    check('position_adjustment_bounded',max(abs(d.observed_x_m-d.raw_observed_x_m).max(),abs(d.leader_x_m-d.raw_leader_x_m).max())<=.300001)
    check('nonnegative_reference_speed',d[['observed_v_m_s','leader_v_m_s']].ge(0).all().all())
    check('positive_reference_gap',d.gap_m.ge(2-1e-7).all())
    check('net_gap_geometry',np.allclose(d.leader_x_m-d.leader_length_m-d.observed_x_m,d.gap_m,atol=1e-7))
    check('204_rollouts_all_present',len(m)==204 and len(p)==204*301 and not p.duplicated(['episode_id','model','parameter_set','time_s']).any())
    sources={eid:w for eid,w in d.groupby('episode_id')}
    mm=m.set_index(['episode_id','parameter_set','model']);scores=[];raw_scores=[];initial=[];gaps=[]
    for (eid,pname,model),w in p.groupby(['episode_id','parameter_set','model']):
        source=sources[eid];row=mm.loc[eid,pname,model]
        initial.append(abs(w.follower_x_m.iloc[0]-source.observed_x_m.iloc[0])<1e-6 and abs(w.follower_v_m_s.iloc[0]-source.observed_v_m_s.iloc[0])<1e-6)
        gaps.append(np.allclose(source.leader_x_m.to_numpy()-source.leader_length_m.to_numpy()-w.follower_x_m.to_numpy(),w.gap_m,atol=1e-6,equal_nan=True))
        if row.status=='success':
            ex=w.follower_x_m.to_numpy()[1:]-source.observed_x_m.to_numpy()[1:]
            ev=w.follower_v_m_s.to_numpy()[1:]-source.observed_v_m_s.to_numpy()[1:]
            scores.append(abs(np.sqrt(np.mean(ex**2))-row.position_rmse_m)<1e-6 and abs(np.sqrt(np.mean(ev**2))-row.speed_rmse_m_s)<1e-6)
            raw=w.follower_x_m.to_numpy()[1:]-source.raw_observed_x_m.to_numpy()[1:]
            raw_scores.append(abs(np.sqrt(np.mean(raw**2))-row.raw_position_rmse_m)<1e-6)
        else:
            scores.append(pd.isna(row.position_rmse_m) and w[w.time_s>=row.failure_time_s].follower_x_m.isna().all())
    check('all_initial_states_match',all(initial));check('prediction_gap_geometry',all(gaps))
    check('independent_rmse_and_failure_accounting',all(scores));check('original_position_rmse_recalculated',all(raw_scores))
    check('213_step_size_checks',len(pd.read_csv(out/'step_size_check.csv'))==213)
    check('five_png_artifacts',all((out/name).read_bytes().startswith(b'\x89PNG') for name in ['summary.png','ngsim_comparison.png','ngsim_trajectories.png','data_quality.png','failure_analysis.png']))
    params=json.loads((out/'parameters.json').read_text())
    check('training_count_in_calibration_record','21 train episodes' in params['loss'])
    before=ROOT/'data/week4_integrity_before.json'
    if before.exists():
        original=json.loads(before.read_text(encoding='utf-8'));folder=ROOT.parent/'week4'
        if original and folder.is_dir():
            changed=[name for name,digest in original.items() if not (folder/name).is_file() or sha256(folder/name)!=digest]
            added=[str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file() and str(f.relative_to(folder)) not in original]
            integrity=dict(unchanged=not changed and not added,checked_files=len(original),changed_or_missing=changed,added=added)
            (out/'week4_integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf-8')
            check('week4_unchanged',integrity['unchanged'])
    payload=dict(passed=True,checks_count=len(checks),checks=checks)
    (out/'validation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'results'))
    args=parser.parse_args();r=validate(args.output);print(f"PASS: {r['checks_count']} result checks")
