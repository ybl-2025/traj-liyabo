"""Quality-first NGSIM selection; source is read-only; all outputs stay in week5.

Selection uses observations only, before any model is simulated. Position spline
denoising is NOT the Montanino/Punzo reconstructed dataset or their algorithm.
"""
from pathlib import Path
import hashlib
import gzip
import json
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

ROOT=Path(__file__).resolve().parent
FILENAME='NGSIM_I80_20050413_1600-1615_SI.csv'


def sha256(path):
    """Hash CSV.gz by its uncompressed contents to match source provenance."""
    digest=hashlib.sha256()
    opener=gzip.open if Path(path).suffix=='.gz' else open
    with opener(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def locate_data(explicit=None):
    paths=[Path(explicit)] if explicit else [ROOT/'data'/FILENAME,ROOT/'data'/(FILENAME+'.gz'),
        ROOT.parent/'week4'/'第4周_NGSIM-I80_课堂数据'/FILENAME,
        ROOT.parent/'week04'/'data'/FILENAME]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError('Place the classroom SI CSV or CSV.gz in data/ or pass --data PATH')


def kinematic_errors(x,v,t):
    return np.diff(x)/np.diff(t)-.5*(np.asarray(v)[:-1]+np.asarray(v)[1:])


def observation_quality(w,cfg):
    """Raw-signal tests on the entire extended window, including both guards."""
    t=w.time_s.to_numpy()
    lengths=np.r_[w.vehicle_length_m,w.leader_length_m]
    result=dict(length_ok=bool(np.all((lengths>=cfg['min_vehicle_length_m'])&(lengths<=cfg['max_vehicle_length_m']))),
        raw_gap_min_m=float(w.gap_m.min()),raw_gap_max_m=float(w.gap_m.max()),
        lateral_span_m=float(max(np.ptp(w.lateral_residual_m),np.ptp(w.leader_lateral_residual_m))),
        lateral_offset_max_m=float(np.max(abs(w.lateral_residual_m-w.leader_lateral_residual_m))),
        raw_accel_max_m_s2=float(max(np.max(abs(np.diff(w.speed_m_s)/np.diff(t))),np.max(abs(np.diff(w.leader_v_m_s)/np.diff(t))))),
        raw_kinematic_rmse_m_s=0.,raw_kinematic_p99_m_s=0.)
    for x,v in ((w.local_y_m,w.speed_m_s),(w.leader_x_m,w.leader_v_m_s)):
        error=kinematic_errors(x,v,t)
        result['raw_kinematic_rmse_m_s']=max(result['raw_kinematic_rmse_m_s'],float(np.sqrt(np.mean(error**2))))
        result['raw_kinematic_p99_m_s']=max(result['raw_kinematic_p99_m_s'],float(np.quantile(abs(error),.99)))
    guard=int(round(cfg['guard_s']*10)); core=w.iloc[guard:len(w)-guard]
    moving=core.speed_m_s>2.
    result['median_time_headway_s']=float(np.median(core.gap_m[moving]/core.speed_m_s[moving])) if moving.any() else np.inf
    result['mean_speed_m_s']=float(core.speed_m_s.mean())
    result['raw_speed_range_m_s']=float(max(np.ptp(core.speed_m_s),np.ptp(core.leader_v_m_s)))
    rules=dict(vehicle_length=result['length_ok'],
        gap_range=result['raw_gap_min_m']>=cfg['min_gap_m'] and result['raw_gap_max_m']<=cfg['max_gap_m'],
        lateral_stability=result['lateral_span_m']<=cfg['max_lateral_span_m'] and result['lateral_offset_max_m']<=cfg['max_lateral_offset_m'],
        raw_acceleration=result['raw_accel_max_m_s2']<=cfg['max_raw_accel_m_s2'],
        raw_kinematics=result['raw_kinematic_rmse_m_s']<=cfg['max_raw_kinematic_rmse_m_s'] and result['raw_kinematic_p99_m_s']<=cfg['max_raw_kinematic_p99_m_s'],
        following_relevance=result['median_time_headway_s']<=cfg['max_median_time_headway_s'] and result['mean_speed_m_s']>=cfg['min_mean_speed_m_s'],
        dynamic_information=result['raw_speed_range_m_s']>=cfg['min_speed_range_m_s'])
    result['rejection_reasons']=';'.join(k for k,passed in rules.items() if not passed)
    return result


def reconstruct(w,cfg):
    """Small position-only smoothing and analytic derivatives, without model fits.

    Include guards in fitting then remove them. Reject negative speeds or excessive
    corrections rather than clipping. Raw and processed signals are both exported.
    """
    t=w.time_s.to_numpy()-w.time_s.iloc[0]
    guard=int(round(cfg['guard_s']*10)); keep=slice(guard,len(w)-guard)
    z=w.iloc[keep].copy()
    diagnostics=dict(position_adjustment_max_m=0.,speed_adjustment_rmse_m_s=0.,
        reconstructed_accel_max_m_s2=0.,reconstructed_jerk_p95_m_s3=0.,
        reconstructed_min_speed_m_s=np.inf,reconstructed_kinematic_rmse_m_s=0.)
    for prefix,pos,vel in [('observed','local_y_m','speed_m_s'),('leader','leader_x_m','leader_v_m_s')]:
        spline=UnivariateSpline(t,w[pos].to_numpy(),k=3,s=len(w)*cfg['spline_rms_position_budget_m']**2)
        tt=t[keep]
        x,v,a,jerk=[spline.derivative(k)(tt) if k else spline(tt) for k in range(4)]
        raw_x=w[pos].to_numpy()[keep]; raw_v=w[vel].to_numpy()[keep]
        z[f'raw_{prefix}_x_m']=raw_x; z[f'raw_{prefix}_v_m_s']=raw_v
        z[f'{prefix}_x_m']=x; z[f'{prefix}_v_m_s']=v; z[f'{prefix}_a_m_s2']=a
        diagnostics['position_adjustment_max_m']=max(diagnostics['position_adjustment_max_m'],float(np.max(abs(x-raw_x))))
        diagnostics['speed_adjustment_rmse_m_s']=max(diagnostics['speed_adjustment_rmse_m_s'],float(np.sqrt(np.mean((v-raw_v)**2))))
        diagnostics['reconstructed_accel_max_m_s2']=max(diagnostics['reconstructed_accel_max_m_s2'],float(np.max(abs(a))))
        diagnostics['reconstructed_jerk_p95_m_s3']=max(diagnostics['reconstructed_jerk_p95_m_s3'],float(np.quantile(abs(jerk),.95)))
        diagnostics['reconstructed_min_speed_m_s']=min(diagnostics['reconstructed_min_speed_m_s'],float(v.min()))
        diagnostics['reconstructed_kinematic_rmse_m_s']=max(diagnostics['reconstructed_kinematic_rmse_m_s'],float(np.sqrt(np.mean(kinematic_errors(x,v,tt)**2))))
    z['raw_gap_m']=z.gap_m
    z['gap_m']=z.leader_x_m-z.leader_length_m-z.observed_x_m
    rules=dict(position_correction=diagnostics['position_adjustment_max_m']<=cfg['max_position_adjustment_m'],
        speed_correction=diagnostics['speed_adjustment_rmse_m_s']<=cfg['max_reconstructed_speed_change_rmse_m_s'],
        processed_acceleration=diagnostics['reconstructed_accel_max_m_s2']<=cfg['max_reconstructed_accel_m_s2'],
        processed_jerk=diagnostics['reconstructed_jerk_p95_m_s3']<=cfg['max_reconstructed_jerk_p95_m_s3'],
        nonnegative_speed=diagnostics['reconstructed_min_speed_m_s']>=0,
        processed_gap=bool((z.gap_m>=cfg['min_gap_m']).all() and (z.gap_m<=cfg['max_gap_m']).all()))
    diagnostics['rejection_reasons']=';'.join(k for k,p in rules.items() if not p)
    return z,diagnostics


def sample_stratified(pool,maximum):
    """Lane/time coverage, at most one window per follower; no prediction errors."""
    pool=pool.sort_values(['start_s','follower_id']).drop_duplicates('follower_id')
    groups=[g.reset_index(drop=True) for _,g in pool.groupby('lane_id',sort=True)]
    if not groups:
        return pool
    quota=int(np.ceil(maximum/len(groups))); queues=[]
    for g in groups:
        indices=np.linspace(0,len(g)-1,min(quota,len(g))).round().astype(int)
        queues.append([g.iloc[i] for i in indices])
    selected=[]
    while any(queues) and len(selected)<maximum:
        for q in queues:
            if q and len(selected)<maximum:
                selected.append(q.pop(0))
    return pd.DataFrame(selected).sort_values(['start_s','lane_id','follower_id'])


def prepare(path,out,config_path=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    config_path=Path(config_path or ROOT/'data'/'selection_config.json')
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    d=pd.read_csv(path)
    required=['vehicle_id','frame_id','time_s','lane_id','local_y_m','local_x_m','vehicle_length_m','vehicle_class','speed_m_s']
    if not set(required)<=set(d):
        raise ValueError('Input must be the classroom SI CSV including lateral positions')
    if d.duplicated(['vehicle_id','frame_id']).any():
        raise ValueError('Duplicate vehicle/frame keys')
    raw_rows=len(d); lane_counts=d.groupby('lane_id').size().to_dict()
    d=d[d.lane_id.isin(cfg['lanes'])].sort_values(['frame_id','lane_id','local_y_m']).copy()
    # Remove road curvature before testing lateral stability. These robust median
    # centerlines describe fixed road geometry, not a car-following prediction.
    geometry=[]
    d['lateral_residual_m']=np.nan
    for lane,g in d.groupby('lane_id'):
        med=g.groupby((g.local_y_m/25).astype(int)).agg(y=('local_y_m','median'),x=('local_x_m','median'))
        coeff=np.polyfit(med.y,med.x,3)
        d.loc[g.index,'lateral_residual_m']=g.local_x_m-np.polyval(coeff,g.local_y_m)
        geometry.append(dict(lane_id=int(lane),polynomial_coefficients=list(map(float,coeff))))
    (out/'lane_geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
    grouped=d.groupby(['frame_id','lane_id'],sort=False)
    mapping={'vehicle_id':'leader_id','local_y_m':'leader_x_m','speed_m_s':'leader_v_m_s',
        'vehicle_length_m':'leader_length_m','vehicle_class':'leader_class','local_x_m':'leader_lateral_m',
        'lateral_residual_m':'leader_lateral_residual_m'}
    for old,new in mapping.items():
        d[new]=grouped[old].shift(-1)
    d['gap_m']=d.leader_x_m-d.leader_length_m-d.local_y_m
    finite=np.isfinite(d[required+list(mapping.values())].to_numpy(float)).all(axis=1)
    d=d[finite&(d.vehicle_class==2)&(d.leader_class==2)&(d.gap_m>0)&(d.speed_m_s>=0)&(d.leader_v_m_s>=0)].sort_values(['vehicle_id','frame_id']).copy()
    prev=d.shift()
    breaks=((d.vehicle_id!=prev.vehicle_id)|(d.leader_id!=prev.leader_id)|(d.lane_id!=prev.lane_id)|
        (d.frame_id-prev.frame_id!=1)|~np.isclose(d.time_s-prev.time_s,.1))
    d['run']=breaks.cumsum(); runs=d.groupby('run',sort=False)
    points=int(round((cfg['duration_s']+2*cfg['guard_s'])*10))+1
    stride=int(round(cfg['candidate_stride_s']*10)); rows=[]; arrays={}
    for _,g in runs:
        for start in range(0,len(g)-points+1,stride):
            w=g.iloc[start:start+points]
            a,b=float(w.time_s.iloc[0]),float(w.time_s.iloc[-1])
            split='train' if b<=cfg['train_end_s'] else 'test' if a>=cfg['test_start_s'] and b<=cfg['usable_end_s'] else 'unused'
            eid=f"L{int(w.lane_id.iloc[0])}_F{int(w.vehicle_id.iloc[0])}_L{int(w.leader_id.iloc[0])}_{int(round((a+cfg['guard_s'])*10)):06d}"
            record=dict(episode_id=eid,split=split,follower_id=int(w.vehicle_id.iloc[0]),leader_id=int(w.leader_id.iloc[0]),
                lane_id=int(w.lane_id.iloc[0]),guard_start_s=a,guard_end_s=b,start_s=a+cfg['guard_s'],end_s=b-cfg['guard_s'],
                duration_s=cfg['duration_s'],guard_s=cfg['guard_s'])
            q=observation_quality(w,cfg); reasons=q.pop('rejection_reasons'); record.update(q)
            record['rejection_reasons']=reasons
            if not reasons:
                z,diagnostics=reconstruct(w,cfg)
                reasons=diagnostics.pop('rejection_reasons'); record.update(diagnostics); record['rejection_reasons']=reasons
                if not reasons:
                    z.insert(0,'episode_id',eid); z.insert(1,'split',split); z['follower_id']=record['follower_id']
                    arrays[eid]=z
                    record['observed_min_gap_m']=float(z.gap_m.min())
                    record['observed_speed_range_m_s']=float(np.ptp(z.observed_v_m_s))
            record['quality_pass']=not bool(reasons); rows.append(record)
    candidates=pd.DataFrame(rows).sort_values(['start_s','lane_id','follower_id']); candidates['selected']=False
    passed=candidates[candidates.quality_pass]; possible_test=passed[passed.split=='test']
    reserved=set(possible_test.follower_id)|set(possible_test.leader_id)
    train_pool=passed[(passed.split=='train')&~passed.follower_id.isin(reserved)&~passed.leader_id.isin(reserved)]
    if cfg.get('mode')=='single_episode':
        pool=passed[(passed.lane_id==cfg['selected_lane'])&(passed.guard_end_s<=cfg['usable_end_s'])]
        selected=pool.sort_values(['observed_speed_range_m_s','episode_id'],ascending=[False,True]).head(1).copy()
        selected['split']='case_study'
        for eid in selected.episode_id:
            arrays[eid]['split']='case_study'
    else:
        selected=pd.concat([sample_stratified(train_pool,cfg['maximum_train_episodes']),sample_stratified(possible_test,cfg['maximum_test_episodes'])])
    if selected.empty:
        candidates.to_csv(out/'candidate_quality.csv',index=False,float_format='%.8f')
        raise ValueError('Too few quality episodes; inspect candidate_quality.csv using observations only')
    selected=selected.drop(columns=['selected']).sort_values(['split','start_s','lane_id'])
    candidates.loc[candidates.episode_id.isin(selected.episode_id),'selected']=True
    candidates.to_csv(out/'candidate_quality.csv',index=False,float_format='%.8f')
    selected.to_csv(out/'episodes.csv',index=False,float_format='%.8f')
    columns=['episode_id','split','follower_id','leader_id','lane_id','time_s','frame_id',
        'observed_x_m','observed_v_m_s','observed_a_m_s2','leader_x_m','leader_v_m_s','leader_a_m_s2',
        'leader_length_m','gap_m','raw_observed_x_m','raw_observed_v_m_s','raw_leader_x_m','raw_leader_v_m_s',
        'raw_gap_m','local_x_m','leader_lateral_m','lateral_residual_m','leader_lateral_residual_m']
    data=pd.concat([arrays[e][columns] for e in selected.episode_id],ignore_index=True)
    data.to_csv(out/'following_observations.csv',index=False,float_format='%.8f')
    reasons=sorted({r for text in candidates.rejection_reasons for r in str(text).split(';') if r})
    counts=[dict(reason=reason,windows=int(candidates.rejection_reasons.str.split(';').apply(lambda rr:reason in rr).sum())) for reason in reasons]
    pd.DataFrame(counts).to_csv(out/'quality_rejection_counts.csv',index=False)
    selected.groupby(['split','lane_id']).size().rename('episodes').reset_index().to_csv(out/'selected_lane_counts.csv',index=False)
    train=selected[selected.split=='train']; test=selected[selected.split=='test']
    train_ids=set(train.follower_id)|set(train.leader_id); test_ids=set(test.follower_id)|set(test.leader_id)
    audit=dict(selection_version=cfg['version'],source_filename=Path(path).name,source_sha256=sha256(path),
        raw_rows=raw_rows,lane_rows={str(k):int(v) for k,v in lane_counts.items()},eligible_stable_runs=runs.ngroups,
        extended_candidate_windows=len(candidates),quality_pass_windows=len(passed),
        candidates_train_after_vehicle_exclusion=len(train_pool),candidates_test=len(possible_test),
        selected_train=len(train),selected_test=len(test),selected_episodes=len(selected),selected_points=len(data),duration_s=cfg['duration_s'],
        shared_train_test_vehicle_ids=sorted(train_ids&test_ids),
        selected_raw_kinematic_rmse_max_m_s=float(selected.raw_kinematic_rmse_m_s.max()),
        selected_reconstructed_kinematic_rmse_max_m_s=float(selected.reconstructed_kinematic_rmse_m_s.max()),
        selected_raw_accel_max_m_s2=float(selected.raw_accel_max_m_s2.max()),
        selected_position_adjustment_max_m=float(selected.position_adjustment_max_m.max()),
        selected_reconstructed_accel_max_m_s2=float(selected.reconstructed_accel_max_m_s2.max()),
        config=cfg,config_sha256=sha256(config_path),
        selection='observations only; lanes2-5; persistent pair with5s guards; lateral/kinematic screening; one episode/follower; lane/time coverage; no model errors',
        processing='local cubic position smoothing RMSbudget0.1m, analytic derivatives; NOT published reconstructed NGSIM',
        split=f"guards included: train end<={cfg['train_end_s']}s; test start>={cfg['test_start_s']}s and end<={cfg['usable_end_s']}s; all eligible test vehicle IDs excluded from train")
    (out/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return selected,data,audit


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data');parser.add_argument('--output',default=str(ROOT/'results'));parser.add_argument('--config')
    args=parser.parse_args()
    _,_,audit=prepare(locate_data(args.data),args.output,args.config)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
