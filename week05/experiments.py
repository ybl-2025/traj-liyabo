"""Deterministic scenarios, training-only calibration and honest rollout metrics."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from models import MODELS, DEFAULTS, simulate

BOUNDS = {
    'OVM': dict(v0=(15,40), kappa=(.1,2), h=(5,50), ell=(2,30)),
    'FVDM': dict(v0=(15,40), kappa=(.1,2), h=(5,50), ell=(2,30), lam=(0,2)),
    'IDM': dict(v0=(15,40), a=(.2,3), b=(.5,5), T=(.3,3), s0=(.5,5)),
}


def scenario(name, sample_dt=.1):
    duration = 60. if name != 'emergency' else 30.
    t = np.arange(round(duration/sample_dt)+1) * sample_dt
    if name == 'constant':
        vl, xl, vf, gap = np.full_like(t,20.), 40.+20.*t, 15., 35.
    else:
        initial, final, onset, decel, gap, vf = (20.,5.,10.,2.,35.,20.) if name=='braking' else (25.,0.,5.,6.,15.,25.)
        elapsed = np.clip(t-onset, 0, (initial-final)/decel)
        vl = initial-decel*elapsed
        # Exact piecewise quadratic leader position, sampled every 0.1s.
        xl = gap+5.+initial*np.minimum(t,onset)+initial*elapsed-.5*decel*elapsed**2
        xl += final*np.maximum(0,t-onset-(initial-final)/decel)
    return dict(t=t, lx=xl, lv=vl, ll=np.full_like(t,5.), x0=0., v0=vf)


def episode_arrays(w):
    t = w.time_s.to_numpy()
    return dict(t=t-t[0], lx=w.leader_x_m.to_numpy(), lv=w.leader_v_m_s.to_numpy(),
                ll=w.leader_length_m.to_numpy(), x0=float(w.observed_x_m.iloc[0]),
                v0=float(w.observed_v_m_s.iloc[0]))


def batch_prediction(model, p, windows, dt=.05):
    """Vectorized training rollout. Same ballistic law; endpoint collision penalty.

    The scalar simulator with continuous substep collision detection is authoritative
    for final evaluation. A rejected training rollout receives penalties thereafter.
    """
    lx = np.stack([w.leader_x_m.to_numpy() for w in windows])
    lv = np.stack([w.leader_v_m_s.to_numpy() for w in windows])
    ll = np.stack([w.leader_length_m.to_numpy() for w in windows])
    obsx = np.stack([w.observed_x_m.to_numpy() for w in windows])
    obsv = np.stack([w.observed_v_m_s.to_numpy() for w in windows])
    x, v = obsx[:,0].copy(), obsv[:,0].copy()
    px, pv = np.zeros_like(obsx), np.zeros_like(obsv)
    px[:,0], pv[:,0] = x,v
    alive = np.ones(len(windows),bool)
    n = int(round(.1/dt))
    hh = .1/n
    for i in range(lx.shape[1]-1):
        for j in range(n):
            f = j/n
            s = lx[:,i]*(1-f)+lx[:,i+1]*f - (ll[:,i]*(1-f)+ll[:,i+1]*f) - x
            vel = lv[:,i]*(1-f)+lv[:,i+1]*f
            safe_s = np.maximum(s,1e-6)  # denominator only; invalid rollouts are penalized
            with np.errstate(over='ignore',invalid='ignore'):
                if model in ('OVM','FVDM'):
                    z = math.tanh(p['h']/p['ell'])
                    desired = p['v0']*(np.tanh((s-p['h'])/p['ell'])+z)/(1+z)
                    a = p['kappa']*(desired-v)
                    if model=='FVDM':
                        a += p['lam']*(vel-v)
                else:
                    desired = p['s0']+np.maximum(0,v*p['T']+v*(v-vel)/(2*math.sqrt(p['a']*p['b'])))
                    a = p['a']*(1-(v/p['v0'])**4-(desired/safe_s)**2)
            alive &= (s>0) & np.isfinite(a) & (np.abs(a)<1e8)
            a = np.where(alive,a,0.)
            moving = np.full(len(v),hh)
            mask = (a<0)&(v+a*hh<0)
            moving[mask] = -v[mask]/a[mask]
            x += v*moving+.5*a*moving**2
            v = np.maximum(0,v+a*hh)
            f1 = (j+1)/n
            rear = (lx[:,i]*(1-f1)+lx[:,i+1]*f1) - (ll[:,i]*(1-f1)+ll[:,i+1]*f1)
            alive &= rear>x
        px[:,i+1] = np.where(alive,x,np.nan)
        pv[:,i+1] = np.where(alive,v,np.nan)
    return px,pv,obsx,obsv


def calibrate(data, out, max_nfev=120):
    windows = [w for _,w in data[data.split=='train'].groupby('episode_id',sort=False)]
    parameters, records = {}, []
    # Fixed starting points, independent of test data; no random test tuning.
    for model in MODELS:
        names = list(BOUNDS[model])
        low = np.array([BOUNDS[model][n][0] for n in names])
        high = np.array([BOUNDS[model][n][1] for n in names])
        starts = [np.array([DEFAULTS[model][n] for n in names]), (low+high)/2]
        def unpack(theta):
            return dict(DEFAULTS[model], **dict(zip(names,map(float,theta))))
        def residual(theta):
            px,pv,ox,ov = batch_prediction(model,unpack(theta),windows)
            # Joint full-trajectory speed/position fit, equal episode durations.
            rx = np.nan_to_num((px[:,1:]-ox[:,1:])/10.,nan=100.,posinf=100.,neginf=-100.)
            rv = np.nan_to_num((pv[:,1:]-ov[:,1:])/2.,nan=100.,posinf=100.,neginf=-100.)
            return np.concatenate([rx.ravel(),rv.ravel()])
        best = None
        for k,start in enumerate(starts):
            fit = least_squares(residual,start,bounds=(low,high),max_nfev=max_nfev,
                                ftol=1e-6,xtol=1e-6,gtol=1e-6, x_scale='jac')
            record = dict(model=model,start=k,objective=float(np.mean(fit.fun**2)),
                          nfev=fit.nfev,status=int(fit.status),message=fit.message,
                          converged=bool(fit.success), parameters=unpack(fit.x))
            records.append(record)
            print(f"{model} start {k}: objective={record['objective']:.5f}, nfev={fit.nfev}, converged={fit.success}",flush=True)
            if best is None or record['objective']<best['objective']:
                best = record
        parameters[model] = best['parameters']
    payload = dict(calibrated=parameters,defaults=DEFAULTS,bounds=BOUNDS,starts=records,
                   loss=f'mean of concatenated (position_error/10m)^2 and (speed_error/2mps)^2; all {len(windows)} train episodes; exclude initial point',
                   dt_s=.05,model_selection='lowest training objective from two fixed starts; no test selection')
    Path(out,'parameters.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return parameters


def metrics(sim, leader_v, observed_x=None, observed_v=None, settle_after=0):
    valid = np.isfinite(sim.x)
    mask = valid.copy()
    mask[0] = False
    full = sim.status=='success'
    err = np.abs(sim.v-np.asarray(leader_v))
    suffix_ok = np.logical_and.accumulate((err<=.5)[::-1])[::-1]
    eligible = np.flatnonzero(suffix_ok & (sim.t>=settle_after))
    row = dict(status=sim.status,failure_time_s=sim.failure_time_s,
               min_gap_m=sim.min_gap_m,max_abs_accel_m_s2=sim.max_abs_accel_m_s2,
               harsh_acceleration=sim.max_abs_accel_m_s2>8.,stop_corrections=sim.stop_corrections,
               evaluated_duration_s=float(sim.t[valid][-1]-sim.t[0]),
               completed_fraction=float(mask.sum()/(len(sim.t)-1)),
               speed_response_rmse_m_s=float(np.sqrt(np.mean((sim.v[mask]-np.asarray(leader_v)[mask])**2))) if mask.any() else None,
               settling_time_s=float(sim.t[eligible[0]]-settle_after) if full and len(eligible) else None,
               max_speed_m_s=float(np.nanmax(sim.v)),
               min_accel_m_s2=float(np.nanmin(sim.a)) if np.isfinite(sim.a).any() else None,
               max_accel_m_s2=float(np.nanmax(sim.a)) if np.isfinite(sim.a).any() else None)
    if observed_x is not None:
        ex = sim.x[mask]-np.asarray(observed_x)[mask]
        ev = sim.v[mask]-np.asarray(observed_v)[mask]
        rx = float(np.sqrt(np.mean(ex**2))) if len(ex) else None
        rv = float(np.sqrt(np.mean(ev**2))) if len(ev) else None
        row.update(position_rmse_m=rx if full else None,speed_rmse_m_s=rv if full else None,
                   gap_rmse_m=rx if full else None,partial_position_rmse_m=rx,partial_speed_rmse_m_s=rv,
                   final_position_error_m=float(sim.x[-1]-np.asarray(observed_x)[-1]) if full else None)
    return row


def evaluate(data, calibrated, out):
    out = Path(out)
    rows, trajectories = [], []
    synthetic = {}
    for name in ('constant','braking','emergency'):
        a = scenario(name)
        for model in MODELS:
            sim = simulate(model,DEFAULTS[model],**dict(t=a['t'],leader_x=a['lx'],leader_v=a['lv'],leader_length=a['ll'],x0=a['x0'],v0=a['v0']))
            synthetic[name,model] = sim
            row = dict(scenario=name,model=model,parameter_set='default',
                       **metrics(sim,a['lv'],settle_after=17.5 if name=='braking' else 9.1666667 if name=='emergency' else 0.))
            rows.append(row)
            trajectories.append(pd.DataFrame(dict(scenario=name,model=model,time_s=sim.t,
                leader_x_m=a['lx'],leader_v_m_s=a['lv'],follower_x_m=sim.x,
                follower_v_m_s=sim.v,accel_m_s2=sim.a,gap_m=sim.gap)))
    pd.DataFrame(rows).to_csv(out/'synthetic_metrics.csv',index=False,float_format='%.8f')
    pd.concat(trajectories).to_csv(out/'synthetic_trajectories.csv',index=False,float_format='%.8f')
    rows, trajectories, real_sims = [], [], {}
    for eid,w in data.groupby('episode_id',sort=False):
        ar = episode_arrays(w)
        for pname,ps in [('default',DEFAULTS),('calibrated',calibrated)]:
            for model in MODELS:
                sim = simulate(model,ps[model],ar['t'],ar['lx'],ar['lv'],ar['ll'],ar['x0'],ar['v0'])
                real_sims[eid,pname,model] = sim
                row=dict(episode_id=eid,split=w.split.iloc[0],parameter_set=pname,model=model,
                         **metrics(sim,ar['lv'],w.observed_x_m,w.observed_v_m_s))
                if 'raw_observed_x_m' in w:
                    valid=np.isfinite(sim.x);valid[0]=False
                    row['raw_position_rmse_m']=float(np.sqrt(np.mean((sim.x[valid]-w.raw_observed_x_m.to_numpy()[valid])**2))) if sim.status=='success' else None
                    row['raw_speed_rmse_m_s']=float(np.sqrt(np.mean((sim.v[valid]-w.raw_observed_v_m_s.to_numpy()[valid])**2))) if sim.status=='success' else None
                rows.append(row)
                trajectories.append(pd.DataFrame(dict(episode_id=eid,split=w.split.iloc[0],parameter_set=pname,model=model,
                    time_s=sim.t,global_time_s=w.time_s.to_numpy(),follower_x_m=sim.x,
                    follower_v_m_s=sim.v,accel_m_s2=sim.a,gap_m=sim.gap)))
    detail = pd.DataFrame(rows)
    detail.to_csv(out/'ngsim_episode_metrics.csv',index=False,float_format='%.8f')
    pd.concat(trajectories).to_csv(out/'ngsim_predictions.csv',index=False,float_format='%.8f')
    summaries = []
    for (split,pname,model),g in detail.groupby(['split','parameter_set','model'],sort=False):
        ok = g[g.status=='success']
        summaries.append(dict(split=split,parameter_set=pname,model=model,episodes=len(g),
            successes=len(ok),collisions=int(g.status.eq('collision').sum()),
            numerical_failures=int(g.status.eq('numerical_failure').sum()),
            failure_rate=float((g.status!='success').mean()), harsh_acceleration_episodes=int(g.harsh_acceleration.sum()),
            position_rmse_m=float(ok.position_rmse_m.mean()),speed_rmse_m_s=float(ok.speed_rmse_m_s.mean()),
            position_rmse_p90_m=float(ok.position_rmse_m.quantile(.9)),
            raw_position_rmse_m=float(ok.raw_position_rmse_m.mean()) if 'raw_position_rmse_m' in ok else None,
            raw_speed_rmse_m_s=float(ok.raw_speed_rmse_m_s.mean()) if 'raw_speed_rmse_m_s' in ok else None,
            minimum_gap_m=float(g.min_gap_m.min()),median_min_gap_m=float(g.min_gap_m.median())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'ngsim_summary.csv',index=False,float_format='%.8f')
    # Same episodes for every model; report separately to expose survivor selection.
    complete = detail.groupby(['split','parameter_set','episode_id']).status.apply(lambda x:x.eq('success').all())
    shared = complete[complete].reset_index()[['split','parameter_set','episode_id']]
    paired = detail.merge(shared,on=['split','parameter_set','episode_id'])
    paired.groupby(['split','parameter_set','model']).agg(episodes=('episode_id','size'),
        position_rmse_m=('position_rmse_m','mean'),speed_rmse_m_s=('speed_rmse_m_s','mean')).reset_index().to_csv(
        out/'ngsim_common_success_metrics.csv',index=False,float_format='%.8f')
    return synthetic,real_sims,detail,summary


def convergence_check(data, calibrated, out):
    rows=[]
    for name in ('constant','braking','emergency'):
        ar=scenario(name)
        for model in MODELS:
            coarse=simulate(model,DEFAULTS[model],ar['t'],ar['lx'],ar['lv'],ar['ll'],ar['x0'],ar['v0'],dt=.05)
            fine=simulate(model,DEFAULTS[model],ar['t'],ar['lx'],ar['lv'],ar['ll'],ar['x0'],ar['v0'],dt=.025)
            valid=np.isfinite(coarse.x)&np.isfinite(fine.x)
            rows.append(dict(experiment=name,model=model,parameter_set='default',status_dt05=coarse.status,status_dt025=fine.status,
                max_position_difference_m=float(np.max(abs(coarse.x[valid]-fine.x[valid]))),
                max_speed_difference_m_s=float(np.max(abs(coarse.v[valid]-fine.v[valid]))),
                min_gap_difference_m=abs(coarse.min_gap_m-fine.min_gap_m)))
    for eid,w in data.groupby('episode_id',sort=False):
        ar=episode_arrays(w)
        for pname, ps in [('default',DEFAULTS),('calibrated',calibrated)]:
            for model in MODELS:
                sims=[simulate(model,ps[model],ar['t'],ar['lx'],ar['lv'],ar['ll'],ar['x0'],ar['v0'],dt=dt) for dt in (.05,.025)]
                valid=np.isfinite(sims[0].x)&np.isfinite(sims[1].x)
                rows.append(dict(experiment=eid,split=w.split.iloc[0],model=model,parameter_set=pname,status_dt05=sims[0].status,status_dt025=sims[1].status,
                    max_position_difference_m=float(np.max(abs(sims[0].x[valid]-sims[1].x[valid]))),
                    max_speed_difference_m_s=float(np.max(abs(sims[0].v[valid]-sims[1].v[valid]))),
                    min_gap_difference_m=abs(sims[0].min_gap_m-sims[1].min_gap_m)))
    check=pd.DataFrame(rows)
    check.to_csv(Path(out)/'step_size_check.csv',index=False,float_format='%.8f')
    return check
