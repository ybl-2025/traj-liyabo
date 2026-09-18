"""Reproduce circle antipode data using the pinned upstream simulator."""
from pathlib import Path
import sys, json, hashlib, itertools, argparse
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model/vendor/socialforce'))
import socialforce
from socialforce.potentials import PedPedPotential, PedPedPotential2D, PedPedPotentialDiamond
torch.set_num_threads(1)
plt.rcParams['font.family'] = 'DejaVu Sans'

def load(fps):
    raw = np.loadtxt(ROOT / 'circle-10m-64-1.txt')
    ids, frames = np.unique(raw[:,0]).astype(int), np.unique(raw[:,1]).astype(int)
    assert len(raw) == len(ids)*len(frames)
    p = np.stack([raw[raw[:,0]==i][np.argsort(raw[raw[:,0]==i,1]),2:4] for i in ids],1)/100
    assert np.all(np.diff(frames)==1)
    # Symmetric smoothing used ONLY for retrospective observed behavior plots.
    smooth = savgol_filter(p, 11, 2, axis=0)
    v = np.gradient(smooth, 1/fps, axis=0)
    return raw, ids, frames, p, v

def simulate(p, fps, pars, noise=0., seed=0, sub=4):
    # Start at frame 47; initial velocity uses preceding 11 frames only.
    k=10
    velocity = np.polyfit(np.arange(11)/fps, p[:11].reshape(11,-1),1)[0].reshape(-1,2)
    goal = -p[0]  # geometric antipodes, never measured endpoints
    pref = np.full(len(goal),pars['speed'])
    state = np.concatenate([p[k]+np.random.default_rng(seed).normal(0,noise,p[k].shape),velocity,
        np.zeros_like(goal),goal,np.full((len(goal),1),pars['tau']),pref[:,None]],1)
    cls=PedPedPotentialDiamond if pars.get('kind')=='diamond' else PedPedPotential2D
    potential = cls(v0=pars['A'],sigma=pars['B'],asymmetry=pars['asym'])
    sim = socialforce.Simulator(ped_ped=potential,delta_t=1/fps,oversampling=sub)
    state=torch.tensor(state,dtype=torch.float32)
    states=[state.numpy().copy()]
    for _ in range(len(p)-k-1):
        # Explicit arrival rule: slow within 0.5m; stop inside 0.15m.
        dist=torch.linalg.norm(state[:,0:2]-state[:,6:8],dim=1)
        state[:,9]=torch.tensor(pref,dtype=state.dtype)*torch.clamp(dist/.5,max=1)
        arrived=dist<.15
        state[arrived,2:6]=0; state[arrived,9]=1e-6
        arrival_positions=state[arrived,0:2].clone()
        state=sim(state).detach()
        state[arrived,0:2]=arrival_positions
        state[arrived,2:6]=0
        assert torch.isfinite(state).all(), 'nonfinite simulation'
        states.append(state.numpy().copy())
    states=np.stack(states)
    return states[:,:,:2],states[:,:,2:4]

def features(p,v,fps):
    delta=p[:,:,None,:]-p[:,None,:,:]
    d=np.linalg.norm(delta,axis=-1)
    idx=np.arange(p.shape[1]); d[:,idx,idx]=np.inf
    rel=v[:,:,None,:]-v[:,None,:,:]
    vv=np.sum(rel**2,axis=-1)
    ttc=np.clip(-np.sum(delta*rel,axis=-1)/np.maximum(vv,1e-8),0,2)
    miss=np.linalg.norm(delta+ttc[:,:,:,None]*rel,axis=-1)
    risk=((ttc>.05)&(ttc<2)&(miss<.6)&(d<3)).any(2)
    speed=np.linalg.norm(v,axis=-1)
    acc=np.gradient(speed,1/fps,axis=0)
    heading=np.unwrap(np.arctan2(v[:,:,1],v[:,:,0]),axis=0)
    turn=np.gradient(heading,1/fps,axis=0)
    return dict(d=d,nearest=d.min(2),risk=risk,speed=speed,acc=acc,turn=turn)

def metrics(p,v,truth,tv,fps,lo=0,hi=None):
    hi=len(p) if hi is None else hi
    f=features(p,v,fps); tf=features(truth,tv,fps)
    e=np.linalg.norm(p-truth,axis=-1)
    path=np.linalg.norm(np.diff(p,axis=0),axis=-1).sum(0)
    goal=-load(fps)[3][0]
    minradius=np.linalg.norm(p,axis=-1).min(0)
    return {'ADE_m':float(e[lo:hi].mean()),'FDE_m':float(e[hi-1].mean()),
        'speed_MAE_mps':float(np.abs(f['speed'][lo:hi]-tf['speed'][lo:hi]).mean()),
        'min_pair_distance_m':float(f['d'][lo:hi].min()),
        'close_pair_frames_lt_0.4m':int(np.sum(f['d'][lo:hi]<.4)//2),
        'mean_speed_mps':float(f['speed'][lo:hi].mean()),
        'mean_min_radius_m':float(minradius.mean()),
        'arrival_fraction':float((np.linalg.norm(p[-1]-goal,axis=1)<.75).mean()),
        'max_speed_mps':float(f['speed'].max()),'finite':bool(np.isfinite(p).all())}

def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    fps=25;raw,ids,frames,p,v=load(fps); k=10; split=202
    truth,tv=p[k:],v[k:]; t=(frames[k:]-frames[k])/fps
    # Limited prefix calibration; second half never enters parameter selection.
    trials=[]; best=None
    candidates=[dict(A=A,B=B,asym=asym,tau=.5,speed=1.3,kind='elliptical') for A,B,asym in itertools.product([1.,2.1],[.3,.5],[0.,1.])]
    candidates += [dict(A=2.1,B=B,asym=asym,tau=.5,speed=speed,kind='diamond') for B,asym,speed in itertools.product([.5,.8],[0.,.2],[2.2,2.6])]
    candidates += [dict(A=2.1,B=.5,asym=0.,tau=.5,speed=speed,kind='elliptical') for speed in [2.2,2.6]]
    for pars in candidates:
        sp,sv=simulate(p,fps,pars)
        m=metrics(sp,sv,truth,tv,fps,hi=split)
        trials.append(dict(parameters=pars,calibration=m))
        if best is None or m['ADE_m']<best[0]: best=(m['ADE_m'],pars,sp,sv)
        print('calibration',pars,round(m['ADE_m'],3),flush=True)
    _,pars,sp,sv=best
    # Upstream default baseline uses the actual 1D default potential equivalence.
    default=dict(A=2.1,B=.3,asym=0.,tau=.5,speed=1.3)
    bp,bv=simulate(p,fps,default)
    no=dict(pars,A=0.)
    np_,nv=simulate(p,fps,no)
    report={'data':{'rows':len(raw),'pedestrians':len(ids),'frames':len(frames),
        'fps_assumed':fps,'position_scale_assumed':.01,'initial_frame':int(frames[k]),
        'calibration_end_frame':int(frames[k+split-1]),'sha256':hashlib.sha256((ROOT/'circle-10m-64-1.txt').read_bytes()).hexdigest()},
        'parameters':pars,'grid':trials,'metrics':{}}
    for name,x,y in [('observed',truth,tv),('default',bp,bv),('selected',sp,sv),('no_interaction',np_,nv)]:
        report['metrics'][name]={'all':metrics(x,y,truth,tv,fps),'holdout':metrics(x,y,truth,tv,fps,lo=split)}
    sensitivity=[]
    for seed in [1,2,3]:
        x,y=simulate(p,fps,pars,noise=.02,seed=seed)
        sensitivity.append(dict(case=f'initial_noise_2cm_seed_{seed}',**metrics(x,y,truth,tv,fps,lo=split)))
    x,y=simulate(p,fps,pars,sub=8)
    sensitivity.append(dict(case='integration_8_substeps',**metrics(x,y,truth,tv,fps,lo=split)))
    for fr in [20,30]:
        _,_,_,pp,vv=load(fr); x,y=simulate(pp,fr,pars)
        sensitivity.append(dict(case=f'assumed_fps_{fr}',**metrics(x,y,pp[k:],vv[k:],fr,lo=split)))
    report['sensitivity']=sensitivity
    f=features(truth,tv,fps); sf=features(sp,sv,fps)
    # Within-person paired comparison, independent of total number of correlated frames.
    effects=[]
    for j in range(len(ids)):
        eligible=(t>1)&(t<12)&(f['speed'][:,j]>.3)
        risk=eligible&f['risk'][:,j]; safe=eligible&~f['risk'][:,j]
        if risk.sum()>=5 and safe.sum()>=5:
            effects.append([int(ids[j]),float(f['acc'][risk,j].mean()-f['acc'][safe,j].mean()),
                float(np.abs(f['turn'][risk,j]).mean()-np.abs(f['turn'][safe,j]).mean())])
    effects=np.array(effects); np.savetxt(out/'behavior_per_person.csv',effects,delimiter=',',header='id,risk_minus_safe_acc_mps2,risk_minus_safe_abs_turn_radps',comments='')
    report['behavior']={'eligible_persons':len(effects),'mean_acc_difference':float(effects[:,1].mean()),
        'fraction_negative_acc_difference':float((effects[:,1]<0).mean()),
        'mean_abs_turn_difference':float(effects[:,2].mean()),'fraction_positive_turn_difference':float((effects[:,2]>0).mean())}
    np.savez_compressed(out/'trajectories.npz',frames=frames[k:],ids=ids,time_s=t,observed=truth,observed_velocity=tv,simulation=sp,simulation_velocity=sv,default=bp,no_interaction=np_)
    rows=np.column_stack([np.repeat(frames[k:],len(ids)),np.tile(ids,len(t)),truth.reshape(-1,2),sp.reshape(-1,2),f['speed'].ravel(),sf['speed'].ravel(),f['risk'].ravel(),(f['acc']<-.3).ravel(),((np.abs(f['turn'])>.35)&(f['speed']>.3)).ravel()])
    np.savetxt(out/'frame_comparison.csv',rows,delimiter=',',header='frame,id,real_x,real_y,sim_x,sim_y,real_speed,sim_speed,predicted_conflict,braking,turning',comments='',fmt='%.6f')
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,x,title in zip(axs,[truth,sp,np_],['Observed','Social force (selected)','No interaction']):
        for z in range(len(ids)):ax.plot(x[:,z,0],x[:,z,1],lw=.7,alpha=.65)
        ax.scatter(x[0,:,0],x[0,:,1],s=7,c='k');ax.set(title=title,xlabel='x (m)',ylabel='y (m)',xlim=(-11,11),ylim=(-11,11));ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'01_trajectories.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True)
    for label,x,y in [('Observed',truth,tv),('Default',bp,bv),('Selected',sp,sv),('No interaction',np_,nv)]:
        ff=features(x,y,fps);axs[0].plot(t,ff['speed'].mean(1),label=label);axs[1].plot(t,(np.linalg.norm(x,axis=-1)<3).sum(1),label=label)
    axs[0].set_ylabel('Mean speed (m/s)');axs[1].set_ylabel('Persons within 3 m')
    axs[2].plot(t,np.linalg.norm(sp-truth,axis=-1).mean(1),label='Selected ADE');axs[2].plot(t,np.linalg.norm(bp-truth,axis=-1).mean(1),label='Default ADE');axs[2].set(ylabel='Position error (m)',xlabel='Time from frame 47 (s)')
    for ax in axs:ax.axvline(t[split],ls='--',c='k',label='Holdout begins');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'02_dynamics.png',dpi=160);plt.close(fig)
    worst=int(np.argmax(np.linalg.norm(sp-truth,axis=-1).mean(0)))
    report['worst_person']={'id':int(ids[worst]),'ADE_m':float(np.linalg.norm(sp[:,worst]-truth[:,worst],axis=-1).mean())}
    fig,ax=plt.subplots(figsize=(7,6));ax.plot(*truth[:,worst].T,label='Observed');ax.plot(*sp[:,worst].T,label='Simulation');ax.scatter(*truth[0,worst],c='k',label='Initial');ax.set_aspect('equal');ax.legend();ax.grid();ax.set(title=f'Failure example: person {ids[worst]}',xlabel='x (m)',ylabel='y (m)');fig.tight_layout();fig.savefig(out/'04_failure.png',dpi=160);plt.close(fig)
    refresh_event(report)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Self-contained playback: neither network nor a server is needed.
    data=json.dumps({'real':truth[::3].round(3).tolist(),'sim':sp[::3].round(3).tolist(),'t':t[::3].round(2).tolist(),'ids':ids.tolist()})
    html='''<!doctype html><meta charset="utf-8"><title>Circle antipode reproduction</title><style>body{font:16px sans-serif;background:#eee;text-align:center}canvas{background:white;border:1px solid #bbb}input{width:70%}</style><h2>真实数据与社会力仿真同步回放</h2><p>左：真实；右：仿真。坐标单位 m，假设 25 fps；从第 47 帧出发。</p><canvas id="c" width="1100" height="550"></canvas><p><button id="b">播放 / 暂停</button> <span id="label"></span></p><input id="s" type="range" min="0"><script>const D=DATA;const c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('s');s.max=D.t.length-1;let play=false;document.getElementById('b').onclick=()=>play=!play;function draw(){let k=+s.value;ctx.clearRect(0,0,1100,550);for(let panel=0;panel<2;panel++){let a=panel?D.sim:D.real,ox=275+550*panel;ctx.strokeStyle='#ddd';ctx.beginPath();ctx.arc(ox,275,230,0,7);ctx.stroke();for(let j=0;j<64;j++){ctx.strokeStyle=`hsl(${j*137.5%360},65%,45%)`;ctx.beginPath();for(let q=0;q<=k;q++){let x=ox+a[q][j][0]*23,y=275-a[q][j][1]*23;q?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(ox+a[k][j][0]*23,275-a[k][j][1]*23,4,0,7);ctx.fill()}}document.getElementById('label').textContent='t = '+D.t[k]+' s'}s.oninput=draw;setInterval(()=>{if(play){s.value=(+s.value+1)%D.t.length;draw()}},120);draw()</script>'''.replace('DATA',data)
    (out/'replay.html').write_text(html,encoding='utf-8')
    print(json.dumps({key:val for key,val in report.items() if key!='grid'},indent=2),flush=True)

def refresh_event(report):
    """Pair the braking person with a predicted conflicting neighbor, not merely nearest."""
    out=ROOT/'results';z=np.load(out/'trajectories.npz')
    p,v,sp,sv,t,ids,frames=[z[key] for key in ['observed','observed_velocity','simulation','simulation_velocity','time_s','ids','frames']]
    fps=report['data']['fps_assumed'];f=features(p,v,fps);sf=features(sp,sv,fps)
    score=np.where(f['risk']&(f['speed']>.4)&(f['acc']<-.3)&(f['acc']>-3)&(np.abs(f['turn'])>.35)&(np.abs(f['turn'])<2)&(t[:,None]>2)&(t[:,None]<12),-f['acc']+np.abs(f['turn']),-np.inf)
    it,j=np.unravel_index(np.argmax(score),score.shape)
    relp=p[it,j]-p[it];relv=v[it,j]-v[it]
    tt=-np.sum(relp*relv,axis=1)/np.maximum(np.sum(relv**2,axis=1),1e-8)
    miss=np.linalg.norm(relp+np.clip(tt,0,2)[:,None]*relv,axis=1)
    eligible=(tt>.05)&(tt<2)&(miss<.6)&(f['d'][it,j]<3)
    other=int(np.argmin(np.where(eligible,f['d'][it,j],np.inf)))
    assert eligible[other]
    report['event']={'frame':int(frames[it]),'time_s':float(t[it]),'ids':[int(ids[j]),int(ids[other])],
        'separation_m':float(f['d'][it,j,other]),'predicted_ttc_s':float(tt[other]),'predicted_miss_m':float(miss[other]),
        'observed_acc_mps2':float(f['acc'][it,j]),'observed_turn_radps':float(f['turn'][it,j])}
    fig,axs=plt.subplots(2,2,figsize=(11,9));sel=(t>t[it]-2)&(t<t[it]+2)
    for who in [j,other]:
        axs[0,0].plot(*p[sel,who].T,label=f'Real {ids[who]}')
        axs[0,0].plot(*sp[sel,who].T,'--',label=f'Sim {ids[who]}')
        axs[0,0].scatter(*p[it,who],s=40)
    axs[0,0].set(xlabel='x (m)',ylabel='y (m)',title=f'Predicted conflict: frame {frames[it]}');axs[0,0].set_aspect('equal')
    axs[0,1].plot(t,f['speed'][:,j],label='Real');axs[0,1].plot(t,sf['speed'][:,j],label='Sim');axs[0,1].set(ylabel='Speed (m/s)',title=f'Person {ids[j]}')
    axs[1,0].plot(t,f['d'][:,j,other],label='Real pair');axs[1,0].plot(t,sf['d'][:,j,other],label='Sim pair');axs[1,0].axhline(.4,c='r',ls=':');axs[1,0].set(ylabel='Pair distance (m)',xlabel='Time (s)')
    # Event window and moving-only turn rate avoid near-stop heading artifacts.
    moving=f['speed'][:,j]>.3
    axs[1,1].plot(t,f['acc'][:,j],label='Real acceleration (m/s2)');axs[1,1].plot(t,np.where(moving,np.abs(f['turn'][:,j]),np.nan),label='Real |turn| (rad/s)');axs[1,1].set(xlabel='Time (s)',ylabel='m/s2 or rad/s',xlim=(t[it]-2,t[it]+2))
    for ax in axs.flat:ax.legend();ax.grid(alpha=.2)
    for ax in [axs[0,1],axs[1,0],axs[1,1]]:ax.axvline(t[it],c='k',ls='--')
    fig.tight_layout();fig.savefig(out/'03_conflict_event.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
