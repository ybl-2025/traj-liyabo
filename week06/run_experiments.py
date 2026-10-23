"""Only Week 6 outputs; corrected participant counts in both scenarios."""
import csv
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from prepare_data import ROOT,inventory
from models import MODELS,PARAMETERS
from experiments import specifications,narrow_agents,circle_agents,run_case

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compute_spec(s):
    args={k:v for k,v in s.items() if k!='key'}
    r,row=run_case(**args);h,hm=run_case(**args,dt=.01)
    step=dict(case=s['key'],scenario=s['scenario'],model=s['model'],base_status=row['status'],half_status=hm['status'],
              status_changed=row['status']!=hm['status'],base_min_gap_m=row['min_clearance_m'],half_min_gap_m=hm['min_clearance_m'],
              max_position_difference_m=float(np.linalg.norm(r['position']-h['position'],axis=-1).max()),
              base_arrival_fraction=row['arrival_fraction'],half_arrival_fraction=hm['arrival_fraction'])
    return r,row,step

def compute_noise(item):
    scene,pairs,seed=item;a=narrow_agents() if scene=='narrow' else circle_agents(pairs=pairs)
    noise=np.random.default_rng(seed).normal(0,.02,a.position.shape)
    # Shift each car's goal by the same offset, preserving its radial heading.
    car=a.half_box[:,0]>0;a.goal[car]+=noise[car];a.position+=noise
    if scene=='narrow':a.position[car,1]=0;a.goal[car,1]=0
    _,row=run_case(a,'SIDE',scene,f'noise_2cm_seed_{seed}',width=3.5 if scene=='narrow' else None,radius=10. if scene=='circle' else None)
    return row

def main():
    out=ROOT/'results';out.mkdir(exist_ok=True);before=inventory()
    runs={};metrics={};agents={};records=[];steps=[]
    specs=specifications()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for s,(r,row,step) in zip(specs,pool.map(compute_spec,specs)):
            key=s['key'];runs[key]=r;metrics[key]=row;agents[key]=s['agents'];records.append(row);steps.append(step)
            np.savez_compressed(out/f'{key}.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)},
                                initial_position=s['agents'].position,initial_velocity=s['agents'].velocity,
                                goals=s['agents'].goal,radius=s['agents'].radius,half_box=s['agents'].half_box)
            print(key,f"{row['pedestrians']} pedestrians + {row['cars']} cars",row['status'],round(row['min_clearance_m'],3),flush=True)
        noise_cases=[('narrow',1,s) for s in [1,2,3]]+[('circle',n,s) for n in range(1,7) for s in [1,2,3]]
        robustness=list(pool.map(compute_noise,noise_cases))
    write_csv(out/'experiment_metrics.csv',records);write_csv(out/'step_size_check.csv',steps);write_csv(out/'robustness.csv',robustness)
    config=dict(version=3,models=list(MODELS),shared_interaction=PARAMETERS,dt_s=.02,sample_s=.1,duration_s=40,
                scenes={'narrow':{'pedestrians':4,'cars':1,'widths_m':[3.5,3.,2.8,2.4],'road_length_m':40},
                        'circle':{'pairs':[1,2,3,4,5,6],'radius_m':10.,'boundary':'none; circle is initial layout only',
                                  'layout':'2N uniform starts; N cars on right semicircle, paired pedestrians at antipodes'}},
                pedestrian=dict(speed_m_s=1.5,tau_s=.5,max_acc_m_s2=3,radius_m=.25),
                car=dict(speed_m_s=3,tau_s=.8,max_acc_m_s2=4,length_m=4.4,width_m=1.8,mobility='fixed initial diameter heading; no reverse'),
                parameter_policy='Same final parameters as first version, no refitting for revised participant counts.',
                continuation='After overlap continue only as diagnostic. Later arrival is never safe success.')
    (out/'parameters.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    initial={key:dict(pedestrians=metrics[key]['pedestrians'],cars=metrics[key]['cars'],position=a.position.tolist(),velocity=a.velocity.tolist(),goals=a.goal.tolist()) for key,a in agents.items()}
    (ROOT/'data/scene_initial_conditions.json').write_text(json.dumps(initial,ensure_ascii=False,indent=2),encoding='utf8')
    from reporting import make_figures,make_readme
    from replay import make_replay
    make_figures(out,runs,agents,metrics,steps);make_readme(out,records,steps,robustness);make_replay(out,runs,agents,metrics)
    after=inventory();changed=[k for k in before if after.get(k)!=before[k]];added=[k for k in after if k not in before]
    integrity=dict(unchanged=not changed and not added,checked_files=len(before),method='paths, sizes, nanosecond modification times',changed_or_missing=changed,added=added)
    (out/'reference_weeks_integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf8');assert integrity['unchanged']
    from validate_results import validate
    print(validate(),flush=True)

if __name__=='__main__':main()
