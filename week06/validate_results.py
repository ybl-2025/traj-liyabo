"""Independent oriented geometry and all six participant-count checks."""
import csv
import json
import numpy as np
from prepare_data import ROOT

def independent_gaps(x,radius,box,initial,goals):
    direction=goals-initial;direction/=np.linalg.norm(direction,axis=1)[:,None]
    left=np.column_stack((-direction[:,1],direction[:,0]));pairs=[[],[],[],[]]
    local_vertices=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
    for i in range(len(radius)):
        for j in range(i+1,len(radius)):
            ci=box[i,0]>0;cj=box[j,0]>0
            if ci and cj:
                # Independent vertex projection formulation rather than analytic supports.
                vi=x[:,i,None,:]+local_vertices[None,:,0,None]*box[i,0]*direction[i]+local_vertices[None,:,1,None]*box[i,1]*left[i]
                vj=x[:,j,None,:]+local_vertices[None,:,0,None]*box[j,0]*direction[j]+local_vertices[None,:,1,None]*box[j,1]*left[j]
                axes=np.array([direction[i],left[i],direction[j],left[j]])
                ai=np.einsum('tvc,ac->tva',vi,axes);aj=np.einsum('tvc,ac->tva',vj,axes)
                sep=np.maximum(ai.min(axis=1)-aj.max(axis=1),aj.min(axis=1)-ai.max(axis=1));g=sep.max(axis=1);kind=3
            elif ci or cj:
                car=i if ci else j;ped=j if ci else i;delta=x[:,ped]-x[:,car]
                local=np.column_stack((delta@direction[car],delta@left[car]));q=np.abs(local)-box[car]
                g=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)-radius[ped];kind=1
            else:g=np.linalg.norm(x[:,i]-x[:,j],axis=1)-radius[i]-radius[j];kind=2
            pairs[0].append(g);pairs[kind].append(g)
    return [np.min(p,axis=0) if p else np.full(len(x),np.inf) for p in pairs]

def validate():
    out=ROOT/'results';checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    with (out/'experiment_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    check('40_experiments',len(rows)==40);cir=[r for r in rows if r['scenario']=='circle']
    check('six_levels_four_models',len(cir)==24 and all(sum(int(r['pedestrians'])==n and int(r['cars'])==n for r in cir)==4 for n in range(1,7)))
    for row in rows:
        road=row['scenario']=='narrow';nped=4 if road else int(row['pedestrians']);ncar=1 if road else nped
        key=f'narrow_{float(row["width_m"]):.1f}_{row["model"]}' if road else f'circle_n{nped}_{row["model"]}'
        z=np.load(out/f'{key}.npz');x=z['position'];v=z['velocity'];t=z['time'];fine=z['fine_time'];prefix=key+'/'
        pm=z['half_box'][:,0]==0;cm=~pm
        check(prefix+'requested_counts',x.shape[1]==nped+ncar and int(pm.sum())==nped and int(cm.sum())==ncar and int(row['cars'])==ncar)
        check(prefix+'finite_40s',np.isfinite(x).all() and np.isfinite(v).all() and np.isclose(t[-1],40))
        check(prefix+'initial_state',np.allclose(x[0],z['initial_position']) and np.allclose(v[0],z['initial_velocity']))
        independent=independent_gaps(x,z['radius'],z['half_box'],z['initial_position'],z['goals']);index=np.rint(t/(fine[1]-fine[0])).astype(int)
        for sample,saved in zip(independent,['min_gap','min_car_gap','min_ped_gap','min_vehicle_gap']):
            check(prefix+saved+'_geometry',np.allclose(sample,z[saved][index],atol=1e-8))
        check(prefix+'fine_minimum',np.isclose(z['min_gap'].min(),float(row['min_clearance_m'])) and np.isclose(z['min_car_gap'].min(),float(row['min_car_clearance_m'])))
        hits=fine[z['min_gap']<-1e-6];collision=bool(len(hits));boundary=np.any(z['wall_gap']<-1e-6)
        check(prefix+'first_overlap',bool(row['collision_time_s'])==collision and (not collision or np.isclose(hits[0],float(row['collision_time_s']))))
        done=np.isfinite(z['arrival_times']);expected='collision' if collision else ('boundary' if boundary else ('success' if done.all() else 'timeout'))
        check(prefix+'status_counts_all_conflicts',row['status']==expected)
        check(prefix+'arrival_by_type',np.isclose(done.mean(),float(row['arrival_fraction'])) and np.isclose(done[pm].mean(),float(row['pedestrian_arrival_fraction'])) and np.isclose(done[cm].mean(),float(row['car_arrival_fraction'])))
        check(prefix+'all_cars_arrived_flag',(row['car_arrived']=='True')==done[cm].all())
        heading=z['goals']-z['initial_position'];heading/=np.linalg.norm(heading,axis=1)[:,None]
        transverse=np.stack((-heading[cm,1],heading[cm,0]),axis=1)
        residual=np.sum((x[:,cm]-x[0,cm])*transverse[None,:,:],axis=-1)
        forward=np.sum(v[:,cm]*heading[None,cm,:],axis=-1)
        check(prefix+'each_car_fixed_heading_no_reverse',np.allclose(residual,0,atol=1e-8) and np.all(forward>=-1e-8))
        check(prefix+'speed_limits',np.all(np.linalg.norm(v,axis=-1)<=np.where(pm,1.5,3.)*1.25+1e-8))
        if road:
            width=float(row['width_m']);ext=np.maximum(z['radius'],z['half_box'][:,1]);wg=(width/2-ext-np.abs(x[:,:,1])).min(axis=1)
            check(prefix+'wall_geometry',np.allclose(wg,z['wall_gap'][index]))
        else:
            check(prefix+'radius10_opposite_pairs',np.allclose(np.linalg.norm(x[0],axis=1),10) and np.allclose(x[0,pm],-x[0,cm]) and np.allclose(z['goals'],-x[0]))
            angles=np.sort(np.mod(np.arctan2(x[0,:,1],x[0,:,0]),2*np.pi));dif=np.diff(np.r_[angles,angles[0]+2*np.pi])
            check(prefix+'uniform_2N_start_angles',np.allclose(dif,np.pi/nped))
            check(prefix+'initial_no_overlap',independent[0][0]>0)
    with (out/'step_size_check.csv').open(encoding='utf-8-sig') as f:step=list(csv.DictReader(f))
    with (out/'robustness.csv').open(encoding='utf-8-sig') as f:noise=list(csv.DictReader(f))
    check('40_half_step_checks',len(step)==40);check('21_noise_cases',len(noise)==21)
    check('reference_weeks_unchanged',json.loads((out/'reference_weeks_integrity.json').read_text(encoding='utf8'))['unchanged'])
    for name in ['summary','circle_layouts','circle_models','circle_dynamics','circle_safety','narrow_comparison','width_comparison','numerical_sensitivity','failure_analysis']:
        check(name+'_png',(out/f'{name}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
    html=(out/'replay.html').read_text(encoding='utf8')
    check('six_replay_groups_embedded',all(f'circle_n{n}_SIDE' in html for n in range(1,7)) and '六组总览' in html)
    check('old_radius_files_not_mixed',not any(out.glob('circle_10.0_*.npz')))
    payload=dict(passed=True,checks_count=len(checks),checks=checks);(out/'validation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    return dict(passed=True,checks_count=len(checks))

if __name__=='__main__':print(validate())
