"""Corrected scenes: four pedestrians / one car; one pedestrian / one car."""
import numpy as np
from models import Agents, simulate, MODELS

SCENARIOS = ('narrow', 'circle')

def build_agents(ped_positions, ped_goals, car_position, car_goal):
    p=np.asarray(ped_positions,dtype=float); g=np.asarray(ped_goals,dtype=float)
    n=len(p); direction=(g-p)/np.linalg.norm(g-p,axis=1)[:,None]
    cp=np.atleast_2d(np.asarray(car_position,dtype=float));cg=np.atleast_2d(np.asarray(car_goal,dtype=float));nc=len(cp)
    cd=(cg-cp)/np.linalg.norm(cg-cp,axis=1)[:,None]
    return Agents(np.vstack((p,cp)),np.vstack((direction*1.5,cd*3.)),
                  np.vstack((g,cg)),np.r_[np.full(n,.25),np.zeros(nc)],
                  np.vstack((np.zeros((n,2)),np.tile([2.2,.9],(nc,1)))),np.r_[np.full(n,1.5),np.full(nc,3.)],
                  np.r_[np.full(n,.5),np.full(nc,.8)],np.r_[np.full(n,3.),np.full(nc,4.)],np.ones((n+nc,2)))

def narrow_agents(width=3.5):
    # A small single-file group. Distinct goals preserve their original order.
    x=np.array([4.,6.,8.,10.]); p=np.column_stack((x,np.zeros(4)))
    return build_agents(p,p+[26.,0.],[36.,0.],[4.,0.])

def circle_agents(radius=10.,pairs=1):
    if not 1<=pairs<=6:raise ValueError('pairs must be between 1 and 6')
    # All 2N starts are uniformly spaced. Cars occupy the right semicircle;
    # each paired pedestrian starts at its exact antipode on the left semicircle.
    theta=(np.arange(pairs)-(pairs-1)/2)*np.pi/pairs
    cp=radius*np.column_stack((np.cos(theta),np.sin(theta)));pp=-cp
    return build_agents(pp,-pp,cp,-cp)

def metric(result, agents, scenario, model, case, width=None, radius=None):
    x=result['position']; v=result['velocity']; t=result['time']; fine=result['fine_time']
    arrival=result['arrival_times']; done=np.isfinite(arrival)
    is_car=agents.half_box[:,0]>0; pi=np.flatnonzero(~is_car); ci=np.flatnonzero(is_car)
    if scenario=='narrow':
        car=ci[0];crossed=(x[:,pi,0]-agents.radius[pi] > x[:,car,None,0]+agents.half_box[car,0]).all(axis=1)
        passed=float(t[np.flatnonzero(crossed)[0]]) if crossed.any() else None
    else:
        # Pairwise exchange of signed progress along each original diameter;
        # every assigned pedestrian must clear its paired vehicle rear.
        direction=(agents.goal[pi]-agents.position[pi]);direction/=np.linalg.norm(direction,axis=1)[:,None]
        separation=np.sum((x[:,pi]-x[:,ci])*direction[None,:,:],axis=-1)
        crossed=(separation>agents.radius[pi]+agents.half_box[ci,0]).all(axis=1)
        passed=float(t[np.flatnonzero(crossed)[0]]) if crossed.any() else None
    status='collision' if result['collision_time'] is not None else ('boundary' if result['boundary_time'] is not None else ('success' if done.all() else 'timeout'))
    speeds=np.linalg.norm(v,axis=-1)
    active=t[:-1,None]<np.where(done,arrival,np.inf)[None,:]
    wait=((speeds[:-1]<.1)&active).sum(axis=0)*(t[1]-t[0])
    return dict(scenario=scenario,case=case,model=model,width_m=width,radius_m=radius,
                agents=len(agents.position),pedestrians=len(pi),cars=len(ci),status=status,
                collision_time_s=result['collision_time'],boundary_time_s=result['boundary_time'],
                min_clearance_m=float(result['min_gap'].min()),min_car_clearance_m=float(result['min_car_gap'].min()),
                min_pedestrian_clearance_m=float(result['min_ped_gap'].min()) if len(pi)>1 else None,
                min_vehicle_clearance_m=float(result['min_vehicle_gap'].min()) if len(ci)>1 else None,
                min_wall_clearance_m=None if width is None else float(result['wall_gap'].min()),
                overlap_duration_s=float((result['min_gap'][:-1]<-1e-6).sum()*(fine[1]-fine[0])),
                arrival_fraction=float(done.mean()),pedestrian_arrival_fraction=float(done[pi].mean()),
                car_arrival_fraction=float(done[ci].mean()),car_arrived=bool(done[ci].all()),all_arrival_time_s=float(arrival.max()) if done.all() else None,
                pass_time_s=passed,pedestrian_mean_wait_s=float(wait[pi].mean()),car_wait_s=float(wait[ci].mean()),
                pedestrian_min_speed_m_s=float(speeds[:,pi].min()),car_min_speed_before_arrival_m_s=float(speeds[:-1,ci][active[:,ci]].min()),
                mean_speed_m_s=float(speeds.mean()))

def run_case(agents,model,scenario,case,width=None,radius=None,dt=.02,duration=40.):
    result=simulate(agents,model,duration,dt=dt,width=width,sample_dt=.1)
    return result,metric(result,agents,scenario,model,case,width,radius)

def specifications():
    specs=[]
    for width in [3.5,3.,2.8,2.4]:
        for model in MODELS:
            specs.append(dict(key=f'narrow_{width:.1f}_{model}',agents=narrow_agents(width),model=model,
                              scenario='narrow',case=f'width_{width:.1f}',width=width,radius=None))
    for pairs in range(1,7):
        for model in MODELS:
            specs.append(dict(key=f'circle_n{pairs}_{model}',agents=circle_agents(10.,pairs),model=model,
                              scenario='circle',case=f'pairs_{pairs}',width=None,radius=10.))
    return specs
