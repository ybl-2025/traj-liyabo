"""Teaching models: identical equations for every agent, heterogeneous parameters."""
from dataclasses import dataclass
import numpy as np

MODELS = ('GOAL', 'DIST', 'PRED', 'SIDE')
LABELS = {'GOAL': '目标驱动（无避让）', 'DIST': '距离排斥',
          'PRED': '预测排斥', 'SIDE': '预测排斥＋绕行偏好'}
PARAMETERS = dict(A=5.0, B=0.45, horizon=2.0, side_strength=3.5,
                  wall_A=12.0, wall_B=0.15, arrival_radius=0.25)

@dataclass
class Agents:
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    radius: np.ndarray
    half_box: np.ndarray  # zeros for discs; fixed-heading oriented car footprint
    speed: np.ndarray
    tau: np.ndarray
    max_acc: np.ndarray
    mobility: np.ndarray  # discs componentwise; boxes projected onto fixed heading

def headings(agents):
    direction=agents.goal-agents.position
    return direction/np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-10)

def relative_geometry(delta, agents, fallback=None):
    """Disc SDF, oriented rectangle SDF, and rectangle SAT separation margin."""
    count=len(agents.position)
    distance = np.linalg.norm(delta, axis=-1)
    normal = delta / np.maximum(distance[..., None], 1e-10)
    gap = distance - agents.radius[:, None] - agents.radius[None, :]
    cars=np.flatnonzero(agents.half_box[:,0]>0);peds=np.flatnonzero(agents.half_box[:,0]==0)
    heading=headings(agents);left=np.column_stack((-heading[:,1],heading[:,0]))
    for j in cars:
        rotation=np.column_stack((heading[j],left[j]))
        d=delta[peds,j]@rotation
        q = np.abs(d) - agents.half_box[j]
        outside = np.maximum(q, 0)
        norm = np.linalg.norm(outside, axis=1)
        signed = norm + np.minimum(np.max(q, axis=1), 0)
        n = np.sign(d) * outside / np.maximum(norm[:, None], 1e-10)
        inside = norm < 1e-10
        for i in np.flatnonzero(inside):
            axis = int(np.argmax(q[i]))
            n[i] = 0
            n[i, axis] = 1 if d[i, axis] >= 0 else -1
        n=n@rotation.T
        if fallback is not None:n[inside]=fallback[peds[inside],j]
        gap[peds,j]=signed-agents.radius[peds];gap[j,peds]=gap[peds,j]
        normal[peds,j]=n;normal[j,peds]=-n
    if len(cars)>1:
        # Separating Axis Theorem: four edge normals, vectorized over every pair.
        h=heading[cars];l=left[cars];b=agents.half_box[cars];k=len(cars)
        hi=np.broadcast_to(h[:,None,:],(k,k,2));hj=np.broadcast_to(h[None,:,:],(k,k,2))
        li=np.broadcast_to(l[:,None,:],(k,k,2));lj=np.broadcast_to(l[None,:,:],(k,k,2))
        axes=np.stack((hi,li,hj,lj),axis=2);d=delta[np.ix_(cars,cars)]
        proj=np.sum(d[:,:,None,:]*axes,axis=-1)
        support_i=b[:,None,0,None]*np.abs(np.sum(hi[:,:,None,:]*axes,axis=-1))+b[:,None,1,None]*np.abs(np.sum(li[:,:,None,:]*axes,axis=-1))
        support_j=b[None,:,0,None]*np.abs(np.sum(hj[:,:,None,:]*axes,axis=-1))+b[None,:,1,None]*np.abs(np.sum(lj[:,:,None,:]*axes,axis=-1))
        margins=np.abs(proj)-support_i-support_j
        separation=margins.max(axis=-1)
        # Average tied active normals: choosing the first edge would make the
        # response depend on agent numbering or tiny rotational roundoff.
        active=np.abs(margins-separation[:,:,None])<1e-9
        signed=np.where(np.abs(proj)<1e-10,0,np.sign(proj))
        n=np.sum(axes*(active*signed)[:,:,:,None],axis=2)
        n/=np.maximum(np.linalg.norm(n,axis=-1)[:,:,None],1e-10)
        if fallback is not None:
            zero=np.linalg.norm(d,axis=-1)<1e-9;n[zero]=fallback[np.ix_(cars,cars)][zero]
        # Use the same selected axis for unordered pairs, including tied axes.
        for i in range(k):
            for j in range(i+1,k):
                gap[cars[i],cars[j]]=gap[cars[j],cars[i]]=separation[i,j]
                normal[cars[i],cars[j]]=n[i,j];normal[cars[j],cars[i]]=-n[i,j]
    np.fill_diagonal(gap, np.inf)
    normal[np.arange(count), np.arange(count)] = 0
    return gap, normal

def geometry(position, agents):
    return relative_geometry(position[:,None,:]-position[None,:,:],agents)

def pair_forecast(position, velocity, agents, horizon):
    delta = position[:, None] - position[None, :]
    relative_v = velocity[:, None] - velocity[None, :]
    dot = np.sum(delta * relative_v, axis=-1)
    t = np.clip(-dot / np.maximum(np.sum(relative_v**2, axis=-1), 1e-9), 0, horizon)
    future = delta + t[..., None] * relative_v
    gap,normal=relative_geometry(future,agents,fallback=geometry(position,agents)[1])
    return gap, normal, t, dot < -1e-8

def acceleration(position, velocity, agents, model, width=None):
    p = PARAMETERS
    toward = agents.goal - position
    dist = np.linalg.norm(toward, axis=1)
    direction = toward / np.maximum(dist[:, None], 1e-9)
    desired_speed = agents.speed * np.minimum(dist / 0.8, 1)
    drive = (desired_speed[:, None] * direction - velocity) / agents.tau[:, None]
    pair = np.zeros_like(position)
    gap, normal = geometry(position, agents)
    if model != 'GOAL':
        magnitude = p['A'] * np.exp(-np.maximum(gap, -0.3) / p['B'])
        pair = np.sum(magnitude[..., None] * normal, axis=1)
        if model in ('PRED', 'SIDE'):
            fg, fn, t, approaching = pair_forecast(position, velocity, agents, p['horizon'])
            risk = np.exp(-np.maximum(fg, 0) / p['B']) * approaching * (t > 0.02)
            # Prospective response added to current clearance response.
            pair += p['A'] * np.sum((risk / (1 + t))[..., None] * fn, axis=1)
            if model == 'SIDE':
                right = np.column_stack((direction[:, 1], -direction[:, 0]))
                # Bounded aggregate: no force proportional to crowd head count.
                urgency = 1 - np.prod(1 - np.clip(risk, 0, 1), axis=1)
                pair += p['side_strength'] * urgency[:, None] * right
    wall = np.zeros_like(position)
    if width is not None:
        extent = np.maximum(agents.radius, agents.half_box[:, 1])
        lower = position[:, 1] + width / 2 - extent
        upper = width / 2 - position[:, 1] - extent
        wall[:, 1] = p['wall_A'] * (np.exp(-np.maximum(lower, -.3)/p['wall_B']) - np.exp(-np.maximum(upper, -.3)/p['wall_B']))
    acc = (drive + pair + wall) * agents.mobility
    cars=agents.half_box[:,0]>0;heading=headings(agents)
    acc[cars]=np.sum((drive+pair+wall)[cars]*heading[cars],axis=1)[:,None]*heading[cars]
    size = np.linalg.norm(acc, axis=1)
    acc *= np.minimum(1, agents.max_acc / np.maximum(size, 1e-10))[:, None]
    return acc

def simulate(agents, model, duration, dt=0.02, width=None, sample_dt=0.1):
    """Synchronous semi-implicit Euler. Never repair overlaps or clip positions."""
    if model not in MODELS:
        raise ValueError(model)
    stride = int(round(sample_dt / dt))
    if not np.isclose(stride * dt, sample_dt):
        raise ValueError('sample_dt must be a multiple of dt')
    x = agents.position.copy(); v = agents.velocity.copy()
    arrived = np.zeros(len(x), dtype=bool)
    arrival_times = np.full(len(x), np.nan)
    positions = [x.copy()]; velocities = [v.copy()]; times = [0.0]
    gaps = []; boundary = []; car_gaps = []; ped_gaps = []; vehicle_gaps=[]; collision_time = None; boundary_time = None
    car_indices = np.flatnonzero(agents.half_box[:, 0] > 0)
    pedestrian_indices = np.flatnonzero(agents.half_box[:, 0] == 0)
    fixed_heading=headings(agents)
    first_arrival = np.linalg.norm(x-agents.goal, axis=1) < PARAMETERS['arrival_radius']
    arrived |= first_arrival; arrival_times[first_arrival] = 0
    for step in range(int(round(duration/dt)) + 1):
        time = step * dt
        gap, _ = geometry(x, agents)
        gaps.append(float(gap.min()))
        car_gaps.append(float(gap[np.ix_(car_indices, pedestrian_indices)].min()) if len(car_indices) and len(pedestrian_indices) else np.inf)
        ped_gaps.append(float(gap[np.ix_(pedestrian_indices, pedestrian_indices)].min()) if len(pedestrian_indices)>1 else np.inf)
        vehicle_gaps.append(float(gap[np.ix_(car_indices,car_indices)].min()) if len(car_indices)>1 else np.inf)
        extent = np.maximum(agents.radius, agents.half_box[:, 1])
        wall_gap = np.inf if width is None else float(np.min(width/2-extent-np.abs(x[:, 1])))
        boundary.append(wall_gap)
        if gap.min() < -1e-6 and collision_time is None: collision_time = time
        if wall_gap < -1e-6 and boundary_time is None: boundary_time = time
        if step == int(round(duration/dt)): break
        a = acceleration(x, v, agents, model, width)
        v += dt * a
        v *= agents.mobility
        # The SAME fixed-heading projection for every vehicle, at any circle angle.
        forward=np.maximum(np.sum(v[car_indices]*fixed_heading[car_indices],axis=1),0)
        v[car_indices]=forward[:,None]*fixed_heading[car_indices]
        limit = agents.speed * 1.25
        speed = np.linalg.norm(v, axis=1)
        v *= np.minimum(1, limit / np.maximum(speed, 1e-10))[:, None]
        v[arrived] = 0
        x += dt * v
        new = (~arrived) & (np.linalg.norm(x-agents.goal, axis=1) < PARAMETERS['arrival_radius'])
        arrival_times[new] = (step+1)*dt; arrived |= new
        v[arrived] = 0
        if (step+1) % stride == 0:
            positions.append(x.copy()); velocities.append(v.copy()); times.append((step+1)*dt)
    return dict(time=np.array(times), position=np.array(positions), velocity=np.array(velocities),
                fine_time=np.arange(len(gaps))*dt, min_gap=np.array(gaps), wall_gap=np.array(boundary),
                min_car_gap=np.array(car_gaps), min_ped_gap=np.array(ped_gaps),
                min_vehicle_gap=np.array(vehicle_gaps),
                arrival_times=arrival_times, collision_time=collision_time, boundary_time=boundary_time)
