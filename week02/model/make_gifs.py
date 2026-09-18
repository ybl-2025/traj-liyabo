"""Render observed and simulated trajectories with identical time and speed scales."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def render(key,velocity_key,title,filename):
    z=np.load(ROOT/'results/trajectories.npz')
    p=z[key];speed=np.linalg.norm(z[velocity_key],axis=-1)
    t=z['time_s'];frames=z['frames']
    assert np.isfinite(p).all() and np.isfinite(speed).all()
    indices=list(range(0,len(t),2))
    if indices[-1]!=len(t)-1:indices.append(len(t)-1)
    fig,ax=plt.subplots(figsize=(7,7),dpi=96)
    fig.subplots_adjust(left=.12,right=.86,bottom=.12,top=.86)
    ax.set(xlim=(-11,11),ylim=(-11,11),xlabel='x (m)',ylabel='y (m)',title=title)
    ax.set_aspect('equal');ax.grid(alpha=.15)
    ax.add_patch(plt.Circle((0,0),10,fill=False,ls='--',ec='#8295a3',lw=1))
    ax.add_patch(plt.Circle((0,0),3,fill=False,ls=':',ec='#8295a3',lw=1))
    ax.scatter(*p[0].T,s=14,facecolors='none',edgecolors='#9ba7af',lw=.7)
    trails=LineCollection([],colors='#aebac4',linewidths=.7,alpha=.6)
    ax.add_collection(trails)
    dots=ax.scatter(*p[0].T,c=speed[0],s=30,cmap='viridis',vmin=0,vmax=3.5,edgecolors='white',lw=.3,zorder=3)
    cb=fig.colorbar(dots,ax=ax,fraction=.046,pad=.025)
    cb.set_label('Speed (m/s)');cb.set_ticks([0,1,2,3,3.5])
    caption=fig.text(.12,.925,'',fontsize=10)
    fig.text(.12,.035,'64 pedestrians | 25 fps assumed | 1x playback | trails: last 1.5 s',fontsize=9,color='#475569')
    images=[];palette=None
    for k in indices:
        lo=max(0,k-38)
        trails.set_segments([p[lo:k+1,j] for j in range(p.shape[1])])
        dots.set_offsets(p[k]);dots.set_array(speed[k])
        center=int((np.linalg.norm(p[k],axis=1)<3).sum())
        caption.set_text(f't = {t[k]:5.2f} s   |   frame {frames[k]}   |   mean speed {speed[k].mean():.2f} m/s   |   center (<3m): {center}')
        fig.canvas.draw()
        im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        if palette is None:palette=im.quantize(colors=256)
        images.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
    durations=[int(round((t[b]-t[a])*1000)) for a,b in zip(indices[:-1],indices[1:])]+[1200]
    target=ROOT/'results'/filename
    images[0].save(target,save_all=True,append_images=images[1:],duration=durations,loop=0,disposal=2,optimize=False)
    plt.close(fig)
    with Image.open(target) as check:
        assert check.n_frames==len(indices)
        assert check.size==(672,672)
        assert check.info['loop']==0
        total=0
        for i in range(check.n_frames):check.seek(i);total+=check.info['duration']
        assert total==sum(durations)
    print(f'{filename}: {len(indices)} frames, {total/1000:.2f}s per loop, {target.stat().st_size/1024**2:.2f} MiB')

if __name__=='__main__':
    render('observed','observed_velocity','Observed circle antipode experiment','05_observed.gif')
    render('simulation','simulation_velocity','Social force simulation (selected parameters)','06_simulation.gif')
