"""Check delivered artifacts and reproduce upstream's zero-speed failure."""
from pathlib import Path
import sys, json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model/vendor/socialforce'))
import socialforce
from socialforce.potentials import PedPedPotential, PedPedPotential2D

def main():
    a=np.load(ROOT/'results/trajectories.npz')
    assert a['observed'].shape == a['simulation'].shape == (415,64,2)
    assert np.array_equal(a['ids'],np.arange(1,65))
    assert np.allclose(a['observed'][0],a['simulation'][0],atol=1e-6)
    for key in a.files:assert np.isfinite(a[key]).all(),key
    report=json.loads((ROOT/'results/report.json').read_text(encoding='utf-8'))
    error=np.linalg.norm(a['simulation']-a['observed'],axis=-1)
    assert np.isclose(error.mean(),report['metrics']['selected']['all']['ADE_m'])
    assert np.isclose(error[202:].mean(),report['metrics']['selected']['holdout']['ADE_m'])
    # Zero preferred speed and zero actual speed: upstream division 0/0.
    s=torch.zeros((2,10));s[:,8]=.5;s[1,0]=5;s[1,6]=5
    failed=socialforce.Simulator().cap_velocity(s)
    assert not torch.isfinite(failed).all()
    # Verify asymmetry=0 baseline really equals upstream default potential.
    s[:,9]=1.3;s[:,6:8]+=1
    torch.testing.assert_close(PedPedPotential().grad_r_ab(s),PedPedPotential2D(asymmetry=0.).grad_r_ab(s))
    result={'artifact_checks':'passed','initial_positions_match':'passed',
        'upstream_default_potential_equivalence':'passed','upstream_zero_speed_nan':'reproduced',
        'scope':'Numerical checks do not imply behavioral agreement or no overlaps.'}
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
