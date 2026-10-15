"""Mechanism, units/sign, geometry and failure accounting checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import math
import numpy as np
from models import DEFAULTS,acceleration,optimal_velocity,simulate
from experiments import batch_prediction,episode_arrays


class ModelChecks(unittest.TestCase):
    def test_ov_endpoints_and_equilibrium(self):
        p=DEFAULTS['OVM']
        self.assertAlmostEqual(optimal_velocity(0,p),0)
        self.assertAlmostEqual(optimal_velocity(1e6,p),p['v0'])
        target=optimal_velocity(30,p)
        self.assertAlmostEqual(acceleration('OVM',30,target,99,p),0)

    def test_fvdm_velocity_difference_both_signs(self):
        ov=acceleration('OVM',25,15,15,DEFAULTS['OVM'])
        for difference in (-5,5):
            value=acceleration('FVDM',25,15,15+difference,DEFAULTS['FVDM'])
            self.assertAlmostEqual(value-ov,DEFAULTS['FVDM']['lam']*difference)
        p=dict(DEFAULTS['FVDM'],lam=0)
        self.assertAlmostEqual(acceleration('FVDM',25,15,10,p),ov)

    def test_idm_standstill_and_approaching(self):
        p=DEFAULTS['IDM']
        self.assertAlmostEqual(acceleration('IDM',p['s0'],0,0,p),0)
        self.assertLess(acceleration('IDM',25,15,10,p),acceleration('IDM',25,15,15,p))
        self.assertLess(acceleration('IDM',10,15,15,p),acceleration('IDM',30,15,15,p))

    def test_idm_equilibrium_and_leader_length(self):
        p=DEFAULTS['IDM']
        v=10.
        s=(p['s0']+v*p['T'])/math.sqrt(1-(v/p['v0'])**p['delta'])
        t=np.arange(101)*.1
        sim=simulate('IDM',p,t,s+7+v*t,np.full_like(t,v),7,0,v)
        self.assertEqual(sim.status,'success')
        np.testing.assert_allclose(sim.gap,s,atol=1e-10)
        np.testing.assert_allclose(sim.v,v,atol=1e-10)

    def test_negative_speed_uses_stopping_distance(self):
        p=dict(DEFAULTS['OVM'],v0=1.,kappa=10.,h=25.,ell=1.)
        # V(large gap)=1 and v=2 => a=-10, stopping after .2s, distance .2m.
        sim=simulate('OVM',p,[0.,1.],[100.,100.],[0.,0.],5.,0.,2.,dt=1.)
        self.assertEqual(sim.status,'success')
        self.assertAlmostEqual(sim.x[-1],.2)
        self.assertEqual(sim.v[-1],0.)
        self.assertEqual(sim.stop_corrections,1)

    def test_collision_not_hidden_or_scored_full(self):
        p=dict(DEFAULTS['OVM'],kappa=.1)
        sim=simulate('OVM',p,np.arange(11)*.1,np.ones(11)*6,np.zeros(11),5.,0.,20.)
        self.assertEqual(sim.status,'collision')
        self.assertLess(sim.failure_time_s,.1)
        self.assertEqual(sim.min_gap_m,0.)
        self.assertTrue(np.isnan(sim.x[1:]).all())
        from experiments import metrics
        row=metrics(sim,np.zeros(11),np.zeros(11),np.zeros(11))
        self.assertIsNone(row['position_rmse_m'])
        self.assertEqual(row['completed_fraction'],0.)

    def test_substep_interior_collision(self):
        # Gap positive at both endpoints but negative in the middle: must fail.
        # kappa=10, V≈0, v=10 => a=-100; stopping at .1s, distance .5m.
        # Rear starts .1m ahead, moves 1m/s, and passes stopped follower by t=1.
        p=dict(DEFAULTS['OVM'],kappa=10.,h=25.,ell=1.)
        sim=simulate('OVM',p,[0.,1.],[5.1,6.1],[1.,1.],5.,0.,10.,dt=1.)
        self.assertEqual(sim.status,'collision')
        self.assertLess(sim.failure_time_s,.1)

    def test_vector_and_scalar_training_paths_match(self):
        import pandas as pd
        w=pd.read_csv(Path(__file__).resolve().parents[1]/'results/following_observations.csv')
        w=w[w.episode_id==w.episode_id.iloc[0]]
        ar=episode_arrays(w)
        for model in DEFAULTS:
            px,pv,_,_=batch_prediction(model,DEFAULTS[model],[w])
            sim=simulate(model,DEFAULTS[model],ar['t'],ar['lx'],ar['lv'],ar['ll'],ar['x0'],ar['v0'])
            np.testing.assert_allclose(px[0],sim.x,atol=1e-8,equal_nan=True)
            np.testing.assert_allclose(pv[0],sim.v,atol=1e-8,equal_nan=True)


if __name__=='__main__':
    unittest.main()
