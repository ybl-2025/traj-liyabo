import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import narrow_agents,circle_agents,build_agents
from models import MODELS,geometry,acceleration,simulate,headings

class ModelChecks(unittest.TestCase):
    def test_all_six_requested_counts_and_no_initial_overlap(self):
        for n in range(1,7):
            a=circle_agents(pairs=n);self.assertEqual(len(a.position),2*n)
            self.assertEqual(np.count_nonzero(a.half_box[:,0]>0),n)
            self.assertGreater(geometry(a.position,a)[0].min(),0)
            np.testing.assert_allclose(a.position[:n],-a.position[n:])

    def test_narrow_counts_unchanged(self):
        a=narrow_agents();self.assertEqual(len(a.position),5);self.assertEqual(np.count_nonzero(a.half_box[:,0]>0),1)

    def test_oriented_circle_rectangle_geometry(self):
        a=build_agents([[0,3.2]],[[0,8]],[[0,0]],[[0,10]])
        g,_=geometry(a.position,a);self.assertAlmostEqual(g[0,1],.75)

    def test_parallel_rectangles_sat_clearance(self):
        a=build_agents([[30,30]],[[35,30]],[[0,0],[5,0]],[[-10,0],[-5,0]])
        g,_=geometry(a.position,a);self.assertAlmostEqual(g[1,2],.6)

    def test_crossing_rectangles_sat_overlap(self):
        a=build_agents([[30,30]],[[35,30]],[[0,0],[0,0]],[[10,0],[0,10]])
        g,_=geometry(a.position,a);self.assertAlmostEqual(g[1,2],-3.1)

    def test_geometry_normals_and_agent_order(self):
        a=circle_agents(pairs=3);a.position[0]+=[.13,.07]
        order=np.array([5,2,0,3,1,4]);b=circle_agents(pairs=3)
        for field in vars(a):setattr(b,field,getattr(a,field)[order].copy())
        ga,na=geometry(a.position,a);gb,nb=geometry(b.position,b)
        np.testing.assert_allclose(ga[np.ix_(order,order)],gb,atol=1e-8)
        np.testing.assert_allclose(na,-na.transpose(1,0,2),atol=1e-8)
        for m in MODELS:np.testing.assert_allclose(acceleration(a.position,a.velocity,a,m)[order],acceleration(b.position,b.velocity,b,m),atol=1e-8)

    def test_rotation_equivariance(self):
        a=circle_agents(pairs=4);a.position[0]+=[.1,.07];b=circle_agents(pairs=4);theta=.37
        rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for field in vars(a):setattr(b,field,getattr(a,field).copy())
        for field in ['position','velocity','goal']:setattr(b,field,getattr(a,field)@rot.T)
        np.testing.assert_allclose(geometry(a.position,a)[0],geometry(b.position,b)[0],atol=1e-8)
        for m in MODELS:np.testing.assert_allclose(acceleration(a.position,a.velocity,a,m)@rot.T,acceleration(b.position,b.velocity,b,m),atol=1e-8)

    def test_preference_breaks_one_pair_symmetry(self):
        a=circle_agents();a.position[:]=[[-2,0],[3,0]]
        np.testing.assert_allclose(acceleration(a.position,a.velocity,a,'PRED')[:,1],0,atol=1e-12)
        self.assertLess(acceleration(a.position,a.velocity,a,'SIDE')[0,1],0)

    def test_every_car_keeps_its_own_heading(self):
        a=circle_agents(pairs=6);r=simulate(a,'SIDE',4);h=headings(a)[6:];left=np.column_stack((-h[:,1],h[:,0]))
        np.testing.assert_allclose(np.sum((r['position'][:,6:]-a.position[6:])*left,axis=-1),0,atol=1e-9)
        self.assertTrue((np.sum(r['velocity'][:,6:]*h,axis=-1)>=-1e-9).all())

    def test_car_car_overlap_is_recorded_not_repaired(self):
        a=circle_agents(pairs=2);r=simulate(a,'GOAL',6)
        self.assertLess(r['min_vehicle_gap'].min(),0);self.assertIsNotNone(r['collision_time'])

if __name__=='__main__':unittest.main()
