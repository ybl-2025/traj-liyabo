import numpy as np
import pytest
import torch
import trajnetbaselines

NAN = float('nan')


def test_simple_grid():
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=4, blur_size=3)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-1.0, -1.0],
    ])
    occupancies = pool.occupancies(obs).numpy().tolist()
    assert occupancies == [[
        1, 0,
        0, 0,
    ], [
        0, 0,
        0, 1,
    ]]

def test_front_grid():
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=4, blur_size=0, front=True)
    obs2 = torch.Tensor([
        [0.0, 0.0],
        [-1, 1],
    ])

    obs1 = torch.Tensor([
        [-1.0, 0.0],
        [-1, 1],
    ])

    occupancies = pool.front_occupancies(obs2, obs1).numpy().tolist()
    assert occupancies == [[
        0, 0,
        0, 0,
    ], [
        0, 0,
        1, 0,
    ]]

def test_simple_grid_directional():
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=4, type_='directional')
    obs1 = torch.Tensor([
        [0.0, 0.0],
        [-1.0, -1.0],
    ])
    obs2 = torch.Tensor([
        [0.1, 0.1],
        [-1.1, -1.1],
    ])
    occupancies = pool.directional(obs1, obs2).numpy().tolist()
    assert occupancies == pytest.approx(np.array([[
        -0.1, 0, 0, 0,
        -0.1, 0, 0, 0,
    ], [
        0, 0, 0, 0.1,
        0, 0, 0, 0.1,
    ]]), abs=0.01)


def test_simple_grid_midpoint():
    """Testing a midpoint between grid cells.

    Using a large pool size as a every data point has to go into a grid
    cell first. Therefore, data can never be exactly between two cells.
    """
    pool = trajnetbaselines.lstm.Pooling(n=2, pool_size=100, blur_size=99)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-1.0, 0.0],
    ])
    occupancies = pool.occupancies(obs).numpy()
    assert occupancies == pytest.approx(np.array([[
        0.5, 0.5,
        0.0, 0,
    ], [
        0, 0.0,
        0.5, 0.5,
    ]]), abs=0.01)


def test_nan():
    pool = trajnetbaselines.lstm.Pooling(n=2)
    obs = torch.Tensor([
        [0.0, 0.0],
        [NAN, NAN],
    ])
    occupancies = pool.occupancies(obs).numpy().tolist()
    assert occupancies == [[
        0, 0,
        0, 0,
    ], [
        0, 0,
        0, 0,
    ]]


def test_embedding_shape():
    pool = trajnetbaselines.lstm.Pooling(n=2, hidden_dim=128)
    obs = torch.Tensor([
        [0.0, 0.0],
        [-0.2, -0.2],
    ])
    embedding = pool(None, None, obs)
    assert embedding.size(0) == 2
    assert embedding.size(1) == 128


def test_gridbased_mean_same_cell_is_order_invariant_and_differentiable():
    """Mean pooling combines all valid neighbours instead of overwriting one."""
    pool = trajnetbaselines.lstm.GridBasedPooling(
        n=2,
        cell_side=1.0,
        type_='occupancy',
        embedding_arch='None',
        same_cell_aggregation='mean',
    )
    obs = torch.tensor([[[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [5.0, 5.0]]])
    values = torch.zeros(1, 4, 3, 1, requires_grad=True)
    with torch.no_grad():
        # For pedestrian 0 the first two neighbours share the centre cell;
        # the third neighbour is outside the grid and must be ignored.
        values[0, 0, :, 0] = torch.tensor([2.0, 4.0, 100.0])

    grid = pool.occupancy(obs.clone(), values)
    assert grid[0, 0, 1, 1].item() == pytest.approx(3.0)
    assert torch.count_nonzero(grid[0]).item() == 1

    grid[0, 0, 1, 1].backward()
    assert values.grad[0, 0, :, 0].tolist() == pytest.approx([0.5, 0.5, 0.0])

    swapped_obs = obs[:, [0, 2, 1, 3]].clone()
    swapped_values = torch.zeros(1, 4, 3, 1)
    swapped_values[0, 0, :, 0] = torch.tensor([4.0, 2.0, 100.0])
    swapped_grid = pool.occupancy(swapped_obs, swapped_values)
    assert swapped_grid[0].numpy() == pytest.approx(grid.detach()[0].numpy())


def test_hiddenstatemlp_rel_pos():
    positions = torch.Tensor([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    rel = trajnetbaselines.lstm.pooling.HiddenStateMLPPooling.rel_obs(positions)
    assert rel.numpy().tolist() == [[
        [0.0, 0.0],
        [1.0, 1.0],
    ], [
        [-1.0, -1.0],
        [0.0, 0.0],
    ]]


def test_hiddenstatemlp():
    positions = torch.Tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    hidden = torch.zeros(3, 128)
    pool = trajnetbaselines.lstm.pooling.HiddenStateMLPPooling()
    result = pool(hidden, None, positions)

test_front_grid()
