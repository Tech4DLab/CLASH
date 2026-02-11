import ot
import torch
from smplx import SMPL
import numpy as np
from scipy.spatial import KDTree
import numpy as np

def calculateBaryCenter(distributions, weights, func='free_support_barycenter'):
    reg = 1e-4
    numItermax = 60000
    numInnerItermax = 100000

    assert len(distributions) == len(weights), "Number of distributions and weights must match"
    assert len(distributions) >= 2, "Need at least two distributions"
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize just in case

    # Uniform masses and mean-centering
    measures_locations = []
    measures_weights = []
    for dist in distributions:
        centered = dist #- np.mean(dist, axis=0)
        measures_locations.append(centered)
        measures_weights.append(ot.unif(len(dist)))

    # Heuristic: initial barycenter = scaled last distribution
    gradient = 'positive'
    if gradient == 'positive':
        XB_init = measures_locations[-1] * 2
    else:
        XB_init = measures_locations[-1] * 0.25

    # Add small noise to init
    XB_init += np.random.normal(0, 0.01, size=XB_init.shape)

    # Compute barycenter
    if func == 'sinkhorn':
        barycenter = ot.bregman.free_support_sinkhorn_barycenter(
            measures_locations=measures_locations,
            measures_weights=measures_weights,
            X_init=XB_init,
            weights=weights,
            reg=reg,
            numItermax=numItermax,
            numInnerItermax=numInnerItermax,
            verbose=False
        )
    elif func == 'free_support_barycenter':
        barycenter = ot.lp.free_support_barycenter(
            measures_locations=measures_locations,
            measures_weights=measures_weights,
            X_init=XB_init,
            weights=weights,
            numItermax=numItermax,
            stopThr=1e-6,
            verbose=False
        )
    elif func == 'generalized_free_support_barycenter':
        barycenter = ot.lp.generalized_free_support_barycenter(
            measures_locations=measures_locations,
            measures_weights=measures_weights,
            X_init=XB_init,
            weights=weights,
            numItermax=numItermax,
            stopThr=1e-10,
            verbose=False
        )
    else:
        raise ValueError("Unknown function type")

    return barycenter


def normalizeData(XT, mode='all'):
    """
    Normalize a collection of shape sequences XT[s][i] (S sequences, t time steps).
    """
    S = len(XT)
    t = len(XT[0])

    XTN = [[XT[s][i] - np.mean(XT[s][i], axis=0) for i in range(t)] for s in range(S)]

    if mode == 'all':
        maxDS = max(np.abs(XTN[s][i]).max() for s in range(S) for i in range(t))
        XTN = [[XTN[s][i] / maxDS for i in range(t)] for s in range(S)]

    elif mode == 'independent':
        for s in range(S):
            for i in range(t):
                maxDS = np.abs(XTN[s][i]).max()
                if maxDS > 0:
                    XTN[s][i] = XTN[s][i] / maxDS

    elif mode == 'std':
        all_points = np.vstack([XTN[s][i] for s in range(S) for i in range(t)])
        global_std = np.std(all_points)
        if global_std > 0:
            XTN = [[XTN[s][i] / global_std for i in range(t)] for s in range(S)]

    elif mode == 'std_independent':
        for s in range(S):
            for i in range(t):
                std = np.std(XTN[s][i])
                if std > 0:
                    XTN[s][i] = XTN[s][i] / std

    else:
        raise ValueError(f"Unknown normalization mode: '{mode}'")

    return XTN


def farthest_point_sampling(points, k):
    """
    Performs farthest point sampling and returns the selected points and indices.
    points: (N, 3) array
    k: number of points to sample
    """
    N = points.shape[0]
    sampled_indices = np.zeros(k, dtype=int)
    distances = np.full(N, np.inf)

    sampled_indices[0] = np.random.randint(N)
    farthest = sampled_indices[0]
    
    for i in range(1, k):
        dist = np.linalg.norm(points - points[farthest], axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)
        sampled_indices[i] = farthest
    
    sampled_points = points[sampled_indices]
    return sampled_points, sampled_indices


def upsample_points(points_reduced, target_num_points):
    points = points_reduced.copy()
    tree = KDTree(points)

    while len(points) < target_num_points:
        new_points = []
        for p in points:
            if len(points) + len(new_points) >= target_num_points:
                break
            dists, idxs = tree.query(p, k=2)
            nearest = points[idxs[1]]
            midpoint = (p + nearest) / 2
            new_points.append(midpoint)
        if not new_points:
            break
        points = np.vstack((points, new_points))
        tree = KDTree(points)

    if len(points) > target_num_points:
        points = points[:target_num_points]

    return points

def fit_smpl_to_target(target_vertices_np, smpl_model, gender='male', num_iters=50, lr=1e-2, pre_betas=None, selected_indices=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMPL(model_path='data/smpl', gender=gender).to(device)
    betas = torch.nn.Parameter(torch.zeros((1, 10), dtype=torch.float32, device=device))
    if pre_betas is not None:
        betas = torch.nn.Parameter(torch.tensor(pre_betas, dtype=torch.float32, device=device).unsqueeze(0))
    
    optimizer = torch.optim.Adam([betas], lr=1e-2)

    target_vertices = torch.tensor(target_vertices_np, dtype=torch.float32, device=device).unsqueeze(0)

    for i in range(num_iters):
        output = model(betas=betas)
        vertices = output.vertices  # [1, 6890, 3]
        vertices_reduced = vertices[:, selected_indices, :]
        loss = torch.nn.functional.mse_loss(vertices_reduced, target_vertices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Iter {i}, Loss: {loss.item():.6f}")

    return betas.detach().cpu().numpy(), vertices.detach().cpu().numpy()