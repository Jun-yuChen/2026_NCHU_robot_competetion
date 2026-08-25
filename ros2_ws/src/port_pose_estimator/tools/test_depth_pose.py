"""Exercise depth_pose_lib against depth synthesised from a CAD model.

There is no camera attached, so the depth image is ray-cast from the mesh with
the real D405 geometry (87 deg FOV, the 24.4 deg mount tilt from the hand-eye
calibration) plus sensor noise and dropped returns inside narrow openings. Every
stage after that is the production code path, so a pass here means the geometry
and the algorithm agree -- it does not prove the sensor behaves as modelled.

Run under the foundationpose env (it has trimesh):

    ~/.local/bin/micromamba run -n foundationpose python tools/test_depth_pose.py
"""
import json
import os
import sys
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from py_gripper import depth_pose_lib as dpl

RJ45_MESH = os.path.expanduser('~/FoundationPose/objects/rj45_test/rj45_test.obj')

# Constants come from build_reference.py rather than from a drawing, so that the
# "centre of the opening" here means the same thing the detector will report.
REF = json.load(open(os.path.join(os.path.dirname(__file__), '..',
                                  'config', 'opening_reference.json')))['rj45_test']
OPENING_CENTROID = np.array(REF['opening_centroid_in_mesh'])
RJ45_PROBE = REF['flip_probe']


def render_depth(mesh, T_cam_obj, K, W, H, noise=0.0004, drop_deep=0.012, seed=0):
    """Ray-cast a depth image of mesh posed at T_cam_obj, camera at the origin.

    Only a window around the principal point is traced. Resolution has to match
    the real sensor (a D405 gives ~0.35mm per pixel at this range): render too
    coarsely and the rectified grid ends up finer than the data feeding it, which
    punches speckle holes through the surface and wrecks the opening's shape.
    Cropping keeps that resolution affordable.
    """
    rng = np.random.default_rng(seed)
    scene = mesh.copy()
    scene.apply_transform(T_cam_obj)

    half = 160
    us = np.arange(int(K[0, 2]) - half, int(K[0, 2]) + half)
    vs = np.arange(int(K[1, 2]) - half, int(K[1, 2]) + half)
    u, v = np.meshgrid(us, vs)
    pix = np.stack([(u.ravel() - K[0, 2]) / K[0, 0],
                    (v.ravel() - K[1, 2]) / K[1, 1],
                    np.ones(u.size)], axis=1)
    loc, idx, _ = scene.ray.intersects_location(np.zeros_like(pix), pix,
                                                multiple_hits=False)

    depth = np.full(u.size, np.nan)
    depth[idx] = loc[:, 2]
    depth = depth.reshape(u.shape)
    depth += rng.normal(0, noise, depth.shape)

    # a D405 rarely gets a return from deep inside a narrow socket; model that,
    # because the pipeline deliberately treats those dropouts as evidence
    surface = np.nanpercentile(depth, 20)
    depth[depth > surface + drop_deep] = np.nan

    # K for the cropped window: the principal point moves with the crop
    Kc = K.copy()
    Kc[0, 2] -= us[0]
    Kc[1, 2] -= vs[0]
    return depth, Kc


def make_pose(yaw_deg, tilt_deg, dist, offset_xy=(0.0, 0.0)):
    """Object pose in the camera frame: opening towards the camera, then yawed."""
    T = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    T = trimesh.transformations.rotation_matrix(np.radians(tilt_deg), [1, 0, 0]) @ T
    T = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1]) @ T
    T[:3, 3] = [offset_xy[0], offset_xy[1], dist]
    return T


def run_single(yaw_deg, seed, verbose=False):
    mesh = trimesh.load(RJ45_MESH, force='mesh')
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    W, H = 1280, 720                     # D405 native; ~0.35mm/px at 235mm
    f = (W / 2) / np.tan(np.radians(87 / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    T_true = make_pose(yaw_deg, tilt_deg=24.4, dist=0.235)
    depth, K = render_depth(mesh, T_true, K, W, H, seed=seed)

    pts = dpl.deproject(depth, K)
    plane = dpl.fit_plane_ransac(pts, seed=seed)
    if plane is None:
        return None, 'plane fit failed'
    n, c, n_inl = plane
    rect = dpl.rectify(pts, n, c)
    ops = dpl.find_openings(rect, min_area=5e-5, max_area=5e-4)
    T_est, info = dpl.solve_single_opening(rect, ops, OPENING_CENTROID, RJ45_PROBE)
    if T_est is None:
        return None, f'{info} (openings found: {len(ops)}, plane inliers {n_inl})'

    pos_err = np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000
    dR = T_est[:3, :3].T @ T_true[:3, :3]
    ang_err = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    if verbose:
        print(f'      plane inliers {n_inl}, openings {len(ops)}, '
              f'size {np.round(info["size_mm"], 1)}mm, margin {info["margin_mm"]:.1f}mm')
    return (pos_err, ang_err), info


def main():
    print('=' * 68)
    print('RJ45 jig (single opening): centre + long axis + depth-asymmetry flip')
    print('=' * 68)
    ok = 0
    results = []
    for yaw in (0, 25, 55, 90, 130, 175, 215, 300):
        for seed in (1, 2):
            res, info = run_single(yaw, seed)
            if res is None:
                print(f'  yaw {yaw:3d} seed {seed}:  FAILED -- {info}')
                continue
            pos, ang = res
            flag = 'ok ' if (pos < 3.0 and ang < 5.0) else 'BAD'
            print(f'  yaw {yaw:3d} seed {seed}:  {flag} position {pos:5.2f} mm, '
                  f'orientation {ang:5.2f} deg')
            results.append((pos, ang))
            ok += flag == 'ok '
    if results:
        arr = np.array(results)
        print(f'\n  {ok}/{len(results)} within 3mm & 5deg  |  '
              f'median {np.median(arr[:,0]):.2f} mm, {np.median(arr[:,1]):.2f} deg')


if __name__ == '__main__':
    main()
