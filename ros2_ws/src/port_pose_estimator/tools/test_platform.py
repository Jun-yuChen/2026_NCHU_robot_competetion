"""Exercise the multi-port path against depth synthesised from the platform CAD.

The physical platform does not exist yet, so the depth image is ray-cast from
server1_all.STL with the real D405 geometry, sensor noise, and dropped returns
inside the port cavities. Everything after that is the production code path --
plane fit, work isolation, rectification, opening detection, and the RANSAC +
Kabsch registration that fixes the heading.

A pass means the geometry and the algorithm agree. It says nothing about how the
sensor will behave on the real part.

    ~/.local/bin/micromamba run -n foundationpose python tools/test_platform.py
"""
import json
import os
import sys
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from py_gripper import depth_pose_lib as dpl

MESH = os.path.expanduser('~/Downloads/server1_all.STL')
REF = json.load(open(os.path.join(os.path.dirname(__file__), '..',
                                  'config', 'opening_reference.json')))['server1']


def render(mesh, T, K, W, H, noise=0.0005, drop=0.010, seed=0, half=190):
    """Ray-cast a depth frame of the part standing on a table.

    The table has to be in the scene. Without it the dominant plane in the frame
    is the part's own top face, so "whatever stands above the dominant plane"
    finds nothing and the pipeline reports the part as missing -- which is what a
    first version of this test did, six times over.
    """
    rng = np.random.default_rng(seed)
    scene = mesh.copy()
    scene.apply_transform(T)

    # A table under the part, carrying the *same* transform. The tilt in these
    # frames comes from how the camera is mounted, not from the part being
    # propped up, so table and part stay parallel -- building the table square to
    # the camera instead leaves the two crossing each other, and the plane fit
    # then latches onto whichever won that frame.
    part_lo = mesh.bounds[0]
    table = trimesh.creation.box(extents=[1.0, 1.0, 0.01])
    table.apply_translation([0.0, 0.0, part_lo[2] - 0.005])
    table.apply_transform(T)
    scene = trimesh.util.concatenate([scene, table])
    # the full frame, not a crop: a window tight enough to be quick also cuts
    # the table out of shot, and then the dominant plane is the part's own face
    us = np.arange(W)
    vs = np.arange(H)
    u, v = np.meshgrid(us, vs)
    pix = np.stack([(u.ravel() - K[0, 2]) / K[0, 0],
                    (v.ravel() - K[1, 2]) / K[1, 1], np.ones(u.size)], axis=1)
    loc, idx, _ = scene.ray.intersects_location(np.zeros_like(pix), pix,
                                                multiple_hits=False)
    d = np.full(u.size, np.nan)
    d[idx] = loc[:, 2]
    d = d.reshape(u.shape) + rng.normal(0, noise, u.shape)
    surface = np.nanpercentile(d, 20)
    d[d > surface + drop] = np.nan       # sensor gives up inside the cavities
    Kc = K.copy()
    Kc[0, 2] -= us[0]
    Kc[1, 2] -= vs[0]
    return d, Kc


def run(yaw_deg, dist, seed, tilt_deg=24.4):
    mesh = trimesh.load(MESH, force='mesh')
    if float(np.max(mesh.extents)) > 1.0:
        mesh.apply_scale(0.001)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    W, H = 640, 360           # half res: the ray caster is the bottleneck here,
    f = (W / 2) / np.tan(np.radians(87 / 2))   # and this only checks the geometry
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])

    T = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    T = trimesh.transformations.rotation_matrix(np.radians(tilt_deg), [1, 0, 0]) @ T
    T = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1]) @ T
    T[:3, 3] = [0, 0, dist]

    depth, Kc = render(mesh, T, K, W, H, seed=seed)
    pts = dpl.deproject(depth, Kc)
    plane = dpl.fit_plane_ransac(pts, seed=seed)
    if plane is None:
        return None, 'plane fit failed', 0
    n, o, _ = plane
    work, _ = dpl.object_above_plane(depth, Kc, n, o,
                                     expected_size=REF['work_size_m'])
    if work is None:
        return None, 'work not isolated', 0
    step = dpl.grid_step_for(float(np.median(work[:, 2])), Kc[0, 0])
    hh = dpl.height_above_plane(work, n, o)
    top = work[hh > np.percentile(hh, 85) - 0.004]
    rect = dpl.rectify(work, n, top.mean(axis=0), step=step)
    ops = dpl.find_openings(rect, min_area=2e-5, max_area=3e-4)
    T_est, info = dpl.solve_opening_pattern(rect, ops, REF['ports'])
    if T_est is None:
        return None, info, len(ops)

    pos = np.linalg.norm(T_est[:3, 3] - T[:3, 3]) * 1000
    dR = T_est[:3, :3].T @ T[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return (pos, ang, info), None, len(ops)


def main():
    print(f'platform: {len(REF["ports"])} ports in the CAD table, '
          f'asymmetry {REF["asymmetry_pct"]:.1f}%')
    for dist in (0.20, 0.25):
        print(f'\n--- working distance {dist*100:.0f} cm ---')
        good = []
        for yaw in (0, 95, 220):
            res, err, n_ops = run(yaw, dist, seed=1)
            if res is None:
                print(f'  yaw {yaw:3d}: FAILED ({n_ops} openings) -- {err}')
                continue
            pos, ang, info = res
            ok = pos < 3.0 and ang < 2.0
            good.append(ok)
            print(f'  yaw {yaw:3d}: {"ok " if ok else "BAD"} '
                  f'{n_ops} openings, {info["n_inliers"]}/{info["n_cad"]} matched, '
                  f'residual {info["mean_residual_mm"]:.2f} mm | '
                  f'position {pos:5.2f} mm, orientation {ang:5.2f} deg')
        if good:
            print(f'  -> {sum(good)}/{len(good)} within 3mm & 2deg')


if __name__ == '__main__':
    main()
