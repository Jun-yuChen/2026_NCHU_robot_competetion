"""Run the monocular pipeline on a captured frame and check it against the truth.

The frame is a real one off the D405, and the labelling below is the physical
platform, read off the hardware -- not off an earlier run of this code. An
earlier version matched all eight ports and got the two outer columns swapped,
which every geometric score in the pipeline called a perfect fit, so a test that
compares the code against itself would have passed it.

    python3 tools/test_mono.py [frame.png] [K.npy] [depth.npy]
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from py_gripper import mono_pose_lib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'test_data')
REF = json.load(open(os.path.join(HERE, '..', 'config',
                                  'opening_reference.json')))['server1']

# port name -> where it sits on the real panel, as seen in the reference frame
TRUTH = {
    (646.8, 202.7): 'rj451', (755.8, 204.5): 'usb3',  (867.6, 206.7): 'hdmi2',
    (646.5, 265.0): 'usb2',  (742.7, 265.4): 'hdmi1', (858.4, 272.8): 'rj452',
    (644.6, 324.2): 'usb1',  (850.2, 331.3): 'usb4',
}


def truth_for(pt, tol=12.0):
    for (x, y), name in TRUTH.items():
        if abs(pt[0] - x) < tol and abs(pt[1] - y) < tol:
            return name
    return None


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'plat_color.png')
    k_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, 'plat_K.npy')
    d_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA, 'plat_depth.npy')
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f'cannot read {img_path}')
        return 1
    K = np.load(k_path)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ports = REF['ports']

    mask, outline = mpl.platform_mask(grey)
    assert mask is not None, 'platform not found'
    px = mpl.silhouette_scale(outline, REF['work_size_m'])
    print(f'[1] platform found, silhouette scale {px:.0f} px/m')

    cands, m = mpl.find_ports(grey, mask, px, ports, K=K)
    print(f'[2] {len(cands)} port candidates')
    assert m is not None, 'no correspondence'
    print(f'[3] matched {m["n_matched"]}/{m["n_cad"]}, '
          f'scale {m["scale_px_per_m"]:.0f} px/m')

    img = np.array([cands[di]['centre'] for _, di in m['pairs']])
    sol = mpl.solve_pose(m['pairs'], ports, img, K)
    assert sol is not None, 'PnP found no camera-facing solution'
    rvec, tvec, err = sol
    print(f'[4] coarse pose: reproj {err:.2f} px, z {tvec.ravel()[2]*1000:.1f} mm')

    fixed = mpl.refine_centroids(grey, m['pairs'], ports, rvec, tvec, K)
    img2 = np.array([f if f is not None else img[i] for i, f in enumerate(fixed)])
    sol = mpl.solve_pose(m['pairs'], ports, img2, K) or sol
    rvec, tvec, err = sol
    mm_px = tvec.ravel()[2] / K[0, 0] * 1000
    print(f'[5] refined pose: reproj {err:.2f} px ({err*mm_px:.2f} mm), '
          f'z {tvec.ravel()[2]*1000:.1f} mm')

    if os.path.exists(d_path):
        inner = cv2.erode(mask, np.ones((21, 21), np.uint8))
        n = mpl.plane_normal_from_depth(np.load(d_path), K, inner)
        if n is not None:
            print(f'[6] depth cross-check: plane normal {np.round(n, 3)}, '
                  f'{mpl.tilt_degrees(rvec, n):.1f} deg from the pose '
                  f'(a plain least-squares fit over the same mask says 15.1 -- '
                  f'that is the handle, not the pose)')

    print('\n[7] correspondence against the physical panel:')
    bad = 0
    for k, (ci, di) in enumerate(m['pairs']):
        got = ports[ci]['name']
        want = truth_for(img[k])
        ok = want is None or got == want
        bad += not ok
        print(f'    ({img[k][0]:6.1f},{img[k][1]:6.1f})  code says {got:6s} '
              f'truth {str(want):6s}  {"ok" if ok else "WRONG"}')
    print(f'    -> {len(m["pairs"]) - bad}/{len(m["pairs"])} correct')

    vis = mpl.debug_image(bgr, outline, cands, m['pairs'], ports, img2,
                          rvec, tvec, K, target='usb2')
    out = os.path.join(DATA, 'mono_check.png')
    cv2.imwrite(out, vis)
    print(f'    wrote {out}')
    return 1 if bad else 0


def check_second_frame():
    """The same panel turned 90 deg, shot at a different distance and light.

    A second frame is here because every regression this pipeline has had was
    invisible in the first one: a threshold window sized for one working
    distance, a blob-width rule that held on one lighting and collapsed on
    another. One frame cannot tell a general rule from a fitted constant.
    """
    img_path = os.path.join(DATA, 'plat_rot90.png')
    k_path = os.path.join(DATA, 'plat_rot90_K.npy')
    if not (os.path.exists(img_path) and os.path.exists(k_path)):
        return 0
    bgr = cv2.imread(img_path)
    K = np.load(k_path)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ports = REF['ports']

    # read off the hardware: rows top to bottom, left to right in this frame
    layout = [['hdmi2', 'rj452', 'usb4'], ['usb3', 'hdmi1'],
              ['rj451', 'usb2', 'usb1']]

    mask, outline = mpl.platform_mask(grey)
    px = mpl.silhouette_scale(outline, REF['work_size_m'])
    cands, m = mpl.match_or_none(grey, mask, px, ports, K)
    if m is None:
        print('\n[8] second frame (turned 90 deg): NO MATCH')
        return 1
    img = np.array([cands[di]['centre'] for _, di in m['pairs']])
    sol = mpl.solve_pose(m['pairs'], ports, img, K)
    # group into rows by y first, then order each row left to right -- sorting
    # by (y, x) up front does not do that, because ports in one row differ by a
    # pixel or two in y and that dominates the comparison
    pts = sorted(((cands[di]['centre'][1], cands[di]['centre'][0]),
                  ports[ci]['name']) for ci, di in m['pairs'])
    rows, cur, last_y = [], [], None
    for (y, x), name in pts:
        if last_y is not None and y - last_y > 40:
            rows.append(cur); cur = []
        cur.append((x, name)); last_y = y
    rows.append(cur)
    rows = [[n for _, n in sorted(r)] for r in rows]
    ok = rows == layout
    print(f'\n[8] second frame (turned 90 deg): {m["n_matched"]}/{m["n_cad"]} matched, '
          f'reproj {sol[2]:.2f} px')
    print(f'    rows found : {rows}')
    print(f'    on hardware: {layout}   {"ok" if ok else "MISMATCH"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main() or check_second_frame())
