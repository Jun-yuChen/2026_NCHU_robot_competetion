"""test_pipeline_v2's new architecture (matched ports anchor their own window,
unmatched ports get a real CAD-relative position from their nearest measured
neighbour), replayed on case109 many times with injected sensor-level noise --
same idea as mono_pose_lib's own "one frame replayed 30 times with
sensor-level noise across N seeds" validation for plane_from_depth.

case109 is the one frame in this repo with a hand-verified truth table
(test_mono_case1.py's TRUTH_PX), so this is the only place accuracy can be
checked against real ground truth instead of just internal agreement --
including for the *recovered* (yellow-cross) ports, which no other test in
this repo can score at all.

    python3 tools/test_pipeline_v2_noise.py [n_trials]
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
                                  'opening_reference.json')))['case1']
PORTS = REF['ports']
CAD_XY = {p['name']: np.array(p['centre']) for p in PORTS if p['kind'] != 'other'}
REAL_NAMES = [p['name'] for p in PORTS if p['kind'] != 'other']

TRUTH_PX = {
    'usb1': (615, 76), 'usb2': (643, 76), 'usb7': (590, 274),
    'hdmi1': (586, 329), 'usb3': (624, 327), 'usb4': (647, 327),
    'usb5': (589, 390), 'usb6': (612, 394), 'rj451': (646, 394),
}

# D405-scale sensor noise: same order of magnitude mono_pose_lib.py's own
# plane_from_depth validation and tools/test_platform.py's synthetic render
# already use for this camera (~0.4-0.5mm depth std).
DEPTH_NOISE_M = 0.0003
GREY_NOISE = 0.5


def process(grey, D, K, seed):
    bgr_for_hsv = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr_for_hsv, cv2.COLOR_BGR2HSV)

    mask = px = None
    for m_, o_ in mpl.platform_from_depth(D, K, max_n=3):
        hint = mpl.scale_from_depth(D, m_, K)
        if hint and mpl.mask_size_consistent(m_, hint, REF['work_size_m']):
            mask, px = m_, hint
            break
    if px is None:
        return None, 'no panel'

    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cands = mpl.canny_candidates(grey, inner, px, PORTS)
    match = mpl.match_ports(cands, PORTS, px, K=K, min_pairs=REF.get('min_pairs'))
    if match is None or not match.get('verified'):
        return None, 'no verified match'

    img = np.array([cands[di]['centre'] for _, di in match['pairs']])
    sol = mpl.solve_pose(match['pairs'], PORTS, img, K, None)
    if sol is None:
        return None, 'no facing pose'
    rvec, tvec, err = sol
    if err > REF.get('max_reproj_px', 4.0):
        return None, f'reprojection {err:.2f}px rejected'

    eroded = cv2.erode(mask, np.ones((21, 21), np.uint8))
    plane = mpl.plane_from_depth(D, K, eroded)
    if plane is None:
        return None, 'no depth plane'
    origin, normal = plane
    R = cv2.Rodrigues(rvec)[0]

    matched_uv = {PORTS[ci]['name']: cands[di]['centre'] for ci, di in match['pairs']}
    measured = {}
    for port in PORTS:
        if port['kind'] == 'other' or port['name'] not in matched_uv:
            continue
        name = port['name']
        quad = mpl._port_quad(port, rvec, tvec, K, None)
        quad = quad - quad.mean(axis=0) + matched_uv[name]
        rect = mpl.ring_rect(grey, hsv, quad) if name == 'hdmi1' else mpl.frame_rect(grey, hsv, quad)
        if rect is None:
            continue
        l, t, r, b = rect
        measured[name] = np.array([(l + r) / 2, (t + b) / 2])

    recovered = {}
    known = dict(measured)
    pending = [n for n in REAL_NAMES if n not in known]
    changed = True
    while changed and pending:
        changed = False
        for name in list(pending):
            if not known:
                break
            nn = min(known, key=lambda m: np.linalg.norm(CAD_XY[name] - CAD_XY[m]))
            P_nn = mpl.ray_plane(known[nn], K, origin, normal)
            if P_nn is None:
                continue
            d_cad = CAD_XY[name] - CAD_XY[nn]
            P_name = P_nn + R @ d_cad
            rp, _ = cv2.projectPoints(P_name.reshape(1, 3), np.zeros(3), np.zeros(3), K, None)
            uv = rp.reshape(2)
            recovered[name] = uv
            known[name] = uv
            pending.remove(name)
            changed = True

    return (measured, recovered), None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    grey0 = cv2.cvtColor(cv2.imread(os.path.join(DATA, 'case109_color.png')), cv2.COLOR_BGR2GRAY)
    K = np.load(os.path.join(DATA, 'case109_K.npy'))
    D0 = np.load(os.path.join(DATA, 'case109_depth.npy'))
    if D0.max() > 10:
        D0 = D0 / 1000.0

    stats = {name: {'measured': [], 'recovered': [], 'missing': 0} for name in REAL_NAMES}
    fails = 0
    for trial in range(n):
        rng = np.random.default_rng(trial)
        grey = np.clip(grey0.astype(np.float64) + rng.normal(0, GREY_NOISE, grey0.shape),
                      0, 255).astype(np.uint8)
        D = D0 + rng.normal(0, DEPTH_NOISE_M, D0.shape)
        D[D0 == 0] = 0
        result, err = process(grey, D, K, trial)
        if err:
            fails += 1
            continue
        measured, recovered = result
        for name in REAL_NAMES:
            truth = np.array(TRUTH_PX[name])
            if name in measured:
                stats[name]['measured'].append(float(np.linalg.norm(measured[name] - truth)))
            elif name in recovered:
                stats[name]['recovered'].append(float(np.linalg.norm(recovered[name] - truth)))
            else:
                stats[name]['missing'] += 1

    print(f'{n} trials, {fails} produced no pose at all (whole-panel rejection)\n')
    print(f'{"port":8s} {"measured":>8s} {"meas err":>10s} {"recovered":>10s} {"rec err":>10s} {"missing":>8s}')
    for name in REAL_NAMES:
        s = stats[name]
        me = f'{np.mean(s["measured"]):.1f}px' if s['measured'] else '--'
        re = f'{np.mean(s["recovered"]):.1f}px' if s['recovered'] else '--'
        print(f'{name:8s} {len(s["measured"]):8d} {me:>10s} {len(s["recovered"]):10d} '
             f'{re:>10s} {s["missing"]:8d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
