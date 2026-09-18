"""Prototype: matched ports get their own cross directly, no PnP window
placement; unmatched ports get a real (not display-only) CAD-relative
position from their nearest MEASURED neighbour, drawn in yellow so it reads
as "inferred" rather than "seen". solve_pose is kept, but only to verify the
match (reprojection error) -- it no longer places any search window.

Test-only: does not touch depth_pose_node.py. Writes an annotated PNG per
captured frame so the result can be looked at directly.

    python3 tools/test_pipeline_v2.py
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
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/pipeline_v2'
os.makedirs(OUT, exist_ok=True)
REF = json.load(open(os.path.join(HERE, '..', 'config',
                                  'opening_reference.json')))['case1']
PORTS = REF['ports']
CAD_XY = {p['name']: np.array(p['centre']) for p in PORTS if p['kind'] != 'other'}
REAL_NAMES = [p['name'] for p in PORTS if p['kind'] != 'other']


def process(case):
    bgr = cv2.imread(os.path.join(DATA, f'{case}_color.png'))
    if bgr is None:
        return None, 'no image'
    K = np.load(os.path.join(DATA, f'{case}_K.npy'))
    D = np.load(os.path.join(DATA, f'{case}_depth.npy'))
    if D.max() > 10:
        D = D / 1000.0
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # steps 1-2: find the panel, get a scale
    mask = px = None
    for m_, o_ in mpl.platform_from_depth(D, K, max_n=3):
        hint = mpl.scale_from_depth(D, m_, K)
        if hint and mpl.mask_size_consistent(m_, hint, REF['work_size_m']):
            mask, px = m_, hint
            break
    if px is None:
        return None, 'no panel'

    # step 3: candidates (unlabelled)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cands = mpl.canny_candidates(grey, inner, px, PORTS)

    # step 4: match
    match = mpl.match_ports(cands, PORTS, px, K=K, min_pairs=REF.get('min_pairs'))
    if match is None or not match.get('verified'):
        return None, 'no verified match'

    # step 5, kept ONLY as a verification gate -- no longer used to place
    # any window. A high reprojection error here is exactly the "wrong
    # correspondence with more inliers" case match_ports' own docstring
    # warns it cannot catch by itself.
    img = np.array([cands[di]['centre'] for _, di in match['pairs']])
    sol = mpl.solve_pose(match['pairs'], PORTS, img, K, None)
    if sol is None:
        return None, 'no facing pose -- correspondence rejected'
    rvec, tvec, err = sol
    max_reproj = REF.get('max_reproj_px', 4.0)
    if err > max_reproj:
        return None, f'reprojection {err:.2f}px > {max_reproj} -- correspondence rejected'

    eroded = cv2.erode(mask, np.ones((21, 21), np.uint8))
    plane = mpl.plane_from_depth(D, K, eroded)
    if plane is None:
        return None, 'no depth plane'
    origin, normal = plane
    R = cv2.Rodrigues(rvec)[0]

    matched_uv = {PORTS[ci]['name']: cands[di]['centre'] for ci, di in match['pairs']}

    # Matched ports: own candidate position anchors the window directly --
    # _port_quad still supplies the CAD-accurate shape/size/orientation for
    # frame_rect/ring_rect to search inside, just recentred here instead of
    # at wherever the coarse pose happened to project it.
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

    # Unmatched (or matched-but-frame_rect-missed) ports: CAD-relative offset
    # from the nearest ALREADY-MEASURED port -- real position now, not a
    # display-only fallback. Repeats until nothing more can be chained (a
    # recovered port can itself anchor the next one).
    recovered = {}
    pending = [n for n in REAL_NAMES if n not in measured]
    changed = True
    known = dict(measured)
    while changed and pending:
        changed = False
        for name in list(pending):
            if not known:
                break
            nn = min(known, key=lambda m: np.linalg.norm(CAD_XY[name] - CAD_XY[m]))
            P_nn = mpl.ray_plane(known[nn], K, origin, normal) if nn in measured or nn in recovered else None
            # need the *3D* point of nn, not just its pixel -- recompute via
            # ray_plane uniformly for both measured and recovered nn's
            uv_nn = measured.get(nn, recovered.get(nn))
            P_nn = mpl.ray_plane(uv_nn, K, origin, normal)
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

    return (measured, recovered, mask), None


def draw(case, measured, recovered, mask):
    bgr = cv2.imread(os.path.join(DATA, f'{case}_color.png'))
    vis = bgr.copy()
    for name, uv in measured.items():
        pt = tuple(np.round(uv).astype(int))
        cv2.drawMarker(vis, pt, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, name, (pt[0] + 6, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 0, 255), 1, cv2.LINE_AA)
    for name, uv in recovered.items():
        pt = tuple(np.round(uv).astype(int))
        cv2.drawMarker(vis, pt, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, name, (pt[0] + 6, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 255), 1, cv2.LINE_AA)
    x, y, w, h = cv2.boundingRect(mask)
    pad = 60
    H, W = vis.shape[:2]
    vis = vis[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    scale = 700 / max(vis.shape[:2])
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return vis


def main():
    cases = ['case102', 'case106', 'case107', 'case108', 'case109', 'case110']
    for case in cases:
        result, err = process(case)
        if err:
            print(f'{case}: {err}')
            continue
        measured, recovered, mask = result
        u1u2 = (np.linalg.norm(measured['usb1'] - measured['usb2'])
                if 'usb1' in measured and 'usb2' in measured else None)
        print(f'{case}: {len(measured)} measured (magenta), {len(recovered)} '
             f'recovered (yellow) -- usb1/usb2 both measured: '
             f'{"usb1" in measured and "usb2" in measured}'
             + (f', px apart {u1u2:.1f}' if u1u2 else ''))
        vis = draw(case, measured, recovered, mask)
        out_path = os.path.join(OUT, f'{case}_v2.png')
        cv2.imwrite(out_path, vis)
        print(f'  wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
