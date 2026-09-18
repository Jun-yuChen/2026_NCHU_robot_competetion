"""Run the case1 monocular pipeline on a captured frame and check it against truth.

case1's panel sits in front of clutter that is comparably dark and touching it in
frame (a chair, boxes, shelving), which is what platform_from_depth exists for --
see mono_pose_lib and the platform_from/port_ladder/min_pairs entries build_reference.py
writes into this model's reference. server1's test_mono.py exercises the plain
Otsu+find_ports path; this file exercises the depth-isolated path and the
block/offset retry ladder, which server1 never touches.

The frame is a real D405 capture at the arm's fixed approach pose for this panel,
with the reflective branding text physically covered (see the frame before that
fix, case108, for what an unmatched false-positive point does to the whole pose --
it is not part of this regression, only the fixed case109 is).

    python3 tools/test_mono_case1.py [frame.png] [K.npy] [depth.npy]
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

# The ports the task actually targets (usb/hdmi/rj45); the rest of the CAD
# table is unlabelled panel features used only to constrain the pose.
TARGETS = {p['name'] for p in REF['ports'] if p['kind'] != 'other'}

# Read off case109_color.png by hand, in pixels -- not from an earlier run of
# this code, same reasoning as server1's test_mono.py.
#
# Hand-read values carry real error, and it is the same size as the
# differences anyone is likely to measure against them: usb1/usb2 were once
# 8 px out, read off the port's top edge rather than its centre, and usb7 was
# 3.5 px out the same way. usb7's y was re-derived from the brightness profile
# across the port -- the silver frame shows as two peaks at y=264 and y=283,
# whose 19 px separation matches this port's own CAD short side (19.7 px at
# this range) to within a pixel, which is what says the peaks found are the
# port's real edges and not some feature inside it. The same pass over
# usb3/usb4 found peaks only 36 px apart against a 48.9 px CAD long side, so
# those are the blue tongue's edges rather than the opening's and were left
# alone. Anything measured against this table to better than a few px is
# measuring this table's own error, not the pipeline's.
TRUTH_PX = {
    'usb1': (615, 76), 'usb2': (643, 76), 'usb7': (590, 274),
    'hdmi1': (586, 329), 'usb3': (624, 327), 'usb4': (647, 327),
    'usb5': (589, 390), 'usb6': (612, 394), 'rj451': (646, 394),
}
# A target hit directly by the detector should land within a couple of its own
# measurement noise; one recovered purely by projection from the pose (no
# direct detection at all) is checked to a looser bound, because it is only as
# good as the pose's extrapolation to that part of the panel.
MATCHED_TOL_PX = 10.0
PROJECTED_TOL_PX = 20.0


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'case109_color.png')
    k_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, 'case109_K.npy')
    d_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(DATA, 'case109_depth.npy')
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f'cannot read {img_path}')
        return 1
    K = np.load(k_path)
    D = np.load(d_path)
    if D.max() > 10:
        D = D / 1000.0                  # mm -> m, if the capture saved it raw
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ports = REF['ports']

    blobs = mpl.platform_from_depth(D, K, max_n=3)
    assert blobs, 'depth-isolated platform not found'
    mask, outline, px = None, None, None
    for m_, o_ in blobs:
        p_ = mpl.scale_from_depth(D, m_, K)
        if p_ and mpl.mask_size_consistent(m_, p_, REF['work_size_m']):
            mask, outline, px = m_, o_, p_
            break
    assert px, 'no depth-isolated candidate had a scale consistent with the CAD panel size'
    print(f'[1] platform found (depth-isolated), scale {px:.0f} px/m')

    ladder = REF['port_ladder']
    cands, m = mpl.find_ports_ladder(grey, mask, px, ports, ladder, K=K,
                                     min_pairs=REF.get('min_pairs'))
    print(f'[2] {len(cands)} port candidates')
    assert m is not None, 'no correspondence'
    assert m['verified'], 'match did not verify against reprojection'
    print(f'[3] matched {m["n_matched"]}/{m["n_cad"]}, '
          f'scale {m["scale_px_per_m"]:.0f} px/m')

    img = np.array([cands[di]['centre'] for _, di in m['pairs']])
    sol = mpl.solve_and_refine(m['pairs'], ports, img, grey, K, candidates=cands)
    assert sol is not None, 'PnP found no camera-facing solution'
    rvec, tvec, err, img, pairs = sol
    print(f'[4] pose: reproj {err:.2f} px, z {tvec.ravel()[2]*1000:.1f} mm '
          f'({len(pairs)}/{len(m["pairs"])} pairs kept after the size check)')

    matched_names = {ports[ci]['name']: k for k, (ci, _) in enumerate(pairs)}
    name2idx = {p['name']: i for i, p in enumerate(ports)}

    # Same tilt cross-check and fixed-normal refit depth_pose_node.py applies
    # live (see its _estimate_mono) -- coplanar PnP is weakest exactly on
    # tilt, so a port far from every matched anchor can look fine near them
    # and be well off further away even though the fit at the anchors
    # themselves is genuinely good. Skipping this step here would score the
    # pipeline against a pose the real system does not actually publish.
    if os.path.exists(d_path):
        inner = cv2.erode(mask, np.ones((21, 21), np.uint8))
        n = mpl.plane_normal_from_depth(D, K, inner)
        if n is not None:
            tilt = mpl.tilt_degrees(rvec, n)
            print(f'\n[6] depth cross-check: plane normal {np.round(n, 3)}, '
                  f'{tilt:.1f} deg from the pose')
            fixed_sol = mpl.refit_with_normal(pairs, ports, img, K, n, rvec, tvec)
            if fixed_sol is not None and fixed_sol[2] <= max(err * 2, 4.0):
                rvec, tvec, err = fixed_sol
                print(f'    fixed-normal refit accepted: reproj now {err:.2f} px')

    allobj = np.array([p['centre'] for p in ports], dtype=np.float64)
    proj, _ = cv2.projectPoints(allobj, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)

    print(f'\n[5] all {len(TARGETS)} target ports against the physical panel:')
    bad = []
    for name in sorted(TARGETS):
        tx, ty = TRUTH_PX[name]
        px_, py_ = proj[name2idx[name]]
        e = float(np.hypot(px_ - tx, py_ - ty))
        if name in matched_names:
            how, tol = 'detected ', MATCHED_TOL_PX
        else:
            how, tol = 'projected', PROJECTED_TOL_PX
        ok = e <= tol
        bad.append(not ok)
        print(f'    {name:8s} {how}  proj=({px_:6.1f},{py_:6.1f})  '
              f'truth=({tx},{ty})  err={e:5.1f}px (tol {tol:.0f})  '
              f'{"ok" if ok else "WRONG"}')
    n_bad = sum(bad)
    print(f'    -> {len(TARGETS) - n_bad}/{len(TARGETS)} within tolerance')

    # debug_image draws the unmatched targets itself now (projected, in blue),
    # scaled to each port's own CAD size -- no need to do that here too.
    vis = mpl.debug_image(bgr, outline, cands, pairs, ports, img, rvec, tvec, K,
                          max_side=2000)
    out = os.path.join(DATA, 'mono_check_case1.png')
    cv2.imwrite(out, vis)
    print(f'    wrote {out}')
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
