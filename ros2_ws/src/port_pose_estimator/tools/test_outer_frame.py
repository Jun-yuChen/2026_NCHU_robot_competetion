"""Pose from the panel's own outer edge, not from matching its ports.

Every other method here (the shipping path, and test_direct_target.py's
frame_rect) solves for pose from many small things -- a dozen ports, each of
which can be missed, mismatched, or measured a few px off. This is the other
extreme: one single, large, unambiguous target. The metal shield's own outer
silhouette is a plain rectangle of a size the CAD already gives exactly
(work_size_m), it is an order of magnitude bigger in the image than any one
port, and Canny finds a plain rectangle far more reliably than it finds a
small recessed opening -- there is nothing inside it to confuse the edge
with.

The trade a big single target makes: translation and in-plane rotation are
pinned by a large baseline, which is exactly where the ports collectively did
well too, but *tilt* (rotation into the plane, coplanar PnP's known weak
axis -- see mono_pose_lib.refit_with_normal) gets no help at all from using
four corners instead of many scattered points; if anything a single rectangle
is a *worse*-conditioned coplanar target than ports spread further apart. So
this is not a strict upgrade -- it is a different bet, worth comparing
against the others on the same real photo rather than assumed better.

Once this pose exists, every hole's position is a straight `pose (x) CAD
centre` projection -- the same operation the shipping path uses, just fed a
pose that came from the frame instead of from the ports.

    python3 tools/test_outer_frame.py [frame.png] [K.npy] [depth.npy]
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
TARGETS = [p['name'] for p in REF['ports'] if p['kind'] != 'other']


def outer_corners_mesh():
    """The panel's own outer rectangle, in the CAD's mesh frame. -> (4,3).

    build_reference.py centres every mesh on its own bounding-box middle
    (`mesh.apply_translation(-mesh.bounds.mean(axis=0))`) before measuring
    anything, so the outer rectangle is exactly +-work_size_m/2 around the
    origin -- no separate corner record was ever saved because it never
    needed to be; it falls straight out of the size already in the reference.
    The z is the ports' own z (they share one value): the panel's general
    face, not the raised mounting lip along its top edge that this session
    already found sits ~2.5 mm proud of it -- close enough for a rectangle
    that is 150+ mm across, but worth remembering if this method's own tilt
    ever needs explaining.
    """
    hx, hy = REF['work_size_m'][0] / 2, REF['work_size_m'][1] / 2
    z = REF['ports'][0]['centre'][2]
    return np.array([[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]],
                    dtype=np.float64)


def find_outer_quad(grey, mask, px_per_m, work_size_m, lo=40, hi=120,
                    close=15, tol=0.25):
    """The shield's own outer rectangle in the image, from Canny. -> (4,2) or None.

    Searched only within a generous margin around the depth mask, not the
    whole frame -- otherwise the case's own outer edge, a monitor behind it,
    anything else rectangular in shot is fair game for "biggest rectangle"
    to land on. Candidates are scored against the CAD's own aspect ratio and
    the depth-given scale's predicted size, the same two checks
    mask_size_consistent already uses for the depth mask itself, so a
    contour has to agree with both the shape and the range this panel is
    actually known to be at, not just look rectangular.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = int(0.15 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = min(xs.max() + pad, grey.shape[1]), min(ys.max() + pad, grey.shape[0])
    crop = grey[y0:y1, x0:x1]

    e = cv2.Canny(cv2.GaussianBlur(crop, (5, 5), 0), lo, hi)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)))
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exp = np.array(sorted(work_size_m)) * px_per_m
    best = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        (_, _), (w, h), ang = cv2.minAreaRect(c)
        obs = sorted([w, h])
        if obs[0] < 20:
            continue
        if not all(1 - tol <= o / e_ <= 1 + tol for o, e_ in zip(obs, exp)):
            continue
        area = cv2.contourArea(c)
        if best is None or area > best[0]:
            box = cv2.boxPoints(cv2.minAreaRect(c))
            best = (area, box + [x0, y0])
    return None if best is None else best[1]


def solve_outer_pose(quad_mesh, quad_img, K):
    """PnP over the outer quad, trying every corner correspondence. -> (rvec,tvec,err).

    boxPoints (and any other rectangle finder) has no notion of which of its
    4 points is which physical corner, and getting this wrong does not fail
    loudly -- it hands back a *different*, still self-consistent pose, a
    smaller-scale echo of the exact correspondence-ambiguity problem
    match_ports solves for the ports themselves (see its own docstring on
    the wrong-but-full-count hypothesis). A first attempt here that matched
    corners by polar angle around each quad's own centroid produced exactly
    that: a technically valid pose with every hole projected to roughly the
    panel's mirror-opposite end. There are only 8 possible correspondences
    for a quadrilateral (4 rotations x 2 reflections), cheap enough to just
    try all of them and keep whichever reprojects best, the same rule
    match_ports itself is decided by.
    """
    best = None
    for flip in (False, True):
        q = quad_img[::-1] if flip else quad_img
        for k in range(4):
            order = np.roll(q, k, axis=0)
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                quad_mesh, order, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            for rvec, tvec in zip(rvecs, tvecs):
                rp, _ = cv2.projectPoints(quad_mesh, rvec, tvec, K, None)
                err = float(np.linalg.norm(rp.reshape(-1, 2) - order, axis=1).mean())
                facing = float(cv2.Rodrigues(rvec)[0][2, 2]) < 0
                if facing and (best is None or err < best[0]):
                    best = (err, rvec, tvec)
    return best


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
        D = D / 1000.0
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ports = REF['ports']
    name2idx = {p['name']: i for i, p in enumerate(ports)}

    mask = px = None
    for m_, o_ in mpl.platform_from_depth(D, K, max_n=3):
        p_ = mpl.scale_from_depth(D, m_, K)
        if p_ and mpl.mask_size_consistent(m_, p_, REF['work_size_m']):
            mask, px = m_, p_
            break
    assert px, 'no depth-isolated candidate matched the CAD panel size'
    print(f'[1] panel found, scale {px:.0f} px/m')

    quad_img = find_outer_quad(grey, mask, px, REF['work_size_m'])
    assert quad_img is not None, 'no Canny rectangle matched the panel outline'
    print(f'[2] outer quad (image px):')
    for c in quad_img:
        print(f'      ({c[0]:7.1f}, {c[1]:7.1f})')

    quad_mesh = outer_corners_mesh()
    best = solve_outer_pose(quad_mesh, quad_img, K)
    assert best is not None, 'no facing correspondence reprojected'
    err, rvec, tvec = best
    print(f'[3] pose from the outer quad alone: reproj {err:.2f} px, '
          f'z {tvec.ravel()[2]*1000:.1f} mm')

    allobj = np.array([p['centre'] for p in ports], dtype=np.float64)
    proj, _ = cv2.projectPoints(allobj, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)

    print(f'\n[4] every target hole, pose(x)CAD via the outer frame:')
    vis = bgr.copy()
    cv2.polylines(vis, [np.round(quad_img).astype(int)], True, (255, 128, 0), 2)
    for name in TARGETS:
        i = name2idx[name]
        uv = proj[i]
        print(f'    {name:8s} ({uv[0]:6.1f},{uv[1]:6.1f})')
        cv2.drawMarker(vis, tuple(np.round(uv).astype(int)), (0, 255, 255),
                       cv2.MARKER_CROSS, 13, 2)
        cv2.putText(vis, name, tuple(np.round(uv).astype(int) + [7, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    x0f, y0f = quad_img.min(axis=0).astype(int)
    x1f, y1f = quad_img.max(axis=0).astype(int)
    H, W = vis.shape[:2]
    pad = 40
    vis = vis[max(0, y0f - pad):min(H, y1f + pad), max(0, x0f - pad):min(W, x1f + pad)]
    s = 2000 / max(vis.shape[:2])
    if s < 1.0:
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = os.path.join(DATA, 'outer_frame_case1.png')
    cv2.imwrite(out, vis)
    print(f'\n    orange = the Canny outer quad used for PnP, yellow = every '
          f'hole projected through it')
    print(f'    wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
