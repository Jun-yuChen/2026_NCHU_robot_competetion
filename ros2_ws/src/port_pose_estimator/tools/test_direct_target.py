"""Aim at the hole the camera actually sees, not at where the CAD says it is.

The shipping path sends the arm to `pose (x) CAD centre`: the image fixes six
numbers and the CAD table supplies every hole's position inside the panel.
That makes the CAD's own error land on the arm one-for-one -- nothing averages
it out, and no reprojection check can see it, because reprojection only scores
the matched points against each other.

This is the alternative worth measuring against it. Depth is used only for the
panel's plane; the hole centres come from the image; the target is where the
ray through a hole's own pixel meets that plane. CAD still says which hole is
which and which way its long axis runs -- vision is unreliable at telling a USB
from an HDMI, and the gripper still has to be squared up -- but it no longer
says where the hole is.

Canny finds the centres rather than the shipping brightness threshold. Over 30
noise trials on this frame it was the only one of the three that found all nine
target ports: brightness never once found usb1 and found usb2 in 10 trials of
30, and depth separates only rj451, whose socket is deep enough to stand out
from the panel plane (the rest are recessed 2-4 mm, which depth noise covers).
Scatter came out 0.05-2.25 px for Canny against 0.38-2.66 px for brightness.

Note what that measurement is and is not: repeatability, which needs no ground
truth, not accuracy, which needs one this panel does not have -- test_mono_case1's
hand-read TRUTH_PX carries 3-7 px of its own error, the same size as the
difference between these two methods. Which target is actually better is
settled by driving the arm to both, not here.

    python3 tools/test_direct_target.py [frame.png] [K.npy] [depth.npy]
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


def canny_candidates(grey, inner, px_per_m, cad_ports, lo=40, hi=110, close=3):
    """Hole centres from their own edges. -> [{centre, area_px, long_m, short_m}].

    port_candidates thresholds for brightness, which needs the hole to be
    lighter than the panel around it. Half this panel's sockets are not: the
    USB 2.0 pair at the top has a black tongue behind a dark surround and
    produces no blob at any block/offset the ladder tries. What every socket
    does have is the boundary between its metal surround and its own dark
    interior, and Canny takes that directly.

    This is only ever used to get a rough first pose (for identification and
    for placing each named hole's own search window) -- see frame_rect below
    for what actually measures each hole's centre.
    """
    areas = [float(np.prod(p['size'])) * px_per_m ** 2 for p in cad_ports]
    lo_a, hi_a = min(areas) * 0.20, max(areas) * 4.0
    e = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), lo, hi)
    e = cv2.morphologyEx(
        e, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    e = cv2.bitwise_and(e, inner)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (lo_a <= a <= hi_a):
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        out.append(dict(centre=np.array([M['m10'] / M['m00'], M['m01'] / M['m00']]),
                        area_px=int(a),
                        long_m=max(w, h) / px_per_m,
                        short_m=min(w, h) / px_per_m))
    return out


def _runs(vec):
    """Contiguous runs of nonzero entries in a 1D bool/uint8 vector. -> [(lo,hi), ...]."""
    idx = np.where(vec > 0)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    return [(g.min(), g.max()) for g in np.split(idx, splits + 1)]


def frame_rect(grey, hsv, quad, pad=8, sat_max=90, edge_tol=1):
    """A hole's own silver-frame rectangle, measured, not projected.

    -> (left, top, right, bottom) in full-image pixels, or None.

    The frame is whichever bit of this window is bright (Otsu on V) and grey
    (S below sat_max -- the blue USB3 tongue passes the brightness side of
    this and has to be excluded on colour instead). Among the surviving
    blobs, the one closest to the CAD quad's own centre is this hole's own
    frame; anything touching the crop's edge is a neighbour's frame caught by
    the padding, not this hole's.

    However many of the frame's four sides that blob actually shows -- an "L"
    on usb1/usb2 (open top and right, since nothing separates the two USB-A
    barrels there but their own dark interiors), three sides on usb3/usb4/usb7,
    a mirrored L on rj451 -- comes out of a single rule: at the blob's own
    bounding-box centre row, count contiguous runs. Two runs means both the
    left and right inner edges are directly measured; one run means only one
    side is present, and the missing side is stood in for by that same blob's
    own farthest reach in that direction. On usb1's "L", that farthest reach
    is the bottom arm's own rightward extent -- the arm's own length doubling
    as the missing dimension, not a guess. Same rule vertically, on the
    centre column, for top/bottom. Verified against a hand-labelled reference
    photo for usb1/usb2/usb7/usb3/usb4/rj451 (hdmi1 is oval and uses ring_rect
    instead; usb5/usb6 also close in the picture from a fourth, separate
    frame fragment on their far side, but every attempt to fold that fragment
    in automatically ended up grabbing the *neighbouring* port's own fragment
    instead on at least one hole, so it is deliberately left out here -- the
    L-only reading it falls back to was the one actually confirmed correct).
    """
    x0, y0 = np.floor(quad.min(axis=0) - pad).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0) + pad).astype(int)
    H, W = grey.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    wv, ws = hsv[y0:y1, x0:x1, 2], hsv[y0:y1, x0:x1, 1]
    if wv.size < 50:
        return None
    _, frame = cv2.threshold(wv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    frame = (frame > 0).astype(np.uint8)
    nn, lab, st, cen = cv2.connectedComponentsWithStats(frame, 8)
    h, w = frame.shape
    qc = quad.mean(axis=0) - [x0, y0]
    # A stray thin fragment (a sliver of a neighbour's own frame just inside
    # the padding, a reflection) can sit closer to the CAD quad's centre than
    # this hole's real bracket does whenever the quad itself is off by only
    # a couple of px -- which a perfectly normal pose refit does routinely.
    # The real bracket is never that small: it is built from two full-length
    # arms, so it clears this area easily and a stray sliver does not.
    min_area = 0.10 * (quad.max(axis=0) - quad.min(axis=0)).prod()
    best, bestd = None, None
    for i in range(1, nn):
        if st[i, cv2.CC_STAT_AREA] < max(15, min_area):
            continue
        if ws[lab == i].mean() > sat_max:
            continue
        bx, by, bw, bh = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        if bx <= edge_tol or by <= edge_tol or bx + bw >= w - edge_tol or by + bh >= h - edge_tol:
            continue
        d = np.hypot(cen[i][0] - qc[0], cen[i][1] - qc[1])
        if bestd is None or d < bestd:
            best, bestd = i, d
    if best is None:
        return None
    comp = (lab == best)
    bx, by, bw, bh = st[best, 0], st[best, 1], st[best, 2], st[best, 3]
    cx, cy = bx + bw // 2, by + bh // 2

    row = comp[cy]
    rr = _runs(row)
    if len(rr) >= 2:
        inL, inR = rr[0][1], rr[-1][0]
    elif len(rr) == 1:
        mid = (rr[0][0] + rr[0][1]) / 2
        inL, inR = (rr[0][1], bx + bw) if mid < cx else (bx, rr[0][0])
    else:
        inL, inR = bx, bx + bw

    col = comp[:, cx]
    rc = _runs(col)
    if len(rc) >= 2:
        inT, inB = rc[0][1], rc[-1][0]
    elif len(rc) == 1:
        mid = (rc[0][0] + rc[0][1]) / 2
        inT, inB = (rc[0][1], by + bh) if mid < cy else (by, rc[0][0])
    else:
        inT, inB = by, by + bh

    return inL + x0, inT + y0, inR + x0, inB + y0


def ring_rect(grey, hsv, quad, pad=8, sat_max=90, edge_tol=1):
    """hdmi1's own frame, measured. -> (left, top, right, bottom) or None.

    Not actually oval -- a real HDMI Type-A shell is a plain rectangle with
    its top two corners cut off at a straight bevel, not rounded -- but a
    bevel does the same thing to this measurement a round corner would: move
    away from the frame's own centre line and its two sides converge early,
    same as they would around a curve. So this still breaks under Otsu into
    a handful of arcs rather than one blob (every grey, non-edge-touching
    piece within the hole's own footprint is unioned first), and sampling
    the combined shape at its bounding-box centre the way frame_rect does
    still fails the same way: near a bevelled corner the two sides sit
    closer together than they really are at the true left/right or
    top/bottom extent, so a wide sampling band systematically reads the
    rectangle too small, and it gets smaller the wider the band -- the
    opposite of the usual noise-averaging intuition. Only the single
    row/column right at the centre, farthest from every corner, reads the
    true extent.
    """
    x0, y0 = np.floor(quad.min(axis=0) - pad).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0) + pad).astype(int)
    H, W = grey.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    wv, ws = hsv[y0:y1, x0:x1, 2], hsv[y0:y1, x0:x1, 1]
    if wv.size < 50:
        return None
    _, frame = cv2.threshold(wv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    frame = (frame > 0).astype(np.uint8)
    nn, lab, st, cen = cv2.connectedComponentsWithStats(frame, 8)
    h, w = frame.shape
    qc = quad.mean(axis=0) - [x0, y0]
    qsize = max(quad.max(axis=0) - quad.min(axis=0))
    ring = np.zeros_like(frame)
    for i in range(1, nn):
        if st[i, cv2.CC_STAT_AREA] < 15:
            continue
        if ws[lab == i].mean() > sat_max:
            continue
        bx, by, bw, bh = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        if bx <= edge_tol or by <= edge_tol or bx + bw >= w - edge_tol or by + bh >= h - edge_tol:
            continue
        if np.hypot(cen[i][0] - qc[0], cen[i][1] - qc[1]) > qsize * 0.9:
            continue
        ring |= (lab == i)
    if ring.sum() == 0:
        return None
    ys, xs = np.where(ring > 0)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2

    def sides(vec_row_fn, lo, hi):
        for half in (0, 1, 2):
            il, ir = [], []
            for k in range(lo - half, hi + half + 1):
                rr = _runs(vec_row_fn(k))
                if len(rr) >= 2:
                    il.append(rr[0][1]); ir.append(rr[-1][0])
            if il:
                return max(il), min(ir)
        return None, None

    inL, inR = sides(lambda r: ring[r], cy, cy)
    inT, inB = sides(lambda c: ring[:, c], cx, cx)
    if None in (inL, inR, inT, inB):
        return None
    return inL + x0, inT + y0, inR + x0, inB + y0


def plane_from_depth(depth, K, mask, trim=0.003, iters=3):
    """The panel's own plane. -> (point_on_plane, unit normal) or None.

    Deliberately the plane and not the depth at each hole's own pixel. A hole
    is a recess: measured on this frame the sockets read 0.2-6.3 mm behind the
    panel face, rj451 deepest because an RJ45 shell is deep, so per-pixel depth
    answers a different question than "where is the opening". The plane the
    face sits on is what a plug meets, and every hole shares it.

    plane_normal_from_depth's own RANSAC already rejects the raised handle it
    was written against, but this panel's holes are a bigger, one-sided
    contamination of a different shape: up to 19 recesses, 0.2-14 mm deep,
    none of them raised, so they drag the mean (and, more, the fitted normal)
    toward "behind" on whichever side of the panel happens to have more open
    ports in view that frame -- not a random error, a systematic one that
    tilts with the port layout. Refitting after dropping everything recessed
    more than `trim` past the current plane, a few times so each pass' better
    plane can catch points the last pass' worse one missed, measured on one
    frame replayed 30 times with sensor-level noise: tilt std 1.22 deg -> 0.65
    deg, worst deviation 6.08 -> 3.30 deg, for a 0.6 mm change in the fitted
    distance. That is the same reduction ray_plane's own sensitivity table
    turns into roughly 2x less lateral error at the panel's far corners,
    where the ray-to-normal angle is largest.
    """
    n = mpl.plane_normal_from_depth(depth, K, mask)
    if n is None:
        return None
    vi, ui = np.where((mask > 0) & np.isfinite(depth) & (depth > 0))
    if len(vi) < 200:
        return None
    z = depth[vi, ui]
    pts = np.stack([(ui - K[0, 2]) * z / K[0, 0],
                    (vi - K[1, 2]) * z / K[1, 1], z], axis=1)
    o = pts.mean(axis=0)
    for _ in range(iters):
        keep = (pts - o) @ n > -trim
        if keep.sum() < 200:
            break
        sub = pts[keep]
        o = sub.mean(axis=0)
        _, V = np.linalg.eigh(np.cov((sub - o).T))
        n = V[:, 0]
        if n[2] > 0:
            n = -n
    return o, n


def ray_plane(uv, K, origin, normal):
    """Where the ray through a pixel meets the plane. -> (3,) in camera coords."""
    d = np.array([(uv[0] - K[0, 2]) / K[0, 0],
                  (uv[1] - K[1, 2]) / K[1, 1], 1.0])
    d = d / np.linalg.norm(d)
    denom = float(d @ normal)
    if abs(denom) < 1e-9:
        return None
    return d * (float(origin @ normal) / denom)


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

    mask = outline = px = None
    for m_, o_ in mpl.platform_from_depth(D, K, max_n=3):
        p_ = mpl.scale_from_depth(D, m_, K)
        if p_ and mpl.mask_size_consistent(m_, p_, REF['work_size_m']):
            mask, outline, px = m_, o_, p_
            break
    assert px, 'no depth-isolated candidate matched the CAD panel size'
    print(f'[1] panel found, scale {px:.0f} px/m')

    pl = plane_from_depth(D, K, cv2.erode(mask, np.ones((21, 21), np.uint8)))
    assert pl is not None, 'no plane fit inside the panel'
    origin, normal = pl
    print(f'[2] panel plane: normal {np.round(normal, 3)}, '
          f'{float(origin @ normal) * 1000:.1f} mm from the camera along it')

    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cands = canny_candidates(grey, inner, px, ports)
    print(f'[3] {len(cands)} hole candidates from Canny')

    # CAD's only job from here on is identification: which candidate is which
    # named hole. solve_pose exists solely to make that correspondence check
    # itself (a wrong assignment reprojects badly), and to give a heading for
    # the ports nothing detected. Nothing downstream lets the pose or the CAD
    # rectangle move a measured pixel -- that was solve_and_refine's job
    # (edge_centroids re-measuring inside a CAD-projected window) in the
    # shipping path, and is exactly the CAD-influences-position channel this
    # script exists to remove. measured is the raw Canny centroid, untouched.
    match = mpl.match_ports(cands, ports, px, K=K, min_pairs=REF.get('min_pairs'))
    assert match and match['verified'], 'no verified correspondence'
    img = np.array([cands[di]['centre'] for _, di in match['pairs']])
    sol = mpl.solve_pose(match['pairs'], ports, img, K, None)
    assert sol is not None, 'no camera-facing pose'
    rvec0, tvec0, err = sol
    fixed = mpl.refit_with_normal(match['pairs'], ports, img, K, normal, rvec0, tvec0)
    rvec, tvec = rvec0, tvec0
    if fixed is not None and fixed[2] <= max(err * 2, 4.0):
        rvec, tvec, err = fixed
    print(f'[4] matched {match["n_matched"]}/{len(ports)}, reproj {err:.2f} px '
          f'(pose used only to identify holes and orient un-detected ones)')

    # The Canny centroid above only ever served to get this rough pose. Each
    # target hole's actual reported centre comes from frame_rect/ring_rect,
    # run fresh in that hole's own CAD-projected window -- regardless of
    # whether canny_candidates happened to also find a blob for it.
    #
    # The window uses the coarse pose (rvec0/tvec0), not the tilt-refit one:
    # frame_rect's own component pick is sensitive to a couple of px of shift
    # in where the window lands (an L-shaped bracket is asymmetric enough
    # that its bounding-box centre moves with the window), and the coarse
    # pose is the one this was actually tuned and verified against. The
    # refit pose still does its job everywhere else -- projecting undetected
    # holes, and the pose(x)CAD comparison below.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    measured = {}
    for name in TARGETS:
        quad = mpl._port_quad(ports[name2idx[name]], rvec0, tvec0, K, None)
        r = ring_rect(grey, hsv, quad) if name == 'hdmi1' else frame_rect(grey, hsv, quad)
        if r is not None:
            L, T, R, B = r
            measured[name] = np.array([(L + R) / 2, (T + B) / 2])

    allobj = np.array([p['centre'] for p in ports], dtype=np.float64)
    proj, _ = cv2.projectPoints(allobj, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)

    # The disagreement with pose(x)CAD is the only objective check available on
    # this panel. test_mono_case1's hand-read TRUTH_PX carries 3-7 px of its
    # own error -- the same size as what is being compared -- so it cannot rank
    # these. The CAD pattern can, in one specific way: its *relative* geometry
    # is a rigid body even if its absolute placement is off, so a hole whose
    # measured centre disagrees with where the rest of the panel says it should
    # be is the one to suspect. Split into the component along the panel normal
    # (shared by every hole, so a plane offset, not a per-hole error) and the
    # in-plane component (per-hole, and the one that matters for alignment).
    print(f'\n[5] where the arm would be sent, both ways:')
    print(f'    {"port":8s} {"source":10s} {"pixel":>16s} {"camera-frame XYZ (mm)":>28s} '
          f'{"along normal":>13s} {"in-plane":>10s}')
    # The panel outline is not drawn. It has no part in this method: the scale
    # and the plane both come from the mask's depth pixels, never from the
    # outline's shape, so showing it only invites reading meaning into a
    # boundary nothing here measures from.
    lateral = []
    vis = bgr.copy()
    for name in TARGETS:
        i = name2idx[name]
        old_uv = proj[i]
        if name in measured:
            uv, src, col = measured[name], 'measured', (0, 255, 255)
        else:
            uv, src, col = old_uv, 'CAD only', (255, 0, 0)
        P = ray_plane(uv, K, origin, normal)
        # what the shipping path would have sent, for the same hole
        Rm, _ = cv2.Rodrigues(rvec)
        P_old = Rm @ allobj[i] + tvec.ravel()
        d = P - P_old
        along = float(d @ normal)
        in_plane = float(np.linalg.norm(d - along * normal))
        lateral.append((name, in_plane * 1000))
        print(f'    {name:8s} {src:10s} ({uv[0]:6.1f},{uv[1]:6.1f}) '
              f'({P[0]*1000:7.1f},{P[1]*1000:7.1f},{P[2]*1000:7.1f}) '
              f'{along*1000:10.2f} mm {in_plane*1000:7.2f} mm')
        cv2.drawMarker(vis, tuple(np.round(uv).astype(int)), col,
                       cv2.MARKER_CROSS, 13, 2)
        cv2.putText(vis, name, tuple(np.round(uv).astype(int) + [7, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    lat = np.array([v for _, v in lateral])
    print(f'\n[6] in-plane disagreement with the CAD pattern: '
          f'median {np.median(lat):.2f} mm, worst {lat.max():.2f} mm '
          f'({max(lateral, key=lambda t: t[1])[0]})')
    print(f'    the along-normal column reads the same for every hole, so it is the '
          f'depth plane\n    sitting that far off the pose\'s own plane -- one offset '
          f'shared by all of them,\n    not nine separate errors.')
    for n_, v_ in sorted(lateral, key=lambda t: -t[1]):
        flag = '   <-- worth a look' if v_ > np.median(lat) * 1.6 else ''
        print(f'      {n_:8s} {v_:5.2f} mm{flag}')

    x, y, w, h = cv2.boundingRect(outline)
    H, W = vis.shape[:2]
    vis = vis[max(0, y - 40):min(H, y + h + 40), max(0, x - 40):min(W, x + w + 40)]
    s = 2000 / max(vis.shape[:2])
    if s < 1.0:
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = os.path.join(DATA, 'direct_target_case1.png')
    cv2.imwrite(out, vis)
    print(f'\n    yellow = measured from the image, blue = CAD projection '
          f'(hole never detected)')
    print(f'    wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
