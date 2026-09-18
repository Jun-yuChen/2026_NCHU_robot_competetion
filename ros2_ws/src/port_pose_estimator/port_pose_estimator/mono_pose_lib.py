"""Pose of a multi-port panel from one greyscale frame, with the CAD as the ruler.

The depth route in depth_pose_lib measures the part and then asks whether the
measurement matches the CAD. That works on the pale RJ45 jig and fails on this
platform, because the platform is matte black: the D405's projected pattern is
absorbed rather than returned, so the silhouette comes back ragged, the top face
reads systematically far, and every length derived from it is wrong by several
percent -- enough that the port pattern no longer matches the CAD at all.

This module inverts the question. The CAD already knows every distance on the
part, so nothing has to be measured in metres. All the image has to supply is
*where* each port is in pixels; the scale, and with it the full pose, comes out
of the perspective solve. That is an ordinary PnP problem with a known planar
target, and it is exactly what the black body is good at: the ports are bright
against it, which is the one thing this part makes easy.

Depth is not used to compute anything here. The caller may pass a depth frame to
plane_normal_from_depth for a second opinion on the panel's tilt, which is the one
quantity a single planar view is genuinely weak at -- but measured against a proper
RANSAC fit the image was already right to 2.4 deg, so it stays a check, not an input.

Measured on a live frame at 285 mm: 8 of 8 ports found, 0.5 mm mean reprojection.
"""
import itertools

import cv2
import numpy as np


# ---------------------------------------------------------------- the part

def platform_candidates(grey, min_area=2000, close_px=9, fill_px=31, max_n=4):
    """Dark blobs that could be the part, best first. -> [(mask, outline), ...].

    Otsu over the whole frame, because black against a pale bench is the
    easiest threshold in the pipeline. Then rank by area x fill ratio rather
    than area alone: cable runs and the shadowed edge of the bench are dark too
    and can carry more pixels than the part while filling a fraction of their
    bounding box, where the part fills about 0.9 of its own.

    Several are returned rather than one, and that matters. On hardware the top
    blob has been the gripper's own black body merged with a cable and a monitor
    bezel, while the real panel sat second because it was half out of frame. No
    ranking over brightness and shape alone separates those reliably -- they are
    all just dark rectangles. What does separate them is whether the CAD's port
    pattern registers inside, so the caller tries these in order and keeps the
    first that does. On a clean frame that is the first one and costs nothing.
    """
    _, dark = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)

    scored = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        box = int(st[i, cv2.CC_STAT_WIDTH]) * int(st[i, cv2.CC_STAT_HEIGHT])
        scored.append((a * (a / box), i))
    scored.sort(reverse=True)

    out = []
    for _, i in scored[:max_n]:
        blob = (lab == i).astype(np.uint8) * 255
        blob = cv2.morphologyEx(
            blob, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_px, fill_px)))
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        outline = max(cnts, key=cv2.contourArea)
        mask = np.zeros_like(grey)
        cv2.drawContours(mask, [outline], -1, 255, -1)   # ports filled back in
        out.append((mask, outline))
    return out


def platform_mask(grey, **kw):
    """The best-scoring dark blob. -> (mask, outline) or (None, None)."""
    got = platform_candidates(grey, max_n=1, **kw)
    return got[0] if got else (None, None)


def platform_from_depth(depth, K, near_margin=0.10, resid_iters=300,
                        resid_tol=0.003, resid_thresh=-0.010,
                        min_area=2000, max_n=1):
    """Depth-isolated dark platform, for a panel whose Otsu blob merges with
    equally dark clutter behind it. -> [(mask, outline), ...].

    platform_candidates cannot separate the part from a chair, a box, or
    shelving that happens to be as dark and touches it in frame -- brightness
    alone does not distinguish them. Depth can: the part sits close to the
    camera and clutter sits further back, so gating on the near cluster of
    the depth histogram, fitting a plane to it, and keeping only the pixels
    recessed below that plane isolates the part's own face even with
    clutter in shot. Measured on five real captures of such a scene, the
    part came out 3x+ closer than everything behind it every time.
    """
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    if valid.sum() < 500:
        return []
    v = depth[valid]
    lo, hi = float(v.min()), float(min(v.min() + 1.5, v.max()))
    h, edges = np.histogram(v, bins=60, range=(lo, hi))
    peak = int(np.argmax(h))
    near_lo = edges[peak]
    near_hi = edges[peak] + (edges[1] - edges[0]) + near_margin
    gate = (valid & (depth >= near_lo - 0.02) & (depth <= near_hi)).astype(np.uint8) * 255
    gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    gate = cv2.morphologyEx(gate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(gate, 8)
    if n < 2:
        return []
    biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    near = lab == biggest

    rng = np.random.default_rng(0)
    vi, ui = np.where(near & np.isfinite(depth) & (depth > 0) & (depth < 2.0))
    if len(vi) < 500:
        return []
    z = depth[vi, ui]
    pts = np.stack([(ui - K[0, 2]) * z / K[0, 0],
                    (vi - K[1, 2]) * z / K[1, 1], z], axis=1)
    sub = pts if len(pts) <= 8000 else pts[rng.choice(len(pts), 8000, replace=False)]
    best_fit = None
    for _ in range(resid_iters):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        hits = int((np.abs((sub - s[0]) @ nrm) < resid_tol).sum())
        if best_fit is None or hits > best_fit[0]:
            best_fit = (hits, nrm, s[0])
    if best_fit is None:
        return []
    _, nrm, o = best_fit
    inl = pts[np.abs((pts - o) @ nrm) < resid_tol * 2]
    c = inl.mean(axis=0)
    _, V = np.linalg.eigh(np.cov((inl - c).T))
    nrm = V[:, 0]
    if nrm[2] > 0:
        nrm = -nrm

    resid = np.full(depth.shape, np.nan)
    resid[vi, ui] = (pts - o) @ nrm
    recessed = ((resid < resid_thresh) & np.isfinite(resid)).astype(np.uint8) * 255
    # Open before labelling. Without it, a one-pixel-wide seam of equally
    # recessed depth -- the panel's bottom edge touching a drive bay's own
    # recessed slots, both below the fitted plane by more than resid_thresh --
    # is enough under 8-connectivity to fuse them into one component, and the
    # scale and outline that follow are then read off the merged shape, not
    # the panel. A real panel is many pixels wide in every direction; a seam
    # like that is not, so opening breaks the seam while leaving the panel
    # itself untouched.
    recessed = cv2.morphologyEx(
        recessed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(recessed, 8)
    if n2 < 2:
        return []
    # Rank by area x fill ratio, not area alone -- same reasoning as
    # platform_candidates. Opening breaks a narrow seam, but a wide one (the
    # panel's bottom edge and a drive bay's slots both genuinely below
    # resid_thresh, at a range where depth noise has widened what counts as
    # "recessed") stays one component and the panel is no longer its biggest
    # piece by shape even where it still is by area: a real panel fills
    # ~0.8-0.85 of its own bounding box, the panel-plus-bay tendril seen at
    # 320 mm filled 0.55.
    def score(i):
        area = st2[i, cv2.CC_STAT_AREA]
        box = st2[i, cv2.CC_STAT_WIDTH] * st2[i, cv2.CC_STAT_HEIGHT]
        return area * (area / box) if box else 0.0
    order = sorted(range(1, n2), key=lambda i: -score(i))

    out = []
    for i in order[:max_n]:
        if st2[i, cv2.CC_STAT_AREA] < min_area:
            continue
        mask = (lab2 == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        outline = max(cnts, key=cv2.contourArea)
        out.append((mask, outline))
    return out


def scale_from_depth(depth, mask, K):
    """px-per-metre from fx / the mask's own median depth. -> float or None.

    silhouette_scale reads the scale off the outline's pixel size against the
    CAD footprint, which assumes the outline is that footprint. A depth-derived
    mask is a segmentation result, not a CAD-sized rectangle -- it can come out
    smaller (a plane-residual cut that excludes the raised frame) or larger
    (background bleeding in) than the true panel, and either one biases
    silhouette_scale directly. A pinhole camera puts a real-world length L at
    range Z across L * fx / Z pixels regardless of what the mask's edges
    happen to include, so reading Z off the depth stream sidesteps the mask's
    own shape entirely.
    """
    z = depth[(mask > 0) & np.isfinite(depth) & (depth > 0)]
    if len(z) < 50:
        return None
    Z = float(np.median(z))
    return None if Z <= 1e-6 else float(K[0, 0]) / Z


def mask_size_consistent(mask, px_per_m, work_size_m, tol=0.4):
    """Sanity-check a depth-isolated mask's own footprint against the CAD's.

    scale_from_depth reads the scale off the mask's median depth alone, not
    its shape, so a mask that is not actually the panel -- a shadow gap, a
    sliver of a recessed drive bay, depth noise near an edge, anything that
    survives platform_from_depth's own scoring on a bad frame -- still yields
    *a* scale, just the wrong one, and every port search downstream inherits
    that contamination silently: candidate blobs get compared against the
    wrong CAD-in-pixels size, a self-consistent but wrong subset can still
    verify, and the resulting pose reprojects fine because reprojection error
    only measures agreement with the (mis-scaled) points it was given, never
    against the real panel. Converting the mask's own bounding box to metres
    with the scale it just produced and checking that against the CAD panel's
    real size catches this before it can poison anything downstream.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or not px_per_m:
        return False
    w = (xs.max() - xs.min()) / px_per_m
    h = (ys.max() - ys.min()) / px_per_m
    obs = sorted([w, h])
    cad = sorted(work_size_m)
    if min(cad) <= 0:
        return False
    return all(1 - tol <= o / c <= 1 + tol for o, c in zip(obs, cad))


def silhouette_scale(outline, work_size_m):
    """Rough px-per-metre from the part's own footprint.

    Deliberately rough. It is a prior for the correspondence search and nothing
    more -- the pose solve derives the true scale from perspective. Measured on
    hardware it came out 5% high, because the silhouette includes the raised
    handle, which stands closer to the camera than the port face; the search
    tolerates a third more than that.
    """
    (_, _), (w, h), _ = cv2.minAreaRect(outline)
    obs, cad = sorted([w, h]), sorted(work_size_m)
    if min(cad) <= 0:
        return None
    return 0.5 * (obs[0] / cad[0] + obs[1] / cad[1])


# --------------------------------------------------------------- the ports

def _adaptive_block(px_per_m, cad_ports, span=6.0, lo=15, hi=151):
    """Neighbourhood for the adaptive threshold, in pixels.

    This has to scale with the working distance and it is not optional. The
    local mean only marks a port as bright if the window around it is mostly
    panel; once the window is comparable to the port, the port's own interior
    drags the mean up and the port stops standing out. The first version fixed
    it at 61 px, which is 6x the port width at the 33 cm the reference frame was
    shot at -- and quietly stopped working when the camera came closer, which is
    exactly when the ports get easier to see. Symptom on hardware: a detection
    that flickered in and out frame to frame.
    """
    short_px = min(p['size'][1] for p in cad_ports) * px_per_m
    return int(np.clip(round(span * short_px) // 2 * 2 + 1, lo, hi))


def port_candidates(grey, mask, px_per_m, cad_ports, block=None, offset=-12,
                    erode_px=9, area_lo=0.20, area_hi=4.0, max_out=40):
    """Bright regions inside the part -> candidate port centres.

    Adaptive rather than global thresholding: these ports are lit by whatever
    happens to reflect off the plastic tongue inside them, so their brightness
    varies by a factor of several across one panel. A global Otsu inside the
    mask found 5 of 8 on the live frame; the local one found 8 of 8.

    Over-detection is the intended behaviour. Handing the matcher a few extra
    blobs costs milliseconds and it discards them; missing a real port costs an
    inlier that cannot be recovered. Only the area band prunes here, and it is
    set wide -- a port whose blob merges with a glare patch can double in size.
    The cap on the count is there because the matcher is quadratic in it, and a
    frame that produces eighty blobs has gone wrong in a way more of them will
    not fix.
    """
    if block is None:
        block = _adaptive_block(px_per_m, cad_ports)
    inner = cv2.erode(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px)))
    bw = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, block, offset)
    bw = cv2.bitwise_and(bw, inner)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    areas = [float(np.prod(p['size'])) * px_per_m ** 2 for p in cad_ports]
    lo, hi = min(areas) * area_lo, max(areas) * area_hi
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if not (lo <= a <= hi):
            continue                      # noise below, the handle slot above
        cnt, _ = cv2.findContours((lab == i).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (w, h), _ = cv2.minAreaRect(cnt[0])
        out.append(dict(centre=np.array(cen[i], dtype=np.float64),
                        area_px=a,
                        long_m=max(w, h) / px_per_m,
                        short_m=min(w, h) / px_per_m))
    if len(out) > max_out:
        mid = float(np.median(areas))
        out.sort(key=lambda c: abs(c['area_px'] - mid))
        out = out[:max_out]
    return out


def find_ports(grey, mask, px_per_m, cad_ports, K=None, dist=None,
               offsets=(-12, -8, -18, -6, -25), scales=(1.0, 0.6, 1.5, 0.35),
               prefer=None, min_candidates=4, **kw):
    """Detect and match together, retrying the threshold. -> (candidates, match).

    One threshold setting is one guess at how much darker than its surroundings
    a port happens to be, and on this panel that varies between sockets, let
    alone between rooms. Rather than tune the guess, try a few and keep whatever
    registers best.

    "Best" means verified: a match that reprojects well beats one that merely
    matched more points but does not (see match_ports). Stopping only on a
    verified result, not just a high count, is what keeps this loop from
    settling for the same wrong-but-full-count hypothesis on every attempt.

    `px_per_m` is retried at several multiples for the same reason, and that one
    is not a refinement. It arrives from the part's silhouette, which is only as
    good as the segmentation: when the panel's dark blob merges with something
    else dark behind it -- a monitor bezel, the gripper's own body -- the
    silhouette is of the merged region and the scale comes out far too high.
    Measured on such a frame it read 5601 px/m against a true 1900, which put
    every port outside the area band and left two candidates. The merged mask
    itself was fine; feeding the same mask a scale anywhere between 1500 and
    3000 matched seven or eight ports at a consistent 350 mm.

    The common case costs one attempt: the loop stops as soon as an attempt
    verifies all but one port. On the frame above it took three, 268 ms in all,
    against 18 ms when the first scale is right -- worth paying on a frame that
    was otherwise producing nothing.
    """
    # Whatever worked last frame is tried first. Both ladders exist for frames
    # that need them, but a scene changes far more slowly than it is sampled, so
    # a frame that needed the third scale and the fourth offset almost certainly
    # still does -- and paying the search again every frame is what turns a
    # recovery path into a permanent 3 Hz. With the winner tried first the
    # merged-silhouette case settles back to roughly the cost of a clean frame.
    order = [(sc, off) for sc in scales for off in offsets]
    if prefer in order:
        order.remove(prefer)
        order.insert(0, prefer)

    best = (([], None), -1, False)
    for scale, offset in order:
        px = px_per_m * scale
        cands = port_candidates(grey, mask, px, cad_ports,
                                offset=offset, **kw)
        if len(cands) < min_candidates:
            if len(cands) > best[1]:
                best = ((cands, None), len(cands), False)
            continue
        match = match_ports(cands, cad_ports, px, K=K, dist=dist)
        verified = bool(match and match.get('verified'))
        score = match['n_matched'] if match else 0
        if match is not None:
            match['tried'] = (scale, offset)
        if (verified, score) > (best[2], best[1]):
            best = ((cands, match), score, verified)
        if verified and score >= len(cad_ports) - 1:
            return best[0]
    return best[0]


def find_ports_ladder(grey, mask, px_per_m, cad_ports, ladder, K=None, dist=None,
                      min_candidates=4, prefer=None, min_pairs=None,
                      max_reproj_px=6.0, **kw):
    """find_ports, but retrying explicit (block, offset) pairs. -> (candidates, match).

    find_ports's ladder retries px_per_m at a handful of multiples, which
    moves the adaptive block size only indirectly and not very far: its
    widest scale (0.35x) still leaves the block several times too large for
    a panel whose ports run close together. Two sockets 1.75 mm apart at
    their nearest edge merged into one oversized blob at every setting that
    ladder reached and never separated; retrying block directly does,
    because it is the thing that actually needs to change.

    find_ports only stops early on len(cad_ports) - 1, which on an 8-port
    table is a common result and keeps most frames to one attempt. On a
    19-port table carrying eleven unlabelled points alongside the eight the
    task targets, 18 is not a bar real captures clear -- validation topped
    out around ten -- so that condition was never true and every frame paid
    for the entire ladder.

    Lowering the bar to min_pairs + 1 fixed that and broke something worse:
    on case109 the very first ladder entry verified with 8 points and
    stopped the search there, and all 8 sat in the bottom third of the
    panel. Reprojection error on those 8 was fine -- reprojection checks
    internal agreement, not where the points are -- and the pose was 18 deg
    off the depth plane's own normal, because 8 points packed into a third
    of the panel barely constrain the two-thirds they are not in. Widening
    the condition to also require the matched points to span half the
    panel's own extent did not fix it either: the entry that tripped this
    both cleared the count and spanned 86 of 140 mm, and was still the same
    18 deg off. A span that wide can still be two clusters at its ends with
    nothing between them, or near-collinear, and either ill-conditions a
    PnP solve the same way a small span does. Distinguishing "wide" from
    "well-conditioned" needs something closer to the fit's own condition
    number, not a derived geometric proxy for it -- and that has not been
    worked out yet, so the bar stays at the one value actually measured
    safe. Revisit this only with a way to score conditioning directly, not
    with another proxy tried once and trusted.
    """
    # Tuples, not whatever the reference happened to store. The ladder comes
    # out of JSON as lists and `tried` is recorded as a tuple, so the plain
    # `prefer in order` this used to do was comparing (13, -12) against
    # [13, -12] and never matched -- the preference silently did nothing from
    # the day it was written. It matters more here than the cost argument
    # alone suggests: which rung wins is not stable frame to frame, and the
    # rungs are not equally good. Replaying one real frame 34 times with only
    # sensor noise added, six different rungs won, and the pose they produced
    # ranged from 5.4 px of mean target error to 7.6 px. Starting from the
    # rung that worked last time is what keeps a stationary panel on one
    # answer instead of resampling that spread every frame.
    order = [tuple(e) for e in ladder]
    prefer = tuple(prefer) if prefer is not None else None
    if prefer in order:
        order.remove(prefer)
        order.insert(0, prefer)
    good_enough = (len(cad_ports) - 1 if min_pairs is None
                  else min(len(cad_ports) - 1, min_pairs + 3))

    best = (([], None), -1, False)
    for block, offset in order:
        cands = port_candidates(grey, mask, px_per_m, cad_ports,
                                block=block, offset=offset, **kw)
        if len(cands) < min_candidates:
            if len(cands) > best[1]:
                best = ((cands, None), len(cands), False)
            continue
        match = match_ports(cands, cad_ports, px_per_m, K=K, dist=dist,
                            min_pairs=min_pairs, max_reproj_px=max_reproj_px)
        verified = bool(match and match.get('verified'))
        score = match['n_matched'] if match else 0
        if match is not None:
            match['tried'] = (block, offset)
        if (verified, score) > (best[2], best[1]):
            best = ((cands, match), score, verified)
        if verified and score >= good_enough:
            return best[0]
    return best[0]


def match_or_none(grey, mask, px_per_m, cad_ports, K, dist=None):
    """find_ports, for callers that only want the result. -> (candidates, match)."""
    return find_ports(grey, mask, px_per_m, cad_ports, K=K, dist=dist)


def kind_of(short_m, cad_ports, margin=0.002):
    """Which port type a blob's short side is consistent with, or None.

    Only the short side separates these: USB is 5.2 mm across and HDMI 4.8, which
    no camera at this range will tell apart, but RJ45 is 9.5 and is unmistakable.
    That one distinction is worth having -- see match_ports for what it guards.
    """
    kinds = {}
    for p in cad_ports:
        kinds.setdefault(p['kind'], p['size'][1])
    hits = [k for k, s in kinds.items() if abs(short_m - s) <= margin]
    return hits or None


# ------------------------------------------------------------ the matching

def _similarity(cad_a, cad_b, det_i, det_j):
    """The mirrored similarity carrying cad a->b onto det i->j, or None.

    Mirrored, always. The part lies with its port face toward the camera, so the
    object's +z points back along the line of sight, and a right-handed object
    frame therefore projects into the image left-handed. Searching the
    unmirrored family as well is not a harmless generalisation: this platform's
    port *positions* are symmetric about its x axis to within 1.3 mm -- only the
    port *types* break it -- so an unmirrored hypothesis matches all eight
    centres just as well and lands the arm on the wrong column. It did, on
    hardware, before this constraint went in.
    """
    v = det_j - det_i
    nv = float(np.linalg.norm(v))
    u = (cad_b - cad_a) * [1.0, -1.0]                # the mirror
    nu = float(np.linalg.norm(u))
    if nv < 1e-9 or nu < 1e-9:
        return None, None
    ct = float(u @ v) / (nu * nv)
    st = float(u[0] * v[1] - u[1] * v[0]) / (nu * nv)
    s = nv / nu
    A = s * np.array([[ct, -st], [st, ct]]) @ np.diag([1.0, -1.0])
    return A, s


def match_ports(candidates, cad_ports, px_hint, K=None, dist=None,
                scale_tol=0.35, tol_frac=0.30, min_pairs=None, use_kind=True,
                max_reproj_px=6.0, max_checks=8):
    """Decide which blob is which port. -> dict or None.

    Correspondence is unknown, so hypothesise it: any two blobs paired with any
    two CAD ports fix a similarity outright, and the rest of the CAD either
    lands on blobs or it does not. That alone is not enough to pick a winner --
    see the note on verification below for why.

    An earlier version seeded from the centroid and spread of the whole detected
    set instead. That is cheaper and quite wrong: both statistics move when a
    port is missed, so one dropped blob shifted every projection and the match
    collapsed. Measured over subsets, it recovered the right answer for 6 of 36
    seven-port cases. Seeding from pairs does not care what else was detected.
    """
    if len(candidates) < 2 or len(cad_ports) < 2:
        return None
    # Refusing beats guessing. With the bar at four -- the fewest points a PnP
    # can use -- a badly lit frame can strand the matcher on a handful of ports
    # that happen to fit a rotated pose, and it then publishes that pose with
    # every appearance of confidence. Measured against known ground truth while
    # withholding detections, three quarters of the CAD table is the point where
    # the wrong hypotheses run out of inliers before the right one does: at that
    # bar every withheld-port case either came back correct or returned nothing,
    # where at four two of them came back confidently wrong.
    if min_pairs is None:
        min_pairs = max(4, int(round(0.75 * len(cad_ports))))
    det = np.array([c['centre'] for c in candidates])
    cad = np.array([p['centre'][:2] for p in cad_ports])
    spacing = min(np.linalg.norm(cad[a] - cad[b])
                  for a in range(len(cad)) for b in range(a + 1, len(cad)))
    tol = tol_frac * spacing * px_hint

    allowed = None
    if use_kind:
        allowed = [kind_of(c['short_m'], cad_ports) for c in candidates]

    # Only the two seed correspondences are held to the full type check, never
    # the inliers. A blob that has merged with a glare patch measures far too
    # wide -- one HDMI came out 9.4 mm across, which reads as an RJ45 -- and
    # holding every assignment to its measured type threw that port out of a
    # hypothesis the rest of the panel had already settled. As a filter on
    # seeds it still prunes most of the search and costs nothing when it is
    # wrong.
    def compatible(ci, di):
        return allowed is None or allowed[di] is None \
            or cad_ports[ci]['kind'] in allowed[di]

    # A blob's measured width is NOT used to constrain assignments, and the
    # attempt to do so is worth recording. The half-turn ambiguity below is only
    # contradicted by port type, and RJ45 (9.5 mm across) looks well separated
    # from USB/HDMI (5.2/4.8) -- on the reference frame RJ45 blobs measured
    # 10.7-12.0 mm and the rest 5.8-6.7. So assignments were held to that
    # boundary, and it did fix the rotated match.
    #
    # It also broke detection outright at another camera angle. What the
    # threshold segments is not the opening but whatever is bright, and when the
    # light catches a socket's metal shell the blob spans the shell instead:
    # measured on a live frame, short sides ran 0.7 to 16.8 mm for ports that are
    # 4.8 to 9.5 mm, and six to seven of eight blobs landed on the RJ45 side of a
    # boundary only two belong on. Every correct assignment was then refused and
    # the panel stopped registering at all.
    #
    # The lesson is that this measurement is a weak cue, not a constraint -- it
    # is fine for pruning seed hypotheses, where being wrong only costs a search
    # branch, and unfit for rejecting inliers, where being wrong costs the frame.
    # The half-turn case is instead handled by requiring most of the table to
    # match (see min_pairs above), which refuses rather than guesses.

    # Every *distinct* hypothesis that reaches min_pairs inliers, not just the
    # single one with the most of them. On hardware, one port (hdmi1, dead
    # centre of the panel) sometimes fails to detect cleanly, and a stray blob
    # -- a glare fleck, a shadow -- occasionally sits close enough to some
    # rotation of the pattern to fill its slot within tolerance. That produces
    # an 8-point hypothesis whose *count* beats the genuine 7-point one while
    # its fit is visibly worse: on live frames this happened in 19 of 20
    # consecutive frames, each time landing a pose 130+ px from where the image
    # actually shows the panel. A plain rigid 2D fit cannot tell these apart --
    # under this camera's ~24 deg tilt even the correct correspondence only
    # reaches 5-7 px average residual, so a wrong one at 6-8 px looks the same
    # to it. What does tell them apart is whether the resulting 3D pose
    # reprojects convincingly, which the loop below checks directly.
    found = {}
    for i, j in itertools.combinations(range(len(det)), 2):
        for a, b in itertools.permutations(range(len(cad)), 2):
            if not (compatible(a, i) and compatible(b, j)):
                continue
            A, s = _similarity(cad[a], cad[b], det[i], det[j])
            if A is None or abs(s / px_hint - 1.0) > scale_tol:
                continue
            proj = (cad - cad[a]) @ A.T + det[i]
            d = np.linalg.norm(proj[:, None, :] - det[None, :, :], axis=2)
            taken, pairs, total = set(), [], 0.0
            order = sorted(((c, int(d[c].argmin())) for c in range(len(cad))
                            if d[c].min() < tol), key=lambda p: d[p[0], p[1]])
            for ci, di in order:
                if di in taken:
                    continue
                taken.add(di)
                pairs.append((ci, di))
                total += float(d[ci, di])
            if len(pairs) < min_pairs:
                continue
            key = frozenset(pairs)
            prev = found.get(key)
            if prev is None or total < prev[0]:
                found[key] = (total, s)

    if not found:
        return None

    ranked = sorted(found.items(), key=lambda kv: (-len(kv[0]), kv[1][0]))

    # Verify by reprojection when the intrinsics are available. This is what
    # actually breaks the tie the 2D fit cannot: the wrong 8-point hypothesis
    # above reprojects at 130+ px (it is not a plane under any pose), the
    # correct 7-point one at ~2 px. First hypothesis to pass wins outright, so
    # the common case -- the top-ranked one is already right -- costs one extra
    # PnP solve, a fraction of a millisecond.
    if K is not None:
        for pairs, (total, s) in ranked[:max_checks]:
            pairs = list(pairs)
            sol = solve_pose(pairs, cad_ports,
                             np.array([det[di] for _, di in pairs]), K, dist)
            if sol is not None and sol[2] <= max_reproj_px:
                return dict(pairs=pairs, scale_px_per_m=s,
                           n_matched=len(pairs), n_cad=len(cad),
                           match_error_px=total / max(len(pairs), 1),
                           reproj_px=sol[2], verified=True)

    # Nothing passed verification (or K was not given) -- fall back to the
    # plain 2D ranking, same as before this function checked anything in 3D.
    pairs, (total, s) = ranked[0]
    pairs = list(pairs)
    return dict(pairs=pairs, scale_px_per_m=s, n_matched=len(pairs),
                n_cad=len(cad), match_error_px=total / max(len(pairs), 1),
                verified=False)


# ------------------------------------------------------------- the geometry

def _object_points(pairs, cad_ports):
    """CAD port centres, in the mesh frame the rest of the stack expects.

    Keeping the z of the port face (rather than flattening to z=0) is what puts
    the returned origin at the mesh's bounding-box centre, which is the frame
    arm_cmd adds each port's `centre` to. Flatten it and every port comes out
    one panel-thickness off.
    """
    return np.ascontiguousarray(
        [cad_ports[ci]['centre'] for ci, _ in pairs], dtype=np.float64)


def solve_pose(pairs, cad_ports, image_points, K, dist=None):
    """Planar PnP over the matched ports. -> (rvec, tvec, residual_px) or None.

    SQPNP, not IPPE, and the reason is worth recording. IPPE is the textbook
    choice here -- it is closed-form for coplanar points and returns both of the
    poses a plane admits rather than silently converging into one. It is also,
    in OpenCV 4.5.4, sensitive to the order the points are passed in. Measured
    on one verified-correct seven-port correspondence, feeding the same points
    in six different orders gave 1.72 px twice and 129.44 px four times, the bad
    runs also reporting the panel as facing away.

    That is what the flicker on hardware actually was. The correspondence search
    hands its pairs over in whatever order they came out of a set, so the
    correct hypothesis was being scored as garbage on most frames and discarded
    by the facing check, leaving a wrong hypothesis to win by default. Every
    earlier explanation -- the tilt disagreement, the half-turn matches -- was a
    symptom of this.

    SQPNP is globally optimal for the reprojection cost and gave 2.01 px on
    every ordering tried. It returns a single pose rather than the pair, which
    costs nothing here: the choice between them was always made by reprojection
    anyway, and the tilt ambiguity that remains is settled against depth by the
    caller. The facing check stays as a sanity filter -- a pose with the port
    face turned away cannot be the one we are looking at.
    """
    if len(pairs) < 4:
        return None
    dist = np.zeros(5) if dist is None else dist
    obj = _object_points(pairs, cad_ports)
    img = np.ascontiguousarray(image_points, dtype=np.float64)
    try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist,
                                                  flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok or not len(rvecs):
        return None

    scored = []
    for rvec, tvec in zip(rvecs, tvecs):
        rp, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.linalg.norm(rp.reshape(-1, 2) - img, axis=1).mean())
        facing = float(cv2.Rodrigues(rvec)[0][2, 2]) < 0
        scored.append((not facing, err, rvec, tvec))
    scored.sort(key=lambda s: (s[0], s[1]))
    away, err, rvec, tvec = scored[0]
    if away:
        return None                      # every solution has the panel face down
    return rvec, tvec, err


def refine_centroids(grey, pairs, cad_ports, rvec, tvec, K, dist=None, pad=6):
    """Re-measure each port's centre inside its own projected footprint.

    The first pass takes whatever the adaptive threshold produced, and on a port
    that is catching a specular highlight that blob is only part of the opening
    -- one HDMI came out 3.6 mm off its true centre this way, which alone
    doubled the reprojection error of the whole fit.

    Once a coarse pose exists, each port's outline can be projected and the
    measurement redone in a window a few pixels bigger than the port. Otsu is
    well conditioned there in a way it never is over the panel, because the
    window contains one port and its immediate surround and nothing else.
    Measured: 2.25 px mean over the panel, 1.04 px after one pass, converged.
    """
    dist = np.zeros(5) if dist is None else dist
    H, W = grey.shape[:2]
    out = []
    for ci, _ in pairs:
        p = cad_ports[ci]
        c = np.array(p['centre'][:2], dtype=np.float64)
        along = np.array(p['long_axis'][:2], dtype=np.float64)
        across = np.array([-along[1], along[0]])
        L, S = p['size']
        corners = np.array(
            [[*(c + su * along * L / 2 + sv * across * S / 2), p['centre'][2]]
             for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))], dtype=np.float64)
        q, _ = cv2.projectPoints(corners, rvec, tvec, K, dist)
        q = q.reshape(-1, 2)
        x0, y0 = np.floor(q.min(axis=0) - pad).astype(int)
        x1, y1 = np.ceil(q.max(axis=0) + pad).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        win = grey[y0:y1, x0:x1]
        if win.size < 40:
            out.append(None)
            continue
        _, bw = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
        if n < 2:
            out.append(None)
            continue
        b = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        out.append(np.array([cen[b][0] + x0, cen[b][1] + y0]))
    return out


def edge_centroids(grey, pairs, cad_ports, rvec, tvec, K, dist=None, pad=18):
    """Re-measure each port from its own silver-to-black boundary, not its fill.

    refine_centroids re-measures by Otsu, which takes a side on brightness --
    the tongue, or whatever inside the port happens to reflect -- and inherits
    every asymmetry that brightness has, the same problem the panel-wide
    detection has, just in a smaller window. A port on this shield has a
    lighter metal surround against the dark body and a darker interior
    against that surround; the edge between them is a real boundary, not a
    brightness call, and Canny finds it directly instead of thresholding
    either side. Measured over nine ports on a real capture: mean error 7.1 px
    with the fill-based measurement, 5.9 px against this one, with 7 of the
    9 individually better and none worse by more than 1.4 px -- worth trying
    as a genuine alternative, not assuming it wins.

    A wide pad matters here specifically: the boundary this looks for sits at
    the port's own edge, so a window barely bigger than the port can clip it
    before the strongest contour closes.
    """
    dist = np.zeros(5) if dist is None else dist
    H, W = grey.shape[:2]
    out = []
    for ci, _ in pairs:
        p = cad_ports[ci]
        c = np.array(p['centre'][:2], dtype=np.float64)
        along = np.array(p['long_axis'][:2], dtype=np.float64)
        across = np.array([-along[1], along[0]])
        L, S = p['size']
        corners = np.array(
            [[*(c + su * along * L / 2 + sv * across * S / 2), p['centre'][2]]
             for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))], dtype=np.float64)
        q, _ = cv2.projectPoints(corners, rvec, tvec, K, dist)
        q = q.reshape(-1, 2)
        x0, y0 = np.floor(q.min(axis=0) - pad).astype(int)
        x1, y1 = np.ceil(q.max(axis=0) + pad).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        win = grey[y0:y1, x0:x1]
        if win.size < 40:
            out.append(None)
            continue
        edges = cv2.Canny(cv2.GaussianBlur(win, (3, 3), 0), 25, 70)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            out.append(None)
            continue
        bx, by, bw_, bh_ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        out.append(np.array([bx + bw_ / 2 + x0, by + bh_ / 2 + y0]))
    return out


def solve_and_refine(pairs, cad_ports, image_points, grey, K, dist=None,
                     candidates=None, size_ratio_max=1.8,
                     err_ratio_max=2.5, err_abs_min=4.0):
    """solve_pose, then a refine pass, a size check and an error check, each
    kept only if it reprojects better. -> (rvec, tvec, err, image_points, pairs).

    refine_centroids re-measures each port in a window a few pixels bigger
    than the port itself, which is well conditioned when neighbouring ports
    are far enough away that the window contains only one of them --
    server1's spacing (measured: 2.25 px mean reprojection over the panel,
    1.04 px after refine). On a panel with ports closer together than that
    window is wide, "well conditioned" stops holding: Otsu inside the window
    picks whatever is brightest, which can be a neighbour's edge rather than
    this port's own tongue. Measured on such a panel, refine moved the mean
    reprojection error from 3.16 px to 11.11 px -- worse on every point, not
    just the crowded ones.

    match_ports deliberately never rejects an inlier for measuring the wrong
    size -- a blob merged with a glare patch is still the right port, just
    measured wrong, and earlier versions that filtered on size threw those
    out along with genuine mismatches. That is the right call for whether to
    accept the correspondence. It is not the right call for whether to trust
    that blob's own centroid: at 320 mm real captures put candidates 2-5x
    their assigned port's own CAD area often enough that it is not one bad
    port, it is routine at this range, and a centroid pulled across a fused
    blob is not the port's centre. So here, after the fact, a second pass
    tries dropping whichever inliers measured furthest from their own port's
    size and re-solving -- not to decide the correspondence, only to decide
    whether that point's own position is worth trusting in the pose that
    uses it.

    A size mismatch is a cause worth naming, but not the only way a point ends
    up wrong, and the fact of being wrong is visible directly once any pose
    exists to check it against: that point's own reprojection error stands
    out from the rest. So after the size check, whatever is currently best is
    used to measure every point's own residual, and anything well above the
    others -- more than err_ratio_max times the median, floored at
    err_abs_min px so a tight, low-error frame does not flag ordinary noise
    -- goes through the same drop-and-compare pass the size check does. This
    is the direct version of what size mismatch was only a proxy for.

    Returns pairs alongside the pose because this function can drop points
    the caller's own `pairs` still lists; anything downstream that walks
    pairs against image_points (the tilt refit, the debug draw) has to walk
    the returned ones together, not the ones it started with.
    """
    def try_drop(cur_pairs, cur_img, drop, best):
        keep = [k for k in range(len(cur_pairs)) if k not in drop]
        if len(keep) < 4:
            return best
        sol_d = solve_pose([cur_pairs[k] for k in keep], cad_ports,
                           cur_img[keep], K, dist)
        if sol_d is not None and sol_d[2] < best[2]:
            return (*sol_d, cur_img[keep], [cur_pairs[k] for k in keep])
        return best

    # All flagged at once, then each alone: the first is cheap and enough
    # when one bad measurement is dragging the rest along -- the common
    # case, since a merged blob needs a specific nearby feature to merge
    # with. Dropping all of them together can overcorrect on a frame where
    # several are flagged at once (fewer points left, worse spread), so each
    # is also tried alone: whichever single point is actually the one
    # distorting the fit shows up as a real improvement on its own even
    # where the everything-at-once version does not.
    def try_drop_each(cur_pairs, cur_img, flagged, best):
        if not flagged:
            return best
        best = try_drop(cur_pairs, cur_img, set(flagged), best)
        for k in flagged:
            best = try_drop(cur_pairs, cur_img, {k}, best)
        return best

    sol = solve_pose(pairs, cad_ports, image_points, K, dist)
    if sol is None:
        return None
    best = (*sol, image_points, pairs)

    # edge_centroids is the primary refinement, applied unconditionally
    # rather than kept only if it reprojects better. Measured against the
    # panel's real physical port centres (not just internal agreement) on a
    # nine-port capture, it beat the Otsu-based refine_centroids on 7 of 9
    # individually and was never worse by more than 1.4 px -- but its own
    # *reprojection* error came out higher (5.27 px against 4.00), because
    # reprojection error measures how well a set of points agrees with
    # itself, not with the panel, and a wrong-but-mutually-consistent set
    # can score better on it than a right-but-slightly-less-tidy one. Gating
    # on reprojection here would silently throw out the more accurate
    # answer for the more self-consistent one -- comparing solve_and_refine's
    # own name against its actual selection rule made that the wrong
    # default. Where edge_centroids finds nothing for a specific port,
    # Otsu's refine_centroids is tried for that port alone before falling
    # back to the raw candidate.
    fixed = refine_centroids(grey, pairs, cad_ports, sol[0], sol[1], K, dist)
    fixed_e = edge_centroids(grey, pairs, cad_ports, sol[0], sol[1], K, dist)
    img2e = np.array([
        e if e is not None else (o if o is not None else image_points[i])
        for i, (e, o) in enumerate(zip(fixed_e, fixed))])
    sol2e = solve_pose(pairs, cad_ports, img2e, K, dist)
    if sol2e is not None:
        best = (*sol2e, img2e, pairs)

    if candidates is not None and len(pairs) > 4:
        def sized_ok(ci, di):
            got = candidates[di]['long_m'] * candidates[di]['short_m']
            want = cad_ports[ci]['size'][0] * cad_ports[ci]['size'][1]
            return want > 0 and got / want <= size_ratio_max
        flagged = [k for k, (ci, di) in enumerate(pairs) if not sized_ok(ci, di)]
        best = try_drop_each(pairs, image_points, flagged, best)

    rvec_b, tvec_b, _, img_b, pairs_b = best
    if len(pairs_b) > 4:
        objp = np.array([cad_ports[ci]['centre'] for ci, _ in pairs_b])
        rp, _ = cv2.projectPoints(objp, rvec_b, tvec_b, K, dist)
        per_err = np.linalg.norm(rp.reshape(-1, 2) - img_b, axis=1)
        bar = max(err_ratio_max * float(np.median(per_err)), err_abs_min)
        flagged = [k for k, e in enumerate(per_err) if e > bar]
        best = try_drop_each(pairs_b, img_b, flagged, best)

    return best


def plane_normal_from_depth(depth, K, mask, tol=0.0015, iters=200, seed=0,
                            max_samples=6000):
    """A cross-check on the pose's tilt, from the depth stream. -> unit normal or None.

    Nothing in the pose uses this. It exists because tilt is the weak axis of any
    planar PnP -- rotating the panel about an in-plane axis moves its points
    mostly along the line of sight, which the image barely registers -- so it is
    the one number worth having a second opinion on.

    RANSAC rather than a plain least-squares fit, and that distinction is the
    whole point of this function. Fitting every depth pixel inside the platform
    mask gives a surface flat to 7 mm and a normal 15 deg from the pose's, which
    looks like the pose being badly wrong; it is the raised handle standing proud
    of the port face and dragging the fit. Rejecting it leaves 25000 points flat
    to 0.4 mm, and that normal agrees with the monocular pose to 2.4 deg.

    So the answer this returns is "the image was right", which is why the depth
    stream is now only ever consulted, never believed.
    """
    v, u = np.where((mask > 0) & np.isfinite(depth))
    if len(v) < 500:
        return None
    z = depth[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0],
                    (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= max_samples else pts[rng.choice(len(pts), max_samples,
                                                             replace=False)]
    best = None
    for _ in range(iters):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        hits = int((np.abs((sub - s[0]) @ n) < tol).sum())
        if best is None or hits > best[0]:
            best = (hits, n, s[0])
    if best is None:
        return None
    _, n, o = best
    inl = pts[np.abs((pts - o) @ n) < tol]
    if len(inl) < 200:
        return None
    c = inl.mean(axis=0)
    _, V = np.linalg.eigh(np.cov((inl - c).T))
    n = V[:, 0]
    return -n if n[2] > 0 else n


def tilt_degrees(rvec, normal):
    """Angle between the pose's panel normal and an independently measured one."""
    z = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0][:, 2]
    return float(np.degrees(np.arccos(np.clip(abs(z @ np.asarray(normal)), -1, 1))))


def refit_with_normal(pairs, cad_ports, image_points, K, normal, rvec, tvec,
                      dist=None):
    """Re-solve with the panel's tilt fixed. -> (rvec, tvec, residual_px) or None.

    A coplanar PnP admits two solutions related by a reflection, and nothing in
    the image alone tells them apart when the view is close to fronto-parallel
    -- see match_ports and the node's tilt cross-check for what that looks like
    in practice (the wrong twin, not noise). Depth does not have this ambiguity:
    a RANSAC fit over tens of thousands of points has exactly one normal. Fixing
    the pose's z-axis to that normal and re-solving for only yaw and translation
    removes the ambiguity structurally, rather than trying to pick the right
    twin after the fact.

    `normal` is in camera coordinates, pointing at the camera. `rvec`/`tvec` seed
    the search (their z-axis is discarded, only the yaw about the new z and the
    translation carry over) -- the seed only affects which of any local optima
    least_squares lands in, and fixing z removes the one degeneracy that would
    have given it two very different optima to choose between.
    """
    from scipy.optimize import least_squares

    dist = np.zeros(5) if dist is None else dist
    obj = _object_points(pairs, cad_ports)
    img = np.ascontiguousarray(image_points, dtype=np.float64)

    z = np.asarray(normal, dtype=np.float64)
    z = z / np.linalg.norm(z)
    if z[2] > 0:
        z = -z                            # object +z looks back at the camera
    # any frame with that z will do; the free yaw below spans the rest
    seed = np.array([1.0, 0.0, 0.0])
    if abs(z @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    x0 = np.cross(seed, z)
    x0 /= np.linalg.norm(x0)
    B = np.column_stack([x0, np.cross(z, x0), z])

    R0 = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    yaw0 = float(np.arctan2((B.T @ R0)[1, 0], (B.T @ R0)[0, 0]))

    def pose(p):
        c, s = np.cos(p[0]), np.sin(p[0])
        R = B @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return cv2.Rodrigues(R)[0], p[1:4].reshape(3, 1)

    def resid(p):
        rv, tv = pose(p)
        rp, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        return (rp.reshape(-1, 2) - img).ravel()

    p0 = np.concatenate([[yaw0], np.asarray(tvec, dtype=np.float64).ravel()])
    try:
        sol = least_squares(resid, p0, method='lm', max_nfev=200)
    except Exception:
        return None
    rv, tv = pose(sol.x)
    err = float(np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1).mean())
    return rv, tv, err


# -------------------------------------------- image-only hole centre (純影像處理找孔中心法)
#
# frame_rect/ring_rect find a named hole's own centre from its own pixels --
# its silver frame against its dark interior -- with no CAD position in the
# loop at all. CAD still says which hole is which (match_ports/solve_pose,
# above); it no longer supplies a fallback position for a hole this frame
# failed to measure either -- see depth_pose_node's per-hole correction,
# which now refuses to publish for its own target port rather than fall
# back to pose (x) CAD. plane_from_depth and ray_plane turn a measured pixel
# into the 3D point a plug actually has to meet, using the depth stream only
# for the panel's own face plane, never for the hole itself (a hole is a
# recess, so its own depth reads "behind the face", not "where the opening
# is"). See tools/test_direct_target.py for the repeatability measurement
# this grew out of.

def find_outer_quad(grey, mask, px_per_m, work_size_m, lo=40, hi=120,
                    close=15, tol=0.25):
    """The panel's own outer rectangle in the image, from Canny. -> (4,2) or None.

    Ported from the standalone experiment in tools/test_outer_frame.py, which
    tried using this quad for the *pose* (PnP against the CAD's outer
    rectangle) and was abandoned for that -- a single big rectangle is a
    worse-conditioned coplanar target for tilt than many scattered ports, and
    a plain rectangle cannot resolve its own 4-fold corner correspondence
    without the extra machinery solve_outer_pose there needed. Reused here
    for a narrower job that plays to its actual strength: this rectangle is
    the most reliable large, unambiguous thing in the frame to read an
    *orientation* off, and orientation from a rectangle needs only its own
    long/short edges, not a corner correspondence -- see panel_axes_pose.

    Searched only within a generous margin around the depth mask, not the
    whole frame -- otherwise the case's own outer edge, a monitor behind it,
    anything else rectangular in shot is fair game for "biggest rectangle" to
    land on. Candidates are scored against the CAD's own aspect ratio and the
    depth-given scale's predicted size, the same two checks
    mask_size_consistent already uses for the depth mask itself, so a contour
    has to agree with both the shape and the range this panel is actually
    known to be at, not just look rectangular.
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


def panel_axes_pose(quad, K, origin, normal):
    """The panel's own coordinate frame, from its outer rectangle alone. -> rvec or None.

    Y is the rectangle's own long side, oriented toward smaller image row
    (screen "up" is +Y); X is the short side, oriented toward larger image
    column (screen "right" is +X); Z is the depth-measured face normal
    (plane_from_depth). None of the three comes from PnP or a CAD rotation --
    `quad` is find_outer_quad's own 4 corners, read straight off the image.

    Each axis is still actually measured in 3D via ray_plane, the same way a
    hole's own pixel is, rather than assumed from the image angles alone: a
    rectangle photographed off-square has its own on-screen angles distorted
    by perspective (its two pairs of sides are not drawn perpendicular in
    pixels even though they are on the real panel), and only intersecting
    each edge's own direction with the depth plane corrects for that.
    """
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    lens = [np.linalg.norm(b - a) for a, b in edges]
    long_i = 0 if (lens[0] + lens[2]) >= (lens[1] + lens[3]) else 1
    short_i = (long_i + 1) % 4

    def axis_3d(edge, want):
        # 7 points along the edge (not just the 2 endpoints of a +-0.3
        # span), each its own ray_plane hit, direction from an SVD line fit
        # rather than one point-to-point vector -- local Canny noise on any
        # single sample gets averaged out instead of going straight into
        # the angle, the same reasoning plane_from_depth's own many-point
        # fit is built on. +-0.4 (not all the way to the corners) because a
        # rectangle's corner is exactly where perspective and the other
        # edge's own line both distort this edge's apparent path most.
        a, b = edge
        d2 = b - a
        if not want(d2):
            d2 = -d2
        mid = (a + b) / 2
        pts = [ray_plane(mid + d2 * t, K, origin, normal)
              for t in np.linspace(-0.4, 0.4, 7)]
        pts = np.array([p for p in pts if p is not None])
        if len(pts) < 2:
            return None
        centred = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred)
        v = Vt[0]
        if v @ (pts[-1] - pts[0]) < 0:   # SVD's sign is arbitrary; match
            v = -v                       # the direction t itself increases in
        nv = np.linalg.norm(v)
        return None if nv < 1e-9 else v / nv

    y = axis_3d(edges[long_i], lambda d: d[1] < 0)
    x = axis_3d(edges[short_i], lambda d: d[0] > 0)
    if y is None or x is None:
        return None
    z = np.asarray(normal, dtype=np.float64)
    z = z / np.linalg.norm(z)
    y = y - (y @ z) * z
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        return None
    y = y / ny
    if np.cross(y, z) @ x < 0:      # keep the frame right-handed and matching X
        z = -z
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    return cv2.Rodrigues(np.column_stack([x, y, z]))[0]


def canny_candidates(grey, inner, px_per_m, cad_ports, lo=40, hi=110, close=3):
    """Hole centres from their own edges. -> [{centre, area_px, long_m, short_m}].

    port_candidates (the production identification route, above) thresholds
    for brightness, which needs the hole to be lighter than the panel around
    it -- and over 30 noise trials on one frame it never once found usb1 and
    found usb2 in only 10 of 30, because the USB 2.0 pair's dark tongue and
    dark surround produce no bright blob at any block/offset the ladder
    tries. What every socket does have is the boundary between its metal
    surround and its own dark interior, and Canny takes that directly --
    same trials, found all nine every time.

    This is only ever used to get a rough first pose, purely to place each
    named hole's own frame_rect/ring_rect search window accurately -- a
    window placed from a pose that never saw usb1/usb2 directly (production
    identification routinely doesn't) is off by just enough to flip which
    metal fragment frame_rect picks for them. It is never used for a hole's
    actual position; see frame_rect below for that.
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


def frame_rect(grey, hsv, quad, pad=12, sat_max=90, edge_tol=1, close_k=5):
    """A hole's own silver-frame rectangle, measured, not projected.

    -> (left, top, right, bottom) in full-image pixels, or None.

    The frame is whichever bit of this window is bright (Otsu on V) and grey
    (S below sat_max -- the blue USB3 tongue passes the brightness side of
    this and has to be excluded on colour instead). Among the surviving
    blobs, the one closest to the CAD quad's own centre is this hole's own
    frame; anything touching the crop's edge is a neighbour's frame caught by
    the padding, not this hole's -- pad needs enough margin that the real
    bracket does not itself brush that edge on an ordinary frame: usb7's own
    three-sided bracket already fills most of an 8px-padded window, and at
    that pad a few px of pose noise was enough to push its own bottom edge
    into the exclusion and hand the read to a smaller, wrong blob instead
    (confirmed on a real capture, not assumed); 12px stopped that.

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
    photo for usb1/usb2/usb7/usb3/usb4/rj451 (hdmi1 is bevel-cornered and uses
    ring_rect instead; usb5/usb6 also close in the picture from a fourth,
    separate frame fragment on their far side, but every attempt to fold that
    fragment in automatically ended up grabbing the *neighbouring* port's own
    fragment instead on at least one hole, so it is deliberately left out
    here -- the L-only reading it falls back to was the one actually
    confirmed correct).

    Sensor-level noise can perforate the true bracket's own arm with a few
    dropout pixels exactly at the sampled centre row/column, which the
    2-runs branch above then misreads as a real double-sided edge with black
    between -- producing a tiny, badly-placed rectangle from an otherwise
    correctly-selected blob (confirmed on real usb5/6 noise trials: this is
    what actually broke them, not a wrong blob being picked). Closing gaps
    with a corner-sized kernel fixes this cheaply because best is already
    picked by this point: closing the isolated best mask alone, after
    selection, cannot bridge into the blue tongue or any other blob the way
    closing the whole thresholded frame before selection did in an earlier,
    abandoned attempt at this -- there is nothing else left in `comp` to
    merge with. Recovered the confirmed truth pixel on every one of 20
    independently bad usb5/6 noise trials out of 30, matched exactly.

    `quad` must come from the coarse identification pose, not a tilt-refit
    one: a tilt refit's own few-px shift is enough to move the window and
    flip which fragment counts as "closest to the quad centre" on an
    asymmetric L-bracket, which is what broke usb5/usb6 during development.
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
    # Repair noise dropouts inside the winning blob only -- see the "sensor
    # noise" paragraph above. Everything upstream (which blob is `best`) is
    # already decided; this cannot change that answer, only patch small
    # holes within it.
    comp = cv2.morphologyEx((lab == best).astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((close_k, close_k), np.uint8))
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

    The two sides can still meet somewhere other than the centre, though --
    confirmed on a real capture where the ring's two side arcs only actually
    touch down near the bottom bevel, so the column running through the
    ring's own bounding-box centre passes only empty space between the arcs
    for most of its length and clips two tiny disjoint fragments where they
    meet at the bottom. That is exactly 2 runs, so the plain rule above took
    them for the true top and bottom and reported a 6 px sliver hugging the
    bottom edge. A "2 runs" reading is only trustworthy when its own span
    is a real fraction of the ring's own bounding box in that direction;
    one that is not gets treated as this line missing the ring the same way
    zero runs would, which is what lets the half=1/2 widening or the final
    bbox fallback take over instead of a coincidental double-hit winning.
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

    def sides(vec_row_fn, lo, hi, full_lo, full_hi):
        min_span = 0.5 * (full_hi - full_lo)
        for half in (0, 1, 2):
            il, ir = [], []
            for k in range(lo - half, hi + half + 1):
                rr = _runs(vec_row_fn(k))
                if len(rr) >= 2 and (rr[-1][0] - rr[0][1]) >= min_span:
                    il.append(rr[0][1]); ir.append(rr[-1][0])
            if il:
                return max(il), min(ir)
        # Nothing plausible at any width -- the ring's own outer bbox in
        # this direction is still a real measurement of it, just not an
        # "inner edge" one; better than refusing outright.
        return full_lo, full_hi

    inL, inR = sides(lambda r: ring[r], cy, cy, bx0, bx1)
    inT, inB = sides(lambda c: ring[:, c], cx, cx, by0, by1)
    return inL + x0, inT + y0, inR + x0, inB + y0


def plane_from_depth(depth, K, mask, trim=0.003, iters=3):
    """The panel's own plane. -> (point_on_plane, unit normal) or None.

    Deliberately the plane and not the depth at each hole's own pixel. A hole
    is a recess: measured on a real frame the sockets read 0.2-6.3 mm behind
    the panel face, rj451 deepest because an RJ45 shell is deep, so per-pixel
    depth answers a different question than "where is the opening". The
    plane the face sits on is what a plug meets, and every hole shares it.

    plane_normal_from_depth's own RANSAC already rejects a raised outlier
    (the handle it was written against), but this panel's holes are a
    bigger, one-sided contamination of a different shape: up to 19 recesses,
    0.2-14 mm deep, none of them raised, so they drag the mean (and, more,
    the fitted normal) toward "behind" on whichever side of the panel
    happens to have more open ports in view that frame -- not a random
    error, a systematic one that tilts with the port layout. Refitting after
    dropping everything recessed more than `trim` past the current plane, a
    few times so each pass' better plane can catch points the last pass'
    worse one missed, measured on one frame replayed 30 times with
    sensor-level noise across 3 seeds: tilt std 0.5-0.7 deg -> 0.04-0.05 deg,
    which ray_plane's own sensitivity turns into roughly an order of
    magnitude less lateral error at the panel's far corners, where the
    ray-to-normal angle is largest.
    """
    n = plane_normal_from_depth(depth, K, mask)
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


def inconsistent_ports(measured_3d, ports, tol_floor=0.004, tol_frac=0.08):
    """Which measured holes disagree with the rest of the panel's own CAD
    layout. -> set of port names.

    Every hole's own 3D point (ray_plane) is independent of every other's --
    that is the whole point of not letting CAD supply position -- but the
    panel itself is one rigid body, so the *distance* between any two real
    holes must still match the distance CAD says is between them, to within
    ordinary measurement noise. It does not need a pose or a scale to check:
    two already-measured 3D points and two already-known CAD points give
    both distances directly.

    This catches exactly the failure mode that reprojection error cannot: a
    hole whose frame_rect/ring_rect grabbed the wrong metal fragment still
    returns a confident, in-range-looking pixel, and nothing about that
    pixel alone looks wrong. But a fragment that is not this hole's own
    frame is not this hole's own distance to its neighbours either -- the
    jumps seen this session ran 15-20+ px, several mm at this panel's scale,
    well outside where plane-tilt and pixel noise alone land (~1-2 mm
    typical). tol_frac scales with distance because a small absolute error
    matters more between two close holes than two far ones; tol_floor keeps
    that from vanishing for the closest pairs.

    A point that disagrees with most of the others it was compared against
    is treated as the wrong one -- not by re-deriving a pose, just a vote:
    each pairwise disagreement counts once against each of that pair's two
    points, and a point flagged by at least half its comparisons loses.
    With exactly 2 measured points there is only one pair, and a
    disagreement there cannot say which of the two is at fault -- both are
    flagged rather than guessed at, since holding neither's last known-good
    value is safer than trusting an unverifiable pair. With fewer than 2
    there is nothing to compare at all.
    """
    names = list(measured_3d.keys())
    if len(names) < 2:
        return set()
    if len(names) == 2:
        a, b = names
        d_meas = float(np.linalg.norm(measured_3d[a] - measured_3d[b]))
        d_cad = float(np.linalg.norm(
            np.asarray(next(p for p in ports if p['name'] == a)['centre']) -
            np.asarray(next(p for p in ports if p['name'] == b)['centre'])))
        tol = max(tol_floor, tol_frac * d_cad)
        return {a, b} if abs(d_meas - d_cad) > tol else set()
    cad = {p['name']: np.asarray(p['centre'], dtype=np.float64) for p in ports}
    bad_votes = {n: 0 for n in names}
    total_votes = {n: 0 for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d_meas = float(np.linalg.norm(measured_3d[a] - measured_3d[b]))
            d_cad = float(np.linalg.norm(cad[a] - cad[b]))
            tol = max(tol_floor, tol_frac * d_cad)
            total_votes[a] += 1
            total_votes[b] += 1
            if abs(d_meas - d_cad) > tol:
                bad_votes[a] += 1
                bad_votes[b] += 1
    # >= half, not a strict majority: confirmed on a real corrupted point
    # that landed exactly on a 4-of-8 tie (it disagreed with everything
    # spatially near it and agreed with everything far away, since a small
    # absolute jump is a large fraction of a short CAD distance but a small
    # one of a long CAD distance) -- a coin-flip split is already reason
    # enough to distrust a point, not a reason to wave it through.
    return {n for n in names if bad_votes[n] >= 0.5 * total_votes[n]}


def recover_missing(measured_3d, ports, R):
    """Fill in a real socket this frame never measured (or measured but
    flagged, see inconsistent_ports) from its nearest still-good neighbour
    plus CAD's own relative offset. -> {port_name: (3,) point}, recovered
    entries only -- callers that need "did this port come from a real
    measurement or from this" should keep this dict separate from
    measured_3d rather than merge them.

    This is deliberately the same pose (x) CAD arithmetic the rest of this
    session spent so much effort removing from any *target* port's own
    published position -- reintroduced here only as an explicit, visible
    fallback so a panel showing 9 named crosses can mean something (every
    named socket has *a* position to check by eye) even on a frame where
    one hole's own measurement failed safely rather than wrong. It answers
    "where should this hole roughly be, given where a nearby real one
    actually is", not "where is this hole" -- callers must never treat a
    recovered point as equivalent to a measured one for anything that
    actually moves the arm.

    `R`: camera-frame rotation whose columns are the panel's own local
    axes (see panel_axes_pose) -- CAD's own centre-to-centre vector is
    expressed in that local frame and has to be rotated into camera
    coordinates before it means anything as a 3D offset. Nearest is
    measured in CAD's own layout (mm on the panel), not in the camera-frame
    points themselves, since a neighbour that is physically close on the
    real panel is also the one least likely to have accumulated a
    meaningfully different plane-tilt or lens-distortion error.
    """
    cad = {p['name']: np.asarray(p['centre'], dtype=np.float64)
          for p in ports if p.get('kind') != 'other'}
    present = [n for n in cad if n in measured_3d]
    if not present:
        return {}
    recovered = {}
    for name in cad:
        if name in measured_3d:
            continue
        nearest = min(present, key=lambda p: np.linalg.norm(cad[p] - cad[name]))
        recovered[name] = measured_3d[nearest] + R @ (cad[name] - cad[nearest])
    return recovered


def pose_matrix(rvec, tvec):
    """(rvec, tvec) -> the 4x4 T_camera_object the rest of the stack passes around."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


# ---------------------------------------------------------------- bring-up

def _port_quad(p, rvec, tvec, K, dist):
    """A port's own CAD footprint, projected. -> (4, 2) pixel corners.

    A circle drawn on a rectangular port cannot show whether the fit is
    off in the direction along the port or across it -- both look like the
    same amount of circle overhanging the opening. Projecting the actual
    rectangle, at the CAD's own length, width and orientation, shows a
    real port-shaped gap on the specific side it exists on instead.
    """
    c = np.array(p['centre'], dtype=np.float64)
    along = np.array(p['long_axis'][:2], dtype=np.float64)
    across = np.array([-along[1], along[0]])
    L, S = p['size']
    corners = np.array(
        [[*(c[:2] + su * along * L / 2 + sv * across * S / 2), c[2]]
         for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))], dtype=np.float64)
    q, _ = cv2.projectPoints(corners, rvec, tvec, K, dist)
    return q.reshape(-1, 2)


def debug_image(bgr, outline, candidates, pairs, cad_ports, image_points,
                rvec, tvec, K, dist=None, target=None, max_side=700, crop_pad=40,
                show_other=True, measured_uv=None, show_outline=True,
                show_candidates=True, panel_centre=None, recovered_uv=None):
    """What the detector saw, drawn on the frame it saw it in.

    The depth route drew its debug view on the rectified grid, which is honest
    about what that algorithm works on but hard to check against the part in
    front of you. Here the annotation goes on the camera image, so a wrong
    correspondence -- the failure that matters, because it is the one that reads
    as success -- is obvious at a glance: the label sits on the wrong socket.

    measured_uv: optional {port_name: np.array([u, v]) | None}, one entry per
    real socket (every non-'other' entry in cad_ports). When given, every
    socket it names that this frame actually measured (see frame_rect above)
    gets a magenta cross at its own pixel; a name with no measurement draws
    nothing at all, rather than a CAD-projected guess. Omit entirely (the
    default) to keep the plain identification-only view -- a panel with no
    frame_rect/ring_rect tuned for its own port shapes has nothing to put in
    this dict and should never pass one.

    panel_centre: optional (3,) point in camera coords -- the panel's own
    outer-rectangle midpoint (see find_outer_quad), not tvec, which is the
    current target port's own position now, not the panel's. When given,
    draws the panel's long axis (green) and short axis (orange) from there;
    see panel_axes_pose for what rvec's own X/Y columns mean here.

    recovered_uv: optional {port_name: np.array([u, v])}, for a socket
    mpl.recover_missing filled in from a nearby real one rather than this
    frame measuring it -- drawn as a yellow cross, same shape as a real
    measurement so it is just as easy to spot, but a different colour so it
    still reads as "known from a neighbour, not measured here" rather than
    being mistaken for one.
    """
    dist = np.zeros(5) if dist is None else dist
    vis = bgr.copy()
    if outline is not None and show_outline:
        # Smoothed for display only -- the outline drawn here is not the one
        # anything downstream measures from (scale comes off the depth
        # stream, not the outline's pixel size; see scale_from_depth), so
        # simplifying it costs nothing but the per-pixel jitter a depth-
        # thresholded boundary always has.
        peri = cv2.arcLength(outline, True)
        smooth = cv2.approxPolyDP(outline, max(1.5, 0.003 * peri), True)
        cv2.drawContours(vis, [smooth], -1, (255, 128, 0), 2)
    if show_candidates:
        for c in candidates:
            cv2.circle(vis, tuple(np.round(c['centre']).astype(int)), 4, (110, 110, 110), 1)

    if pairs is not None and rvec is not None:
        obj = _object_points(pairs, cad_ports)
        rp, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rp = rp.reshape(-1, 2)
        # A fixed marker radius reads fine at the range this was first tuned
        # against, then silently stops meaning anything once the part is
        # further off: a 9 px circle drawn on an 18 mm port at 220 mm and on
        # a 13 px-wide one at 320 mm are not the same claim. And a circle
        # cannot show which side of a rectangular port a fit missed on --
        # see _port_quad. So the marker is the port's own CAD rectangle,
        # projected through this pose, at this port's own size: it sits
        # inside the opening it is marking when the fit is right, and the
        # gap reads as a real port-shaped gap, on the actual side it is on,
        # when it is not.
        for k, (ci, _) in enumerate(pairs):
            # "other" ports (case1's eleven unlabelled panel features) anchor
            # the pose the same as any other point, but nobody is ever going
            # to insert into one. Shown by default -- which points are doing
            # the anchoring is exactly what explains a target port's own
            # accuracy when it has few near neighbours of its own -- with
            # show_other=False for a view crowded enough to want just the
            # eight sockets the task cares about.
            is_other = cad_ports[ci].get('kind') == 'other'
            if not show_other and is_other:
                continue
            if measured_uv is not None and not is_other:
                # Every real socket gets its own image-only hole-centre
                # marker in the dedicated pass below instead -- the
                # identification candidate this loop would otherwise draw
                # for it is a different, less trustworthy pixel (see
                # frame_rect's own docstring), and should not compete with
                # it on screen.
                continue
            name = cad_ports[ci].get('name', str(ci))
            hit = target is not None and name == target
            col = (0, 255, 255) if hit else (0, 220, 0)
            pt = np.round(image_points[k]).astype(int)
            rpt = tuple(np.round(rp[k]).astype(int))
            if is_other:
                # An "other" port's CAD size is whatever the mesh's own
                # opening happened to be, unfiltered -- a VGA cutout's
                # extent runs 31.75 x 10 mm because the ray-cast segmented
                # the mounting-screw region as part of the same opening, not
                # because a 31.75 mm socket exists there. Drawing that as a
                # port-shaped rectangle claims a boundary nobody should read
                # anything from; a plain cross makes no such claim.
                cv2.drawMarker(vis, tuple(pt), col, cv2.MARKER_CROSS, 8, 1)
                cv2.drawMarker(vis, rpt, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
                label_at = pt
            else:
                quad = _port_quad(cad_ports[ci], rvec, tvec, K, dist)
                cv2.polylines(vis, [np.round(quad).astype(int)], True, col, 2)
                cv2.drawMarker(vis, tuple(pt), col, cv2.MARKER_CROSS, 8, 1)
                cv2.drawMarker(vis, rpt, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
                label_at = quad[np.argmin(quad[:, 1])]    # above the top corner
            cv2.putText(vis, name, tuple(np.round(label_at).astype(int) + [3, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # A target socket the correspondence search never landed on is not a
        # blank on this view -- the pose the matched points fixed applies to
        # it exactly as it does to any of them, since the panel is one rigid
        # body. Projecting it too, in a shape distinct from a match, is the
        # difference between "not detected" reading as "not usable" and
        # reading as what it actually is: a position the operator can trust
        # the same way, just without a second, independent measurement to
        # check it against.
        matched = {cad_ports[ci]['name'] for ci, _ in pairs}
        missing = [i for i, p in enumerate(cad_ports)
                  if p.get('kind') != 'other' and p['name'] not in matched]
        if measured_uv is None:
            for i in missing:
                p = cad_ports[i]
                quad = _port_quad(p, rvec, tvec, K, dist)
                cv2.polylines(vis, [np.round(quad).astype(int)], True, (255, 0, 0), 2)
                corner = quad[np.argmin(quad[:, 1])]
                cv2.putText(vis, p['name'], tuple(np.round(corner).astype(int) + [3, -6]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

        # Two arrows from the panel's own centre (panel_centre: the Canny
        # outer-rectangle's own midpoint -- see find_outer_quad -- turned
        # into a 3D point the same way a hole's own pixel is, via ray_plane),
        # not from tvec: tvec is the current target port's own position now,
        # not the panel's, and drawing from it would put the origin
        # somewhere on the panel's edge rather than its middle. Long axis
        # only (R's own Y column, screen up = +Y) -- the short axis (X) is
        # never independently used for its own value (see panel_axes_pose:
        # it only decides a sign, the real X is Y cross Z), so drawing it
        # claimed an independent measurement that was not actually there.
        if panel_centre is not None:
            Rm = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
            pts3 = np.array([panel_centre, panel_centre + 0.04 * Rm[:, 1]])
            pp, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), K, dist)
            pp = np.round(pp.reshape(-1, 2)).astype(int)
            cv2.arrowedLine(vis, tuple(pp[0]), tuple(pp[1]), (0, 255, 0), 2, tipLength=0.25)
            cv2.putText(vis, '+Y', tuple(pp[1] + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Every real socket the image-only hole-centre method (see frame_rect
        # above) actually measured this frame -- magenta at its own measured
        # pixel. One this frame did not measure draws nothing at all: a CAD
        # projection here would claim a position nothing this frame actually
        # supports, which is exactly the pose (x) CAD contamination this
        # method exists to avoid -- see depth_pose_node's per-hole
        # correction, which likewise refuses to publish rather than fall
        # back to one for its own target port.
        if measured_uv is not None:
            for p in cad_ports:
                if p.get('kind') == 'other':
                    continue
                uv = measured_uv.get(p['name'])
                if uv is None:
                    continue
                pt = tuple(np.round(uv).astype(int))
                cv2.drawMarker(vis, pt, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(vis, p['name'], (pt[0] + 6, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        if recovered_uv is not None:
            for name, uv in recovered_uv.items():
                pt = tuple(np.round(uv).astype(int))
                cv2.drawMarker(vis, pt, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(vis, name, (pt[0] + 6, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # crop to the part before scaling: the panel covers a quarter of the frame at
    # working distance, and a whole-frame thumbnail leaves the labels unreadable
    # on the phone the stream usually gets watched on
    if outline is not None and crop_pad is not None:
        x, y, w, h = cv2.boundingRect(outline)
        H, W = vis.shape[:2]
        vis = vis[max(0, y - crop_pad):min(H, y + h + crop_pad),
                  max(0, x - crop_pad):min(W, x + w + crop_pad)]
    scale = max_side / max(vis.shape[:2])
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return vis


# ------------------------------------------------------------- over frames

class PoseSmoother:
    """Median-filters a stream of (rvec, tvec), rejecting single-frame jumps.

    The task this exists for holds one pose for seconds at a time -- the arm
    is stationary between attempts, waiting for the part -- so consecutive
    frames should barely differ. When they visibly do, per-frame noise (a
    weak candidate on this frame's ladder attempt, a stray reflection) is the
    likelier explanation than the part having actually moved, and a median
    over a short recent window is free variance reduction there: it takes
    the same amount of per-frame computation as trusting whichever frame just
    happened to solve, and is wrong less often.

    A frame far from the current median is not blended in -- one bad frame
    should not be able to drag a five-frame median a fifth of the way toward
    itself -- but push still returns the median, so a single flickered frame
    does not interrupt the published pose at all; the caller sees `accepted`
    go False and can log it, but has a usable pose regardless.

    History expires after max_age_s: if nothing has verified in a while, the
    part may genuinely have moved, or this is a cold start, and the next
    accepted frame should not be measured against a memory that old.

    size defaults to 9 rather than 5 because 5 measurably is not enough here.
    This panel is 44 x 159 mm -- long and narrow -- and the detected ports
    span nearly all of its length but only about half its width, so rotation
    about the long axis is weakly constrained and centroid noise turns into
    disproportionate error at any port far from the matched cluster (usb1 and
    usb2, at the top, are typically recovered by projection rather than
    detected at all). Replaying one real frame 60 times with sensor-level
    noise added, the worst target-port error was 36 px unsmoothed, still
    22.7 px at size 5, and 17.9 px at size 9 -- where outliers past 20 px
    disappeared entirely. The median barely moves across all three (12.2 ->
    11.5 px): the window is not there to improve a typical frame, it is there
    to stop an occasional bad one from being published, which is what a
    visibly crooked frame on the live stream actually is. Note that per-frame
    reprojection error does not predict this at all -- across those trials a
    1.41 px frame gave 19.2 px of target error and a 3.05 px frame gave 11.5
    -- so max_reproj_px cannot substitute for the window, and vice versa.
    """

    def __init__(self, size=9, max_rot_jump_deg=8.0, max_trans_jump_m=0.015,
                max_age_s=2.0, confirm_streak=3):
        self.size = size
        self.max_rot_jump_deg = max_rot_jump_deg
        self.max_trans_jump_m = max_trans_jump_m
        self.max_age_s = max_age_s
        self.confirm_streak = confirm_streak
        self._hist = []          # [(rvec(3,), tvec(3,), timestamp), ...]
        self._streak = 0         # consecutive accepted pushes since the last reject/reset

    def reset(self):
        self._hist = []
        self._streak = 0

    def push(self, rvec, tvec, now):
        """One frame's raw pose in, the smoothed pose out.

        -> (rvec, tvec, accepted, confirmed). accepted is False for a frame
        judged too far from recent history to be this frame's own
        measurement -- the pose returned is still the current median, not
        this frame's, and the streak resets to 0.

        confirmed is True once `confirm_streak` frames in a row have been
        accepted. A single accepted frame proves only that it agreed with
        whatever came right before it, which on a cold start can be one
        other frame -- confirmed exists for a caller that needs to know the
        estimate has actually settled, not merely that the latest sample
        did not jump, before treating it as good enough to act on.
        """
        stale = bool(self._hist) and now - self._hist[-1][2] > self.max_age_s
        self._hist = [h for h in self._hist if now - h[2] <= self.max_age_s]
        if stale:
            self._streak = 0        # a gap this long means start the streak over
        rv = np.asarray(rvec, dtype=np.float64).ravel()
        tv = np.asarray(tvec, dtype=np.float64).ravel()

        if len(self._hist) >= 3:
            med_rv = np.median(np.array([h[0] for h in self._hist]), axis=0)
            med_tv = np.median(np.array([h[1] for h in self._hist]), axis=0)
            R1, _ = cv2.Rodrigues(rv)
            R2, _ = cv2.Rodrigues(med_rv)
            dR = R1 @ R2.T
            ang = float(np.degrees(
                np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            dist = float(np.linalg.norm(tv - med_tv))
            if ang > self.max_rot_jump_deg or dist > self.max_trans_jump_m:
                self._streak = 0
                return med_rv.reshape(3, 1), med_tv.reshape(3, 1), False, False

        self._hist.append((rv, tv, now))
        self._hist = self._hist[-self.size:]
        self._streak += 1
        rvs = np.array([h[0] for h in self._hist])
        tvs = np.array([h[1] for h in self._hist])
        confirmed = self._streak >= self.confirm_streak
        return (np.median(rvs, axis=0).reshape(3, 1),
               np.median(tvs, axis=0).reshape(3, 1), True, confirmed)


class HoleSmoother:
    """Median-filters and gap-holds one named hole's own measured pixel.

    frame_rect/ring_rect fail outright on some frames for reasons that have
    nothing to do with the panel moving -- glare, the wrong metal fragment
    winning Otsu that frame -- and falling back to the CAD projection on
    exactly those frames would reintroduce the CAD's own un-averaged position
    error right when there is no reason to think the CAD guess got any
    better that frame. Holding the last accepted pixel across a short gap,
    and reporting the median of a short recent window rather than whichever
    frame happened to land, gives this measurement the same free variance
    reduction PoseSmoother already gives the 6-DOF pose -- same rationale,
    applied to the one thing that actually varies frame to frame here: the
    hole's own pixel, not the whole pose.

    Keyed by port name so a node tracking one target port (the common case)
    and one tracking several both work off the same instance.
    """

    def __init__(self, size=9, max_gap=30):
        self.size = size
        self.max_gap = max_gap
        self._hist = {}   # name -> [np.array([u, v]), ...], most recent last
        self._gap = {}    # name -> frames since the last accepted measurement

    def push(self, name, uv):
        """This frame's own measured pixel (or None) in for one named hole.

        -> np.array([u, v]) | None. None means the hole has gone unmeasured
        for more than max_gap frames in a row -- long enough that holding
        the old value is no longer defensible, and the caller should fall
        back to the CAD projection for this frame instead.
        """
        hist = self._hist.setdefault(name, [])
        if uv is not None:
            hist.append(np.asarray(uv, dtype=np.float64))
            del hist[:-self.size]
            self._gap[name] = 0
        else:
            self._gap[name] = self._gap.get(name, 0) + 1
        if not hist or self._gap[name] > self.max_gap:
            return None
        return np.median(np.array(hist), axis=0)