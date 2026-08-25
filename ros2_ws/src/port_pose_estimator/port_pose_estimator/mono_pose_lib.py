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


def pose_matrix(rvec, tvec):
    """(rvec, tvec) -> the 4x4 T_camera_object the rest of the stack passes around."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return T


# ---------------------------------------------------------------- bring-up

def debug_image(bgr, outline, candidates, pairs, cad_ports, image_points,
                rvec, tvec, K, dist=None, target=None, max_side=700, crop_pad=40):
    """What the detector saw, drawn on the frame it saw it in.

    The depth route drew its debug view on the rectified grid, which is honest
    about what that algorithm works on but hard to check against the part in
    front of you. Here the annotation goes on the camera image, so a wrong
    correspondence -- the failure that matters, because it is the one that reads
    as success -- is obvious at a glance: the label sits on the wrong socket.
    """
    dist = np.zeros(5) if dist is None else dist
    vis = bgr.copy()
    if outline is not None:
        cv2.drawContours(vis, [outline], -1, (255, 128, 0), 2)
    for c in candidates:
        cv2.circle(vis, tuple(np.round(c['centre']).astype(int)), 4, (110, 110, 110), 1)

    if pairs is not None and rvec is not None:
        obj = _object_points(pairs, cad_ports)
        rp, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rp = rp.reshape(-1, 2)
        for k, (ci, _) in enumerate(pairs):
            name = cad_ports[ci].get('name', str(ci))
            hit = target is not None and name == target
            col = (0, 255, 255) if hit else (0, 220, 0)
            pt = np.round(image_points[k]).astype(int)
            cv2.circle(vis, tuple(pt), 13 if hit else 9, col, 2)
            cv2.drawMarker(vis, tuple(np.round(rp[k]).astype(int)),
                           (0, 0, 255), cv2.MARKER_CROSS, 11, 1)
            cv2.putText(vis, name, tuple(pt + [12, -9]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        ax = np.float64([[0, 0, 0], [0.04, 0, 0], [0, 0.04, 0], [0, 0, 0.04]])
        ap, _ = cv2.projectPoints(ax, rvec, tvec, K, dist)
        ap = np.round(ap.reshape(-1, 2)).astype(int)
        for q, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.arrowedLine(vis, tuple(ap[0]), tuple(ap[q]), col, 2, tipLength=0.25)

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
