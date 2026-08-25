"""Pose estimation from depth alone, for parts lying flat with the opening up.

An alternative to the FoundationPose route (which stays untouched). The problem
this solves is narrower than general 6D pose: the work always lies flat on the
table with its opening facing the camera, so the plane itself pins down Z and
tilt, and only the in-plane position and heading are left to find. That lets the
whole thing run on classical geometry -- no GPU, no network, no separate Python
environment -- and it measures the opening directly instead of inferring it from
the part's outer silhouette.

Two ways to finish, sharing everything up to opening detection:

  solve_single_opening()  one opening (the RJ45 jig). Centre gives position,
                          the opening's long axis gives heading mod 180 deg, and
                          the depth asymmetry across the opening breaks the tie.

  solve_opening_pattern() many openings (the multi-port platform). The pattern of
                          centres is matched against the CAD table by RANSAC +
                          Kabsch, which fixes the heading outright -- an
                          asymmetric pattern has only one way to line up.

Everything here is plain numpy/scipy/cv2 and holds no ROS or hardware state, so
it can be exercised on synthetic depth rendered from the CAD (see
tools/test_depth_pose.py) with no camera attached.
"""
import numpy as np
import cv2
from scipy import ndimage


# ---------------------------------------------------------------- geometry

def deproject(depth, K, z_range=(0.05, 1.0)):
    """Depth image (metres) -> (N,3) points in the camera frame.

    Standard pinhole model. Depth images store the z-component, not the ray
    length, so the multiply below is by the normalised-by-z ray, not a unit ray.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    ok = np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
    z = z[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def plane_from_points(pts):
    """Least-squares plane through a point set -> (unit normal, centroid).

    Via the 3x3 covariance rather than an SVD of the (N,3) matrix: numpy's svd
    builds the full N x N left factor, which for a real frame (~10^5 points)
    tries to allocate tens of gigabytes. The smallest eigenvector of the
    covariance is the same answer for O(N) work.
    """
    centroid = pts.mean(axis=0)
    d = pts - centroid
    n = np.linalg.eigh(d.T @ d)[1][:, 0]
    return n, centroid


def fit_plane_ransac(pts, tol=0.0015, iters=300, seed=0, max_samples=6000):
    """Dominant plane of a point cloud -> (unit normal, point on plane, inliers).

    RANSAC rather than a plain least-squares fit because the frame also contains
    the table, cabling and the gripper; fitting everything at once would return a
    plane that belongs to nothing. The normal is flipped to face the camera so
    downstream sign conventions are stable.

    Hypotheses are scored against a random subset -- scoring 300 of them against
    every point of a full frame costs tens of millions of distance evaluations
    per frame and will not keep up with the camera. The subset only has to rank
    the hypotheses; the winner is then re-measured against every point.
    """
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= max_samples else pts[rng.choice(len(pts), max_samples,
                                                             replace=False)]

    best_inl, best = -1, None
    for _ in range(iters):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        inl = int((np.abs((sub - s[0]) @ n) < tol).sum())
        if inl > best_inl:
            best_inl, best = inl, (n, s[0])
    if best is None:
        return None

    # refine on the full consensus set: 3 random points fix the hypothesis, but
    # the accuracy should come from every point that agreed with it
    n, p = best
    mask = np.abs((pts - p) @ n) < tol
    if mask.sum() < 50:
        return None
    n, centroid = plane_from_points(pts[mask])
    if n[2] > 0:                      # camera looks down +Z, so face it
        n = -n
    return n, centroid, int(mask.sum())


def height_above_plane(pts, normal, origin):
    """Signed height of each point above a plane, positive towards the camera."""
    return (pts - origin) @ normal


def object_above_plane(depth, K, normal, origin, min_h=0.008, max_h=0.060,
                       min_area_px=800, z_range=(0.05, 1.0), expected_size=None):
    """Isolate the work standing on a plane -> (points, mask).

    A height threshold alone is not enough. The table here is white and
    featureless, which is the worst case for an active-stereo sensor: its depth
    scatters by several millimetres, so thresholding at "above the table" lights
    up the frame with speckle and every fleck becomes a candidate opening.
    Connected components fix that -- noise is scattered and small, the work is one
    contiguous region.

    Which component, though, cannot be "the biggest". That holds only while
    nothing else is in shot: backing the camera off to 335mm widened the view
    enough to include the teach pendant, and at 106x119mm it beats the 43x46mm
    jig outright. expected_size (from the CAD) picks by resemblance instead,
    which does not depend on what else happens to be on the table.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ok = np.isfinite(depth) & (depth > z_range[0]) & (depth < z_range[1])
    if ok.sum() < 500:
        return None, None

    z = depth[ok]
    pts_all = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0],
                        (v[ok] - K[1, 2]) * z / K[1, 1], z], axis=1)
    hh = np.full(depth.shape, np.nan)
    hh[ok] = height_above_plane(pts_all, normal, origin)

    band = ((hh > min_h) & (hh < max_h)).astype(np.uint8)
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    if n_lab <= 1:
        return None, None
    cand = [i for i in range(1, n_lab)
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    if not cand:
        return None, None

    if expected_size is None:
        best = max(cand, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    else:
        ew, eh = sorted([float(expected_size[0]), float(expected_size[1])])[::-1]
        def mismatch(i):
            m = lab == i
            zz = depth[m]
            mmpx = float(np.median(zz)) / K[0, 0]
            w = stats[i, cv2.CC_STAT_WIDTH] * mmpx
            h = stats[i, cv2.CC_STAT_HEIGHT] * mmpx
            w, h = sorted([w, h])[::-1]
            return abs(w - ew) / ew + abs(h - eh) / eh
        best = min(cand, key=mismatch)
        if mismatch(best) > 0.8:
            return None, None

    mask = (lab == best)
    # take the whole silhouette, holes included: the socket itself reads as a
    # gap here, and it is the one part that must not be discarded
    mask = ndimage.binary_fill_holes(mask) & ok
    zz = depth[mask]
    pts = np.stack([(u[mask] - K[0, 2]) * zz / K[0, 0],
                    (v[mask] - K[1, 2]) * zz / K[1, 1], zz], axis=1)
    return pts, mask


def object_from_grey(grey, depth, K, normal, origin, expected_size,
                     min_h=0.004, max_h=0.060, z_range=(0.05, 1.0)):
    """Segment the work by brightness, confirmed by height -> (points, mask).

    object_above_plane() finds the work by how far it stands off the table, which
    works while the sensor can see it. On this platform it cannot: the body is
    black, IR is absorbed, and only about half the pixels return anything -- the
    silhouette that comes back is ragged and bleeds well past the real edges.

    In greyscale the same part is unmistakable, black against a white table. What
    brightness cannot tell on its own is a raised part from a dark mark printed
    flat on the table, so the two are combined: a candidate must look right *and*
    stand above the plane.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = np.isfinite(depth) & (depth > z_range[0]) & (depth < z_range[1])
    if valid.sum() < 500:
        return None, None

    thr, _ = cv2.threshold(grey.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx((grey < thr).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((7, 7), np.uint8))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n_lab <= 1:
        return None, None

    z = depth[valid]
    pts_all = np.stack([(u[valid] - K[0, 2]) * z / K[0, 0],
                        (v[valid] - K[1, 2]) * z / K[1, 1], z], axis=1)
    hh = np.full(depth.shape, np.nan)
    hh[valid] = height_above_plane(pts_all, normal, origin)

    ew, eh = sorted([float(expected_size[0]), float(expected_size[1])])[::-1]
    best, best_err = None, 1e9
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] < 800:
            continue
        m = lab == i
        zz = depth[m & valid]
        if len(zz) < 100:
            continue
        raised = np.nanmean((hh[m] > min_h) & (hh[m] < max_h))
        if raised < 0.3:                  # flat marking on the table, not a part
            continue
        mmpx = float(np.median(zz)) / K[0, 0]
        a, b = sorted([stats[i, cv2.CC_STAT_WIDTH] * mmpx,
                       stats[i, cv2.CC_STAT_HEIGHT] * mmpx])[::-1]
        err = abs(a - ew) / ew + abs(b - eh) / eh
        if err < best_err:
            best, best_err = i, err
    if best is None or best_err > 0.8:
        return None, None

    mask = ndimage.binary_fill_holes(lab == best) & valid
    zz = depth[mask]
    pts = np.stack([(u[mask] - K[0, 2]) * zz / K[0, 0],
                    (v[mask] - K[1, 2]) * zz / K[1, 1], zz], axis=1)
    return pts, mask


largest_object_above_plane = object_above_plane   # kept: older call sites


def select_above_plane(pts, normal, origin, min_h=0.004, max_h=0.060):
    """Keep the points sitting on top of a reference plane.

    The work has to be separated from the table before anything can be measured
    on it, and depth alone will not do it: the camera is mounted at 24.4 deg, so
    a flat table already spans ~120mm of depth across the frame (measured: 0.20m
    at one edge, 0.32m at the other). Picking "the nearest returns" therefore
    selects the near edge of the table, not the part standing on it.

    Height above the *fitted table plane* is the quantity that actually
    distinguishes them, and it is immune to the tilt.
    """
    h = height_above_plane(pts, normal, origin)
    return pts[(h > min_h) & (h < max_h)]


def correct_plane_scale(rect, grey_grid, true_size, normal, origin):
    """Re-place the reference plane so measured size matches the CAD -> new origin.

    Absolute depth is not trustworthy on this hardware. The platform is black,
    black absorbs the projector's IR, and the returns come back systematically
    far -- 0.309m against the 0.291m its own outline implies. Everything the
    rectifier produces then inherits that error as a scale factor, and a port
    pattern 6% oversized will not match the CAD at any tolerance.

    Measuring the part in the rectified grid and comparing against the size the
    CAD already states gives the correction directly. The relation is linear, so
    one pass is enough; working in the rectified grid means the tilt has been
    divided out before anything is measured.
    """
    dark = np.isfinite(grey_grid) & (grey_grid < np.nanpercentile(grey_grid, 45))
    dark = ndimage.binary_opening(dark, np.ones((5, 5)))
    lab, n = ndimage.label(dark)
    if n == 0:
        return None, None
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    blob = (lab == int(np.argmax(sizes)) + 1)
    ys, xs = np.where(blob)
    if len(ys) < 100:
        return None, None
    (_, _), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
    measured = np.array([max(w, h), min(w, h)]) * rect['step']
    truth = np.array([max(true_size), min(true_size)])
    ratio = float(np.mean(measured / truth))
    if not (0.5 < ratio < 2.0):
        return None, ratio
    # the plane sits along the view ray, so shrinking its distance by the same
    # factor shrinks everything the rectifier reads off it
    return origin / ratio, ratio


def grid_step_for(depth_m, fx, points_per_cell=2.0):
    """Choose a rectification cell size the sensor can actually fill.

    One pixel subtends depth/fx metres on the surface, so a grid finer than that
    leaves most cells empty and the surface breaks up into speckle -- which then
    reads as holes everywhere. Ask for a couple of points per cell instead.
    """
    spacing = depth_m / fx
    return float(spacing * np.sqrt(points_per_cell))


def plane_basis(normal):
    """Two orthonormal in-plane axes, laid out to match the camera's view.

    Any perpendicular pair would serve the maths equally, but the rectified grid
    is also what gets drawn for a human, and a basis picked for numerical
    convenience gives a picture whose left and right mean nothing. Projecting the
    camera's own X axis onto the plane makes the debug view read like the camera:
    +a1 to the right, +a2 down.

    Falls back to the least-aligned world axis if the camera's X happens to be
    perpendicular to the plane, which cannot occur while the part faces the lens
    but costs nothing to guard.
    """
    cam_x = np.array([1.0, 0.0, 0.0])
    a1 = cam_x - np.dot(cam_x, normal) * normal
    if np.linalg.norm(a1) < 1e-6:
        ref = np.eye(3)[int(np.argmin(np.abs(normal)))]
        a1 = np.cross(normal, ref)
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(normal, a1)
    # To get +a2 pointing down the image, flip a1 and recompute -- negating a2
    # on its own turns (a1, a2, normal) left-handed, and every cross product
    # downstream assumes otherwise. Doing that silently halved the flip
    # decision's accuracy, from 83% of frames to 47%.
    if a2[1] < 0:
        a1 = -a1
        a2 = np.cross(normal, a1)
    return a1, a2


def rectify(pts, normal, origin, step=0.0005, margin=0.005, values=None):
    """Project onto the plane -> a metric, straight-on view of the surface.

    Removes perspective and camera tilt in one step: after this, distances in
    the grid are true millimetres regardless of how the camera is mounted, so
    measured opening sizes can be compared against the CAD directly.

    Returns the recess grid (NaN where nothing was seen), the two in-plane axes,
    and the offsets needed to map a grid cell back to a 3D camera-frame point.
    """
    a1, a2 = plane_basis(normal)
    rel = pts - origin
    s, t = rel @ a1, rel @ a2
    recess = -(rel @ normal)          # positive = further from the camera

    s0, t0 = s.min() - margin, t.min() - margin
    gi = np.round((s - s0) / step).astype(int)
    gj = np.round((t - t0) / step).astype(int)
    grid = np.full((gi.max() + 1, gj.max() + 1), np.nan)
    # nearest cell wins; with a fine step the collisions are between neighbours
    # on the same surface, so which one lands is immaterial
    grid[gi, gj] = recess

    # An optional second channel riding on the same cells -- greyscale, in
    # practice. Rectifying colour alongside depth keeps the two co-registered
    # and metric, so a feature found in one can be measured in the other's frame
    # without any further correspondence work.
    value_grid = None
    if values is not None:
        value_grid = np.full(grid.shape, np.nan)
        value_grid[gi, gj] = values

    return dict(grid=grid, value_grid=value_grid, a1=a1, a2=a2, s0=s0, t0=t0,
                step=step, normal=normal, origin=origin)


def grid_to_camera(rect, si, tj, recess=0.0):
    """Grid coordinates (may be fractional) -> a 3D point in the camera frame."""
    s = rect['s0'] + si * rect['step']
    t = rect['t0'] + tj * rect['step']
    return rect['origin'] + s * rect['a1'] + t * rect['a2'] - recess * rect['normal']


def camera_to_grid(rect, p):
    """3D camera-frame point -> (fractional) grid coordinates."""
    rel = np.asarray(p) - rect['origin']
    return ((rel @ rect['a1'] - rect['s0']) / rect['step'],
            (rel @ rect['a2'] - rect['t0']) / rect['step'])


def sample_grid(rect, si, tj, nan_value):
    """Read the recess grid at fractional coordinates, nearest cell."""
    grid = rect['grid']
    i, j = int(round(si)), int(round(tj))
    if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
        return None
    v = grid[i, j]
    # inside the part, "no return" means the sensor saw into a cavity rather
    # than that the reading is missing -- scoring it as deep is the honest read
    return nan_value if np.isnan(v) else float(v)


# ------------------------------------------------------- opening detection

def find_openings(rect, min_area=3e-5, max_area=6e-4, min_recess=0.002):
    """Locate openings in the rectified surface.

    Two things count as "opening": cells that read clearly deeper than the
    surface, and cells inside the part's footprint with no return at all. The
    second case matters more than it sounds -- a 12x4.5mm USB mouth often gives
    the sensor nothing to bounce off, and treating that as missing data rather
    than as evidence would throw away the strongest signal available.

    Centres come from image moments, i.e. an average over every pixel of the
    blob, which is what buys sub-pixel (sub-0.5mm) accuracy from a coarse grid.
    """
    grid = rect['grid']
    step = rect['step']
    solid = np.isfinite(grid)

    # footprint = the part's silhouette, holes filled back in
    footprint = ndimage.binary_fill_holes(
        ndimage.binary_closing(solid, np.ones((7, 7))))
    mask = (footprint & ~solid) | (solid & (grid > min_recess))
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))

    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        blob = (lab == i)
        area = blob.sum() * step * step
        if not (min_area <= area <= max_area):
            continue
        ys, xs = np.where(blob)
        m = cv2.moments(blob.astype(np.uint8), binaryImage=True)
        # cv2 treats the array as an image: m10 is the column (axis 1) moment and
        # m01 the row (axis 0) one, which is the opposite way round to how the
        # grid is indexed here
        ci = m['m01'] / m['m00']      # along axis 0 of the grid
        cj = m['m10'] / m['m00']      # along axis 1

        # Orientation from the central second moments rather than from a minimum
        # -area rectangle. An RJ45 mouth is only 12.5 x 9.5mm, so the long side
        # wins by a slim margin and the rectangle -- which is decided by a few
        # hull points -- flips to the short axis under noise. Second moments
        # average over every pixel in the blob and stay put.
        mu20, mu11, mu02 = m['mu20'], m['mu11'], m['mu02']
        ang = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

        # size still comes from the enclosing rectangle, which is what the CAD
        # dimensions can be compared against
        (_, _), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        w, h = max(w, h) * step, min(w, h) * step
        out.append(dict(ci=ci, cj=cj, size=(w, h), area=area,
                        angle_deg=ang, pixels=blob))
    return out


def find_openings_grey(rect, min_area=3e-5, max_area=2.5e-4, bridge=0.004):
    """Locate ports from the rectified greyscale instead of from depth.

    On this platform depth cannot see the ports at all: the body is black, black
    absorbs the projector's IR, and only 49% of the rectified cells come back
    with any value -- the ports are lost in that noise, and thresholding it
    yields blobs 8-16mm apart when the real ports are 29mm apart at closest.
    The same frame in greyscale shows all eight ports sharply, bright metal
    against a black body, with Otsu separating them unaided.

    What is bright is the shell's *rim*, a thin frame, and it arrives broken into
    pieces. Left alone each piece becomes its own detection and the pattern is
    meaningless (12 blobs for 8 ports). Closing across `bridge` and filling
    rejoins each rim into one port-shaped region.
    """
    vg = rect.get('value_grid')
    if vg is None:
        return []
    step = rect['step']
    footprint = ndimage.binary_fill_holes(
        ndimage.binary_closing(np.isfinite(rect['grid']), np.ones((9, 9))))
    usable = footprint & np.isfinite(vg)
    if usable.sum() < 500:
        return []

    thr, _ = cv2.threshold(vg[usable].astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(int(round(bridge / step)), 3)
    blobs = ndimage.binary_fill_holes(
        ndimage.binary_closing(usable & (vg > thr), np.ones((k, k))))

    lab, n = ndimage.label(blobs)
    out = []
    for i in range(1, n + 1):
        blob = lab == i
        area = blob.sum() * step * step
        if not (min_area <= area <= max_area):
            continue
        ys, xs = np.where(blob)
        m = cv2.moments(blob.astype(np.uint8), binaryImage=True)
        mu20, mu11, mu02 = m['mu20'], m['mu11'], m['mu02']
        (_, _), (w, h), _ = cv2.minAreaRect(
            np.stack([xs, ys], axis=1).astype(np.float32))
        out.append(dict(ci=m['m01'] / m['m00'], cj=m['m10'] / m['m00'],
                        size=(max(w, h) * step, min(w, h) * step), area=area,
                        angle_deg=0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02)),
                        pixels=blob))
    return out


def opening_axis_in_plane(op, rect):
    """Unit vector along an opening's long side, in camera-frame 3D."""
    a = np.radians(op['angle_deg'])
    # angle came from (x=axis1, y=axis0), so map it back that way round
    return np.cos(a) * rect['a2'] + np.sin(a) * rect['a1']


# --------------------------------------------------- single-opening solver

def opening_width_at(op, rect, centre_xyz, along, across, offset, span=0.010):
    """Width of the opening on a line offset sideways from its centre.

    Walks along `along` at a fixed sideways `offset` and counts how much of that
    line falls inside the blob, which is the opening's width there.
    """
    blob = op['pixels']
    n = max(int(2 * span / rect['step']), 5)
    inside = 0
    for d in np.linspace(-span, span, n):
        p = centre_xyz + along * d + across * offset
        i, j = camera_to_grid(rect, p)
        i, j = int(round(i)), int(round(j))
        if 0 <= i < blob.shape[0] and 0 <= j < blob.shape[1] and blob[i, j]:
            inside += 1
    return inside * (2 * span / (n - 1))


def rectify_image(rect, image, K):
    """Sample an image onto the rectified grid by inverse mapping.

    Carrying colour along with the depth points loses precisely the pixels that
    matter here: the socket's shell is bare metal, the sensor gets almost no
    return off it, and those pixels are therefore absent from the point cloud.
    Measured, the shell came through at 21mm2 against the 44mm2 it covers in the
    image -- less than half of it, and the missing half is the specular part.

    Going the other way round -- take each grid cell, put it on the plane, project
    it into the image and read what is there -- does not care whether depth
    succeeded at that spot.
    """
    h, w = rect['grid'].shape
    gi, gj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    s = rect['s0'] + gi * rect['step']
    t = rect['t0'] + gj * rect['step']
    p = (rect['origin'][None, None, :]
         + s[..., None] * rect['a1'][None, None, :]
         + t[..., None] * rect['a2'][None, None, :])
    z = p[..., 2]
    u = np.round(K[0, 0] * p[..., 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p[..., 1] / z + K[1, 2]).astype(int)
    ih, iw = image.shape[:2]
    ok = (z > 0) & (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
    out = np.full((h, w), np.nan)
    out[ok] = image[v[ok], u[ok]]
    return out


def find_shell(rect, min_area=2e-5):
    """Segment the socket's metal shell from the rectified greyscale channel.

    Depth cannot see the latch slot on this socket -- measured two ways, both at
    chance, with the CAD outline fitting the two headings within 0.04 IoU of each
    other. The slot is unmissable in greyscale though: the shell is bare metal on
    a black housing, a bimodal split Otsu picks out on its own.
    """
    vg = rect.get('value_grid')
    if vg is None:
        return None
    # Confine the search to the work, but by its *filled outline* rather than by
    # where depth succeeded. Requiring valid depth would exclude the shell all
    # over again -- bare metal is exactly what the sensor fails on.
    footprint = ndimage.binary_fill_holes(
        ndimage.binary_closing(np.isfinite(rect['grid']), np.ones((7, 7))))
    valid = np.isfinite(vg) & footprint
    if valid.sum() < 500:
        return None
    thr, _ = cv2.threshold(vg[valid].astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ndimage.binary_opening(valid & (vg > thr), np.ones((3, 3)))
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] * rect['step'] ** 2 < min_area:
        return None
    return lab == best


def resolve_flip_by_shell(shell, rect, op, min_ratio=1.25):
    """Settle the 180 deg heading from the metal shell's own shape.

    An RJ45 shell is a solid bar across one end and a pair of shoulders with the
    latch slot between them at the other, so its two halves look nothing alike.
    Measured in the rectified grid: 28.9mm2 of metal in the bar half against
    17.0mm2 in the slot half.

    The comparison is on how much metal each half holds. Extent along the long
    axis will not do it -- the side rails run the full length on both halves, so
    both measure ~14.5mm and the ratio sits at 1.01. Area separates them cleanly
    (measured 28.9mm2 against 17.0mm2), and a probe line placed at a fixed offset
    fails differently again: the bar is only a millimetre deep, so the line
    either misses it or leaves the shell altogether.

    The CAD puts the slot on mesh -Y, so the bar end is +Y. Returns
    (+1 | -1, ratio) or (0, ratio) when the halves are too alike to call.
    """
    ys, xs = np.where(shell)
    if len(ys) < 40:
        return 0, 0.0
    th = np.radians(op['angle_deg'])
    along = np.array([np.sin(th), np.cos(th)])
    across = np.array([-along[1], along[0]])

    rel = np.stack([ys - ys.mean(), xs - xs.mean()], axis=1)
    a = rel @ along                    # position along the shell's long axis
    b = rel @ across                   # which half

    areas = {sign: int((np.sign(b) == sign).sum()) for sign in (+1, -1)}
    hi, lo = max(areas.values()), max(min(areas.values()), 1)
    ratio = hi / lo
    if ratio < min_ratio:
        return 0, ratio
    return (+1 if areas[+1] > areas[-1] else -1), ratio


def resolve_flip_by_shape(op, rect, template, template_step, template_centre):
    """Settle the 180 deg heading by fitting the CAD outline, not by sampling it.

    The width-probe version reads the shape at a single offset and asks which
    side is longer. On this socket that is one number against a blob whose edges
    move by a cell or two per frame, and it decided correctly only ~83% of the
    time even with the work fully in view.

    Laying the whole CAD outline over the blob at both candidate headings and
    comparing overlap uses every pixel of the shape instead of one slice of it.
    The rectified grid is already metric and the plane is already known, so the
    only free choice left really is those two headings.

    Returns (+1 | -1, iou_best, iou_other).
    """
    blob = op['pixels']
    scale = template_step / rect['step']
    th = np.radians(op['angle_deg'])

    ys, xs = np.where(blob)
    pad = 8
    i0, i1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, blob.shape[0])
    j0, j1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, blob.shape[1])
    target = blob[i0:i1, j0:j1].astype(np.uint8)
    h, w = target.shape

    scores = {}
    for sign in (+1, -1):
        a = th + (0.0 if sign > 0 else np.pi)
        # template pixels are (row=template y, col=template x); place its centre
        # on the blob centroid and turn it to the candidate heading
        c, s_ = np.cos(a) * scale, np.sin(a) * scale
        M = np.array([[c, -s_, 0.0], [s_, c, 0.0]])
        cx_t, cy_t = template_centre
        M[0, 2] = (op['cj'] - j0) - (M[0, 0] * cx_t + M[0, 1] * cy_t)
        M[1, 2] = (op['ci'] - i0) - (M[1, 0] * cx_t + M[1, 1] * cy_t)
        warped = cv2.warpAffine(template.astype(np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST)
        inter = np.logical_and(warped, target).sum()
        union = np.logical_or(warped, target).sum()
        scores[sign] = inter / union if union else 0.0

    best = max(scores, key=scores.get)
    return best, scores[best], scores[-best]


def resolve_flip(op, rect, centre_xyz, probe):
    """Decide which end of the opening's long axis points along mesh +X.

    The long axis is a line, not a direction, so heading is only known mod
    180 deg, and for a socket the two are not interchangeable -- the latch has to
    enter on one particular side. What separates them is the *shape* of the
    opening rather than its depth: an RJ45 mouth is full width on one side of
    centre and tapers into the latch slot on the other (measured off the CAD:
    12.75mm against 7.25mm at 4mm either side of the centroid). Depth is nearly
    uniform across the whole opening, so comparing widths is both the stronger
    and the more honest signal.

    Both candidate directions are evaluated against their own Y axis in 3D, so
    nothing here depends on the handedness the plane basis came out with.

    Returns (+1 | -1, margin) or (0, margin) when the evidence is too weak to
    call -- the caller should refuse rather than guess.
    """
    z_axis = rect['normal']
    base = opening_axis_in_plane(op, rect)
    scores = {}
    for sign in (+1, -1):
        x_axis = base * sign
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        wide = opening_width_at(op, rect, centre_xyz, x_axis, y_axis, +probe['offset'])
        narrow = opening_width_at(op, rect, centre_xyz, x_axis, y_axis, -probe['offset'])
        scores[sign] = wide - narrow           # want the wide side at +Y

    sign = max(scores, key=scores.get)
    margin = scores[sign]
    if margin < probe['min_margin']:
        return 0, margin
    return sign, margin


def pick_opening(openings, expected_size, max_rel_error=0.45, prefer_near=None,
                 near_weight=0.02, max_centre_offset=None):
    """Choose the candidate that best matches a known opening size.

    Insisting on exactly one candidate is too brittle in practice: the work's
    top face comes back with a few no-return patches of its own, and each is
    indistinguishable from an opening until it is measured. The CAD already says
    how big the real one is, so use that.

    The tolerance is deliberately loose on the short axis -- the sensor loses the
    shallow end of the latch slot, so a real RJ45 mouth measures ~11.0 x 6.4mm
    against the CAD's 12.5 x 9.5mm.
    """
    if not openings:
        return None, 'no openings found'
    ew, eh = float(expected_size[0]), float(expected_size[1])
    scored = []
    for op in openings:
        w, h = op['size']
        err = abs(w - ew) / ew + abs(h - eh) / eh
        # Size alone is not decisive: the work's top face has no-return patches
        # of its own and one of them is regularly a similar size to the socket,
        # so the choice flickers between them. The real opening does not move
        # between frames, so penalise straying from where it was last seen.
        # Weighted in grid cells: a few cells of jitter costs almost nothing
        # (3 cells -> 0.06) while jumping to another blob does not (50 -> 1.0),
        # against a size error that runs around 0.45 for the genuine opening.
        if prefer_near is not None:
            err += near_weight * float(np.linalg.norm(
                np.array([op['ci'], op['cj']], dtype=float) - prefer_near))
        scored.append((err, op))
    # A hard gate, not just a penalty. The CAD puts the socket within a
    # millimetre of the jig's centre, so a candidate far from it is not a
    # marginal choice -- it is the wrong object, and accepting it moves the
    # reported pose enough to reset the flip vote and start the heading
    # oscillating. Better to publish nothing for a frame.
    if prefer_near is not None and max_centre_offset is not None:
        near = [(e, op) for e, op in scored
                if np.linalg.norm(np.array([op['ci'], op['cj']], dtype=float)
                                  - prefer_near) <= max_centre_offset]
        if not near:
            return None, (f'no candidate within {max_centre_offset:.0f} cells of '
                          f'the work centre ({len(scored)} rejected)')
        scored = near
    scored.sort(key=lambda s: s[0])
    err, best = scored[0]
    if err > max_rel_error * 2:
        return None, (f'best candidate {best["size"][0]*1000:.1f}x'
                      f'{best["size"][1]*1000:.1f}mm is too far from the expected '
                      f'{ew*1000:.1f}x{eh*1000:.1f}mm')
    return best, f'{len(openings)} candidates, picked one with size error {err:.2f}'


def solve_single_opening(rect, openings, mesh_opening_centroid, probe,
                         expected_size=None, force_sign=None, opening=None):
    """Pose of a part that has exactly one opening (the RJ45 jig).

    mesh_opening_centroid is where the *detected blob's centroid* sits in mesh
    coordinates -- not the insertion target. The two differ: an RJ45 blob
    includes the latch slot, which drags its centroid off the plug's centre. It
    has to be measured from the CAD with the same detector used here, or the
    offset shows up as a constant position error (see tools/build_reference.py).

    Returns T_camera_object for the *mesh origin*, matching what the
    FoundationPose route publishes, so nothing downstream has to change --
    arm_cmd keeps applying its own HOLE_CENTRE_IN_MESH to reach the socket.
    """
    # `opening` lets the caller hand in a selection it has already made under
    # constraints this function does not know about (the work-centre gate, say).
    # Re-deriving it here would quietly disagree with that choice, and the two
    # answers then show up in different parts of the same debug frame.
    if opening is not None:
        op, pick_note = opening, 'pre-selected'
    elif expected_size is None:
        if len(openings) != 1:
            return None, f'expected 1 opening, found {len(openings)}'
        op, pick_note = openings[0], ''
    else:
        op, pick_note = pick_opening(openings, expected_size)
        if op is None:
            return None, pick_note
    centroid_xyz = grid_to_camera(rect, op['ci'], op['cj'])

    sign, margin = resolve_flip(op, rect, centroid_xyz, probe)
    # force_sign lets the caller substitute a decision accumulated over several
    # frames: the per-frame evidence here is thin, but which way round the part
    # sits does not change between frames, so it should not be re-guessed each one
    if force_sign is not None:
        sign = force_sign
    if sign == 0:
        return None, f'flip undecidable (width margin {margin*1000:.2f}mm)'

    x_axis = opening_axis_in_plane(op, rect) * sign
    z_axis = rect['normal']   # mesh +Z points out of the opening, i.e. at the camera
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])

    # publish the mesh origin, not the opening, so this is a drop-in replacement
    # for what the FoundationPose route puts on the same topic
    origin = centroid_xyz - R @ np.asarray(mesh_opening_centroid)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    return T, dict(margin_mm=margin * 1000, size_mm=np.array(op['size']) * 1000,
                    chosen=op, note=pick_note, frame_sign=sign)


# ------------------------------------------------------- pattern registration

def debug_image(rect, openings, chosen=None, axis=None, axis_y=None, max_side=700):
    """Render the rectified surface with what the detector found drawn on it.

    Shows the view the algorithm actually works on -- straight-on and metric --
    rather than the raw camera image, so a bad plane fit or a mis-shaped blob is
    visible directly instead of having to be inferred from a pose that looks off.

    Grey = surface, dark = recessed or no return, green = accepted opening,
    red dot = centroid, yellow arrow = the +X direction the flip resolved to.
    """
    grid = rect['grid']
    finite = np.isfinite(grid)
    img = np.zeros(grid.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = 0.0, max(float(np.nanpercentile(grid[finite], 98)), 0.004)
        img[finite] = np.clip(255 - (grid[finite] - lo) / (hi - lo) * 220, 20, 255)
    vis = cv2.cvtColor(img.T, cv2.COLOR_GRAY2BGR)      # transpose: axis0 -> x

    for op in openings:
        mask = op['pixels'].T.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        picked = (chosen is not None and op is chosen)
        # Rejected candidates are drawn faintly and without a centre marker.
        # They are no-return patches on the work's own surface -- unavoidable in
        # the data, already excluded by the centre gate -- and marking them the
        # same as the real opening made the display look like the detector was
        # wavering when it was not.
        cv2.drawContours(vis, contours, -1,
                         (0, 220, 0) if picked else (70, 70, 70), 1)
        if picked:
            # the image was transposed above, so grid axis 0 is x, axis 1 is y
            cv2.circle(vis, (int(round(op['ci'])), int(round(op['cj']))),
                       3, (0, 0, 255), -1)

    if chosen is not None and axis is not None:
        # axis arrives as (along a1, along a2) = (grid axis 0, axis 1) = (x, y)
        c = np.array([chosen['ci'], chosen['cj']])
        ax = np.array(axis, dtype=float)
        ax /= max(np.linalg.norm(ax), 1e-9)
        tip = c + ax * (0.008 / rect['step'])
        cv2.arrowedLine(vis, tuple(c.astype(int)), tuple(tip.astype(int)),
                        (0, 235, 235), 2, tipLength=0.3)
        # +Y as well, because that is the claim worth checking by eye: the flip
        # was resolved by putting the opening's wide side on +Y, so if this arrow
        # does not point into the wider half of the blob, the call was wrong
        perp = np.array(axis_y, dtype=float) if axis_y is not None \
            else np.array([-ax[1], ax[0]])
        perp /= max(np.linalg.norm(perp), 1e-9)
        tip_y = c + perp * (0.006 / rect['step'])
        cv2.arrowedLine(vis, tuple(c.astype(int)), tuple(tip_y.astype(int)),
                        (0, 140, 255), 2, tipLength=0.3)      # BGR: orange
        cv2.putText(vis, '+Y -> wide side', (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

    scale = min(max_side / max(vis.shape[:2]), 4.0)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    return vis


def kabsch_2d(P, Q):
    """Least-squares rigid transform taking P onto Q, in closed form.

    SVD-based, so there is no iteration and no initial guess to get wrong. The
    determinant fix stops a reflection being returned when the point sets are
    nearly degenerate.
    """
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, cQ - R @ cP


def register_pattern(detected_xy, cad_xy, tol=0.003, min_inliers=4):
    """Match a detected opening pattern to the CAD one, correspondence unknown.

    Kabsch needs to be told which point pairs with which, and we are not told.
    So hypothesise a pairing from two points, solve in closed form, and score it
    by how many of the remaining CAD openings land on a detected one. Pairs whose
    separations disagree cannot correspond, which prunes most of the search
    before any arithmetic happens.

    An asymmetric pattern makes the correct hypothesis win outright rather than
    by a hair -- the 180 deg-rotated alternative strands nearly every point.
    """
    if len(detected_xy) < 2 or len(cad_xy) < 2:
        return None
    d_len = {(i, j): np.linalg.norm(detected_xy[i] - detected_xy[j])
             for i in range(len(detected_xy)) for j in range(i + 1, len(detected_xy))}
    c_len = {(a, b): np.linalg.norm(cad_xy[a] - cad_xy[b])
             for a in range(len(cad_xy)) for b in range(len(cad_xy)) if a != b}

    best = None
    for (i, j), dij in d_len.items():
        for (a, b), dab in c_len.items():
            if abs(dij - dab) > tol:
                continue
            R, t = kabsch_2d(cad_xy[[a, b]], detected_xy[[i, j]])
            moved = cad_xy @ R.T + t
            dist = np.linalg.norm(moved[:, None, :] - detected_xy[None, :, :], axis=2)
            nearest = dist.argmin(axis=1)
            inl = dist[np.arange(len(cad_xy)), nearest] < tol
            if best is None or inl.sum() > best['n_inliers']:
                best = dict(R=R, t=t, pairing=nearest, inlier_mask=inl,
                            n_inliers=int(inl.sum()))

    if best is None or best['n_inliers'] < min_inliers:
        return None

    # final fit on every correspondence that survived, not just the seed pair
    m = best['inlier_mask']
    R, t = kabsch_2d(cad_xy[m], detected_xy[best['pairing'][m]])
    moved = cad_xy @ R.T + t
    resid = np.linalg.norm(moved - detected_xy[best['pairing']], axis=1)
    best.update(R=R, t=t, residual_m=resid, mean_residual_m=float(resid[m].mean()))
    return best


def solve_opening_pattern(rect, openings, cad_table, tol=0.003):
    """Pose of a part with several openings (the multi-port platform).

    Heading falls out of the pattern match, so there is no flip to resolve --
    that ambiguity only exists when a single opening has to speak for the whole
    part.
    """
    if len(openings) < 2:
        return None, f'need >=2 openings, found {len(openings)}'

    det = np.array([[rect['s0'] + o['ci'] * rect['step'],
                     rect['t0'] + o['cj'] * rect['step']] for o in openings])
    cad = np.array([p['centre'][:2] for p in cad_table])

    fit = register_pattern(det, cad, tol=tol)
    if fit is None:
        return None, (f'pattern did not register '
                      f'({len(det)} detected vs {len(cad)} in CAD)')

    # the 2D fit is expressed in the plane basis; lift it back into 3D
    R2, t2 = fit['R'], fit['t']
    x_axis = R2[0, 0] * rect['a1'] + R2[1, 0] * rect['a2']
    y_axis = R2[0, 1] * rect['a1'] + R2[1, 1] * rect['a2']
    z_axis = rect['normal']
    x_axis /= np.linalg.norm(x_axis)
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    R = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])

    origin = rect['origin'] + t2[0] * rect['a1'] + t2[1] * rect['a2']
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    return T, dict(n_inliers=fit['n_inliers'], n_cad=len(cad),
                   mean_residual_mm=fit['mean_residual_m'] * 1000,
                   pairing=fit['pairing'], detected_xy=det)
