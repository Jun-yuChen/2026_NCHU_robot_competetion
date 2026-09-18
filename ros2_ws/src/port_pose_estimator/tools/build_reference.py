"""Derive depth_pose_lib's reference constants from a CAD model.

The detector reports the centroid of the blob it segments, and for a socket that
blob includes the latch slot -- so its centroid is *not* the middle of the plug
opening. Any constant taken from a drawing instead of from the same segmentation
turns into a fixed position error. This measures it the same way the runtime
does, by ray-casting the port face and running the identical threshold and
moment code.

Also reports the width profile across the opening, which is what tells the two
180-deg candidate headings apart, so the probe offset can be chosen with the
actual numbers in front of you.

Run under the foundationpose env (needs trimesh):

    ~/.local/bin/micromamba run -n foundationpose python tools/build_reference.py
"""
import json
import os
import sys
import numpy as np
import trimesh
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from port_pose_estimator import depth_pose_lib as dpl

# Models live in the package's own cad/ directory so a fresh clone can rebuild
# the port table without hunting for files. A path outside the package is still
# accepted -- pass one on the command line -- but nothing here depends on one.
CAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cad')
MESHES = {
    'rj45_test': os.path.join(CAD_DIR, 'rj45_test.obj'),
    'server1': os.path.join(CAD_DIR, 'server1_all.STL'),
    'case1': os.path.join(CAD_DIR, 'case1_io_shield.STL'),
}

# Ports are told apart by how deep their cavity is -- a far cleaner signal than
# opening size, which only separates them by a millimetre or so. Measured on
# server1_all: USB 8.6mm, RJ45 12.8mm, HDMI 6.4mm.
PORT_TYPES = [('hdmi', 0.0064), ('usb', 0.0086), ('rj45', 0.0128)]

# That rule is specific to a model whose every cavity is closed. A real I/O
# shield is mostly through-holes: only the sockets actually being inserted into
# were modelled with an interior, and the rest are openings cut through a 1mm
# plate. Depth therefore separates nothing, and the classification falls back to
# the opening's own proportions:
#
#   RJ45   14.75 x 12.75  -- far wider across than anything else here
#   HDMI   16.75 x  7.50  -- half a millimetre wider than a USB-A, but its
#                            cavity is 6.3mm against USB's 7.5, which does split
#   USB-A  14-17  x  6.75-7.00
#
# A through-hole is never an insertion target on this panel -- it was left
# hollow precisely because nothing goes into it -- so those become 'other'.
# They still belong in the table: the pattern match needs them. Dropping the
# eleven decoys would leave the six targets inside a 20mm cluster, and a
# coplanar PnP over a baseline that short is not worth solving.
# Measured on this shield: RJ45 10.75 across, HDMI 7.50, USB-A 6.75-7.00.
# Read off the panel's own silkscreen, top to bottom. Only the six that are
# actually inserted into carry a real kind; everything else is 'other', which
# keeps it in the table for the pattern without offering it as a target.
# Coordinates are mm in the recentred mesh and only have to be nearer the port
# they name than any other -- the tightest spacing here is 8.9mm.
KIND_OVERRIDE = {
    'case1': [
        (-0.7,  66.7, 'usb'),    (8.1,  66.7, 'usb'),     # USB 2.0 pair
        (-11.5, 66.0, 'other'),                            # PS/2
        (-11.5, 51.5, 'other'),                            # SPDIF coax
        (-13.5, 24.5, 'other'),  (7.1,  25.0, 'other'),    # DVI, VGA
        (-11.3, -1.5, 'usb'),                              # USB DAC-UP (thru)
        (-15.3, -19.5, 'hdmi'),                            # HDMI
        (-1.3, -19.5, 'usb'),    (7.5,  -19.5, 'usb'),     # USB 3.0 pair
        (-15.2, -43.0, 'usb'),   (-6.6, -43.0, 'usb'),     # USB 3.0 pair
        (5.6,  -43.0, 'rj45'),                             # RJ45
        (-12.0, -61.0, 'other'),                           # optical SPDIF
        (-0.3, -61.0, 'other'),  (9.8,  -61.0, 'other'),
        (-11.0, -73.3, 'other'), (-0.3, -73.3, 'other'),
        (9.8,  -73.3, 'other'),                            # audio jacks
    ],
}

# Names normally follow reading order, which is what makes them predictable
# from the panel itself. The DAC-UP socket is the exception: it sits between
# the top USB pair and the USB 3.0 row, so numbering it in place would make it
# usb3 and push the four sockets below it up one each -- renaming ports that
# are already referred to by number in task files, notes and this repo's own
# tests. Naming it after them keeps every other port where it was.
NAME_OVERRIDE = {
    'case1': [(-11.3, -1.5, 'usb7')],
}
NAME_OVERRIDE_TOL_MM = 5.0      # min spacing on this panel is 8.9mm
# Upper bound raised from 300mm2 once a DVI socket turned up at 307 and was
# silently dropped -- losing a port costs baseline, which is what a coplanar
# PnP is most short of. The handle slot this bound exists to exclude is
# 1092mm2, so there is plenty of room between the two.
PORT_AREA_RANGE = (2e-5, 5e-4)
OUT = os.path.join(os.path.dirname(__file__), '..', 'config', 'opening_reference.json')


def face_plane(D, bins=0.00025):
    """How far the part's own front face sits below the bounding box's top.

    The original code took the bounding box's top *as* the face, which holds for
    a panel whose ports are cut straight into its front. It does not hold once
    the model carries a raised border: an I/O shield's stamped frame stands
    2.25mm proud, and measuring recess from that reads the whole panel as one
    2mm-deep pocket -- 19 ports segment as a single 6161mm2 blob.

    The face is whatever depth the largest flat area sits at, i.e. the mode.
    """
    d = D[np.isfinite(D)]
    if not len(d):
        return 0.0
    h, edges = np.histogram(d, bins=np.arange(0, float(d.max()) + bins, bins))
    i = int(np.argmax(h))
    # the mean of the samples in that bin, not the bin's centre: a part whose
    # front *is* the bounding box top must come back exactly 0, or every port
    # shifts by half a bin and the older models stop reproducing
    inside = d[(d >= edges[i]) & (d < edges[i + 1])]
    return float(inside.mean()) if len(inside) else float(edges[i])


def face_grid(mesh, step=0.00025, pad=0.002):
    """Ray-cast the opening face from straight on -> recess grid in mesh coords."""
    lo, hi = mesh.bounds
    z_top = hi[2]
    xs = np.arange(lo[0] - pad, hi[0] + pad, step)
    ys = np.arange(lo[1] - pad, hi[1] + pad, step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_top + 0.005)], axis=1)
    loc, idx, _ = mesh.ray.intersects_location(
        o, np.tile([0, 0, -1.0], (len(o), 1)), multiple_hits=False)
    D = np.full(len(o), np.nan)
    D[idx] = z_top - loc[:, 2]
    return D.reshape(X.shape), xs, ys, z_top


# How far below the part's face counts as an opening, per model.
#
# 2mm is the original rule and it stays the default, because it is what the
# hardware-validated server1 table was built with. Dropping it to 0.5mm there
# moves four ports by 0.7mm and grows every USB opening by 1mm, and while the
# reprojection error does not budge -- 1.85px either way -- test_mono's check
# against the physical panel falls from 8/8 correct labels to 2/8. server1 is
# only 22.6% asymmetric under a half turn, so a 0.7mm shift is enough to tip
# which of the two readings wins. A table that reprojects identically and
# labels the panel backwards is the exact failure this project keeps meeting.
#
# The I/O shield needs the finer threshold for the opposite reason: at 2mm the
# sockets' own internal structure -- USB tongue, HDMI tongue, RJ45 latch slot --
# sits above the cut, so each opening segments as a ring or a slotted rectangle
# and its centroid is not the middle of the hole. Measured: HDMI 1.34mm off and
# split in two, RJ45 0.70mm off. At 0.5mm the openings fill in and centroid and
# bounding-box centre agree to 0.12mm across all nineteen.
RECESS = {'case1': 0.0005}
DEFAULT_RECESS = 0.002

# Runtime hints for depth_pose_node, merged into the model's entry below.
# Absent for every other model, so their pipeline is untouched.
#
# case1 sits in front of whatever else is in shot -- a chair, boxes,
# shelving -- comparably dark and touching it in frame, which is what
# platform_from: "depth" is for (see mono_pose_lib.platform_from_depth):
# gate the depth stream to the near cluster instead of thresholding
# brightness over the whole image.
#
# port_ladder is the (block, offset) pairs find_ports_ladder retries -- see
# mono_pose_lib.find_ports_ladder for why this panel needs block retried
# directly rather than through find_ports's scale ladder. Ordered by how
# often each won during validation on real captures (case106-109), so the
# common case costs one attempt.
#
# min_pairs=6: match_ports's default demands 0.75 x len(ports) = 14 of 19,
# tuned against server1's 8-port table (0.75 x 8 = 6). case1's table carries
# eleven unlabelled "other" ports alongside the eight usb/hdmi/rj45 ones the
# task actually targets, which inflates that fraction's denominator without
# making any of them easier to detect -- 14/19 was never reached in
# validation. 6 matches server1's own effective bar.
PLATFORM_OVERRIDE = {
    'case1': dict(
        platform_from='depth',
        port_ladder=[[13, -12], [21, -9], [25, -9], [17, -12], [9, -6],
                     [27, -20], [15, -9], [31, -6], [11, -9], [19, -9],
                     [23, -12], [29, -9], [35, -9]],
        min_pairs=6,
    ),
}


def segment(D, step, face=0.0, recess=DEFAULT_RECESS):
    """Same masking rule as depth_pose_lib.find_openings, from the part's face.

    `face` is 0 for a model whose front *is* the top of its bounding box, which
    is every part this tool handled before the I/O shield arrived.
    """
    solid = np.isfinite(D)
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(solid, np.ones((7, 7))))
    # 0.5mm below the face, not 2mm. At 2mm the threshold skips a socket's own
    # internal structure -- a USB tongue, an HDMI tongue, an RJ45 latch slot all
    # sit shallower than that -- so the blob comes out as a ring or a slotted
    # rectangle, and the centroid of a ring with a notch in it is not the middle
    # of the opening. Measured on the I/O shield: the HDMI opening split into
    # two fragments and its recorded centre landed 1.34mm off; the RJ45's latch
    # cutout pulled its centre 0.70mm sideways. Including the internal structure
    # fills the opening back in, after which centroid and bounding-box centre
    # agree to 0.12mm across all nineteen ports.
    mask = (foot & ~solid) | (solid & (D > face + recess))
    if recess < DEFAULT_RECESS:
        # the openings are filled in now; close the one-cell gaps left where a
        # tongue meets the shell so each port stays a single blob
        mask = ndimage.binary_closing(mask, np.ones((3, 3)))
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def classify(name, x_m, y_m, depth):
    """Port kind. -> kind string.

    Depth alone decides it on a model whose every cavity is closed, and that is
    the cleaner signal where it applies. It does not apply to a real I/O shield:
    most of its openings are cut clean through a 1mm plate, so they have no
    depth at all, and the ones that do overlap the older model's bands -- this
    shield's RJ45 measures 12.97mm deep against server1's HDMI at 12.68.

    No geometric rule separates those, so the shield's kinds are stated
    outright. They are read off the silkscreen in a minute and never change,
    which is a better trade than a threshold that quietly reclassifies a
    different panel later.
    """
    table = KIND_OVERRIDE.get(name)
    if table is not None:
        return min(table, key=lambda t: (t[0] - x_m * 1000) ** 2
                                        + (t[1] - y_m * 1000) ** 2)[2]
    return min(PORT_TYPES, key=lambda t: abs(t[1] - depth))[0]


def structure_centroid(blob, D, face, recess, xs, ys, min_cluster=15):
    """Centre of a port's own raised/recessed structure, not the whole cavity.

    A real socket's opening is not one uniform depth: a USB or HDMI tongue, or
    an RJ45 latch slot, sits at its own depth inside a larger cavity, and that
    cavity is usually the bigger area. Taking the centroid of the whole blob
    (cavity plus structure plus any through-hole) weights it toward the cavity,
    which measured 0.4-1.4mm off the structure's own centre across this
    shield's eight target ports -- enough on its own to matter.

    So: isolate the shallow band (between `recess` and 2mm below the face,
    where a tongue or latch typically sits) and centre on that instead.

    Restricted to the single largest connected cluster in that band, because a
    ray grazing a slot's end wall at a shallow angle can read a depth in this
    band from a handful of pixels nowhere near the real structure -- measured
    on usb4, 3 cells (0.16% of the blob) at the two far ends of an elongated
    slot, versus 1826 cells forming the actual tongue. `min_cluster` drops
    anything smaller than a real structure could be.

    Falls back to the whole blob's centroid if no shallow band survives the
    filter -- a through-hole with no modelled internal structure has nothing
    else to centre on.
    """
    shallow = blob & (D > face + recess) & (D <= face + 0.002)
    lab, n = ndimage.label(shallow)
    if n == 0:
        xi, yi = np.where(blob)
        return xs[xi].mean(), ys[yi].mean()
    sizes = ndimage.sum(shallow, lab, range(1, n + 1))
    big = [i + 1 for i, sz in enumerate(sizes) if sz >= min_cluster]
    if not big:
        xi, yi = np.where(blob)
        return xs[xi].mean(), ys[yi].mean()
    keep = np.isin(lab, big)
    xi, yi = np.where(keep)
    return xs[xi].mean(), ys[yi].mean()


def refined_centre(blob, D, face, recess, xs, ys, compact=0.85):
    """The port's centre, per axis. -> (x, y) in the same units as xs/ys.

    Weighted centroid of the whole blob, by default -- this is what fixing the
    recess threshold to 0.5mm was for: with the tongue or latch slot no longer
    carving a notch out of the opening, the blob is one complete, essentially
    convex shape and its centroid sits where a person would call the middle.

    One axis overridden where it does not. A socket's shallow band (0.5-2mm --
    the tongue, or the wall just inside the rim) is not always centred on the
    same thing the whole blob is: for a USB or HDMI tongue it is a compact
    strip running down the middle, well inside the outer shell, and the shell
    itself is not left-right symmetric -- a mounting tab or ground clip extends
    it further on one side. Weighting by the whole shell then pulls the centre
    toward that extra material, 0.6-0.75mm off the tongue's own midpoint.
    RJ45 does not have this problem: its shallow band is a rim that runs almost
    the full width of the opening, not a compact feature, and centring on that
    would be centring on the rim rather than the socket.

    The two are told apart by how much of the whole blob's own width the
    shallow band's bounding box covers on that axis. Below `compact` (measured:
    0.72-0.81 for six USB/HDMI ports here, 0.90 for the RJ45) the shallow band
    is trusted and its own bounding-box midpoint used, which does not suffer
    from the tongue's internal density being asymmetric the way a weighted
    centroid does. At or above it, the whole blob's centroid stands.
    """
    xi, yi = np.where(blob)
    cx, cy = xs[xi].mean(), ys[yi].mean()
    wx0, wx1 = xs[xi].min(), xs[xi].max()
    wy0, wy1 = ys[yi].min(), ys[yi].max()

    shallow = blob & (D > face + recess) & (D <= face + 0.002)
    sxi, syi = np.where(shallow)
    if len(sxi) == 0:
        return cx, cy

    sx0, sx1 = xs[sxi].min(), xs[sxi].max()
    if (wx1 - wx0) > 0 and (sx1 - sx0) / (wx1 - wx0) < compact:
        cx = (sx0 + sx1) / 2
    sy0, sy1 = ys[syi].min(), ys[syi].max()
    if (wy1 - wy0) > 0 and (sy1 - sy0) / (wy1 - wy0) < compact:
        cy = (sy0 + sy1) / 2
    return cx, cy


def merge_split(ports, min_gap=0.004):
    """Fold blobs that are really one opening. -> ports.

    An HDMI socket's raised tongue sits shallower than the recess threshold and
    cuts its opening into two blobs a millimetre apart. Genuine neighbours on
    this shield are 8.9mm apart at the tightest, so anything closer than 4mm is
    one port, not two.
    """
    out = []
    for p in sorted(ports, key=lambda q: -q['size'][0] * q['size'][1]):
        c = np.array(p['centre'][:2])
        if any(np.linalg.norm(c - np.array(q['centre'][:2])) < min_gap for q in out):
            continue
        out.append(p)
    return out


def analyse_ports(name, mesh, D, xs, ys, z_top, step, face=0.0,
                  recess=DEFAULT_RECESS):
    """Multi-opening part -> a table of named ports.

    Where a single-opening part has to squeeze a heading out of one small blob,
    a panel of ports carries it in the pattern itself, so what is needed here is
    just each port's centre, long axis and identity. Sizes barely separate the
    types (a millimetre between USB and HDMI), but cavity depth does, cleanly.
    """
    recess = RECESS.get(name, DEFAULT_RECESS)
    lab, n = ndimage.label(segment(D, step, face, recess))
    cell = step * step
    ports = []
    for i in range(1, n + 1):
        blob = lab == i
        area = blob.sum() * cell
        if not (PORT_AREA_RANGE[0] <= area <= PORT_AREA_RANGE[1]):
            continue                      # skips noise and the handle slot
        xi, yi = np.where(blob)
        d = D[blob]
        d = d[np.isfinite(d)]
        # depth below the part's own face, NaN where the ray went clean through
        depth = float(d.mean() - face) if len(d) else float('nan')
        ex = (xi.max() - xi.min() + 1) * step
        ey = (yi.max() - yi.min() + 1) * step
        # Centre on the whole opening -- cavity, internal structure and all --
        # as one connected region. The recess fix (0.5mm not 2mm) is what makes
        # this correct: it is what stops a tongue or latch slot splitting the
        # blob or carving a notch out of it, which is what put the centre off
        # by 0.4-1.4mm in the first place. A further attempt to isolate just
        # the shallow band and centre on that alone was tried and reverted --
        # measured on usb1, the shallow band's own area is asymmetric along its
        # length (an uneven comb of contact-pin cutouts), and weighting by that
        # area pulled the centre 1mm off the shape's own middle. The whole,
        # unsplit blob does not have this problem.
        cix, ciy = refined_centre(blob, D, face, recess, xs, ys)
        kind = classify(name, xs[xi].mean(), ys[yi].mean(), depth)
        ports.append(dict(
            kind=kind,
            centre=[float(cix), float(ciy),
                    float(z_top - face)],
            size=[float(max(ex, ey)), float(min(ex, ey))],
            long_axis=[1.0, 0.0, 0.0] if ex >= ey else [0.0, 1.0, 0.0],
            # +1 or -1, which end of long_axis the plug enters from. The CAD
            # cannot say -- these cavities carry no latch detail -- so it starts
            # at +1 and is corrected by hand once, from the real hardware.
            flip=1,
            depth=depth,
        ))

    # reading order: rows down the panel, then across each row
    ports.sort(key=lambda p: (-round(p['centre'][1], 3), p['centre'][0]))
    forced = NAME_OVERRIDE.get(name, [])
    taken = set()
    for p in ports:
        if not forced:
            break
        x, y = p['centre'][0] * 1000, p['centre'][1] * 1000
        fx, fy, nm = min(forced, key=lambda t: (t[0] - x) ** 2 + (t[1] - y) ** 2)
        if (fx - x) ** 2 + (fy - y) ** 2 <= NAME_OVERRIDE_TOL_MM ** 2:
            p['name'] = nm
            taken.add(nm)
    count = {}
    for p in ports:
        if 'name' in p:
            continue                      # named outright, see NAME_OVERRIDE
        while True:
            count[p['kind']] = count.get(p['kind'], 0) + 1
            candidate = f"{p['kind']}{count[p['kind']]}"
            if candidate not in taken:
                break                     # never hand out a forced name twice
        p['name'] = candidate

    print(f'--- {name} ---')
    print(f'  {len(ports)} ports (of {n} segmented regions; the rest are noise '
          f'or the handle)')
    print(f"  {'name':8s} {'x(mm)':>8s} {'y(mm)':>8s} {'size(mm)':>15s} {'depth':>7s} {'long':>5s}")
    for p in ports:
        print(f"  {p['name']:8s} {p['centre'][0]*1000:8.2f} {p['centre'][1]*1000:8.2f} "
              f"  {p['size'][0]*1000:6.2f} x {p['size'][1]*1000:5.2f} "
              f"{p['depth']*1000:7.2f} {'X' if p['long_axis'][0] else 'Y':>5s}"
              if np.isfinite(p['depth']) else
              f"  {p['name']:8s} {p['centre'][0]*1000:8.2f} {p['centre'][1]*1000:8.2f} "
              f"  {p['size'][0]*1000:6.2f} x {p['size'][1]*1000:5.2f} "
              f"{'thru':>7s} {'X' if p['long_axis'][0] else 'Y':>5s}")

    # the pattern is what fixes the heading, so how unlike itself the panel is
    # under a half turn is the number that decides whether this will work at all
    Dr = D[::-1, ::-1]
    has, hasr = np.isfinite(D), np.isfinite(Dr)
    both = has & hasr
    sil = 100 * np.logical_xor(has, hasr).sum() / has.sum()
    thr = 0.02 * float(np.linalg.norm(mesh.extents))
    dep = 100 * (np.abs(D[both] - Dr[both]) > thr).mean() * both.sum() / has.sum()
    print(f'  180deg asymmetry        : {sil + dep:.1f}%  '
          f'(the RJ45 jig, which flips at random, is 4.4%)')

    return dict(
        work_size_m=[float(mesh.extents[0]), float(mesh.extents[1])],
        work_height_m=float(mesh.extents[2]),
        ports=ports,
        asymmetry_pct=float(sil + dep),
    )


def load_in_metres(path):
    """Load a mesh, converting from millimetres when that is plainly the unit.

    The pipeline works in metres throughout, but SolidWorks exports STL in
    millimetres and nothing in the file says so. A part whose longest side comes
    out over a metre is not a connector panel, it is a mm file being read as m --
    and left uncaught it asks numpy for a 2TB grid.
    """
    mesh = trimesh.load(path, force='mesh')
    if float(np.max(mesh.extents)) > 1.0:
        mesh.apply_scale(0.001)
        print(f'  (interpreted as millimetres, scaled to metres)')
    return mesh


def analyse(name, path):
    mesh = load_in_metres(path)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    step = 0.00025
    D, xs, ys, z_top = face_grid(mesh, step)
    face = face_plane(D)
    if face > 0.0005:
        print(f'{name}: front face sits {face*1000:.2f} mm below the bounding '
              f'box top (raised border) -- measuring recess from there')
    recess = RECESS.get(name, DEFAULT_RECESS)
    lab, n = ndimage.label(segment(D, step, face, recess))
    if n == 0:
        print(f'{name}: no opening found')
        return None

    # A part with many openings takes the pattern route; the single-opening
    # analysis below exists only for parts that cannot.
    cell = step * step
    n_ports = sum(1 for i in range(1, n + 1)
                  if PORT_AREA_RANGE[0] <= (lab == i).sum() * cell <= PORT_AREA_RANGE[1])
    if n_ports >= 3:
        return analyse_ports(name, mesh, D, xs, ys, z_top, step, face, recess)

    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    blob = lab == (int(np.argmax(sizes)) + 1)
    xi, yi = np.where(blob)
    cx, cy = xs[xi].mean(), ys[yi].mean()

    # long axis: the opening's wider extent
    ex = xs[xi].max() - xs[xi].min()
    ey = ys[yi].max() - ys[yi].min()
    long_axis = [1.0, 0.0, 0.0] if ex >= ey else [0.0, 1.0, 0.0]

    print(f'--- {name} ---')
    print(f'  openings segmented      : {n}')
    print(f'  blob centroid (mesh, mm): ({cx*1000:+.2f}, {cy*1000:+.2f}, {z_top*1000:+.2f})')
    print(f'  blob extent x/y (mm)    : {ex*1000:.2f} x {ey*1000:.2f}')
    print(f'  long axis in mesh       : {long_axis}')

    # width profile across the short axis, measured from the centroid
    print('  width across the opening, relative to the centroid:')
    rows = []
    for off in np.arange(-0.006, 0.00601, 0.001):
        j = int(round((cy + off - ys[0]) / step))
        w = blob[:, j].sum() * step if 0 <= j < blob.shape[1] else 0.0
        rows.append((off, w))
        print(f'     {off*1000:+5.1f} mm : {w*1000:6.2f} mm')

    # pick the probe offset where the two sides disagree most
    best_off, best_gap = 0.0, 0.0
    for off, w in rows:
        if off <= 0:
            continue
        j = int(round((cy - off - ys[0]) / step))
        w_neg = blob[:, j].sum() * step if 0 <= j < blob.shape[1] else 0.0
        if abs(w - w_neg) > best_gap:
            best_off, best_gap = off, abs(w - w_neg)
    print(f'  best flip probe offset  : {best_off*1000:.1f} mm '
          f'(width gap {best_gap*1000:.2f} mm)')

    # The flip probe cannot be taken from the CAD. What the detector segments is
    # not the CAD opening: the sensor loses the shallow end of the latch slot,
    # and no-return patches next to the socket merge into the same blob and fill
    # the taper back in. Measured on the D405 at 1280x720 and ~140mm, through the
    # real pipeline, the width across the blob runs
    #     -3.0mm 5.3 | -2.0mm 8.4 | 0mm 10.0 | +2.0mm 11.3 | +3.0mm 2.8
    # so the usable separation is ~2.8mm at +/-2.0mm, against the CAD's 5.5mm at
    # +/-3.0mm. Probing where the CAD says leaves both samples on the blob's edge
    # and the margin collapses to under 1mm.
    HARDWARE_PROBE_OFFSET = 0.002
    HARDWARE_MIN_MARGIN = 0.0012
    print(f'  CAD-optimal offset      : {best_off*1000:.1f} mm (gap {best_gap*1000:.2f} mm)')
    print(f'  using hardware-measured : {HARDWARE_PROBE_OFFSET*1000:.1f} mm '
          f'(min margin {HARDWARE_MIN_MARGIN*1000:.1f} mm)')
    # Save the opening's actual outline, not just scalars taken off it. Deciding
    # the 180deg heading from a single width sample throws away the rest of the
    # shape, and on a blob this noisy that one number is not enough. With the
    # rectified grid already metric, the template can be laid straight onto a
    # detected blob and the two orientations compared by overlap.
    ty, tx = np.where(blob)
    tmpl = blob[ty.min():ty.max() + 1, tx.min():tx.max() + 1]
    tmpl_centre = [float(cx - xs[tx.min()]) / step, float(cy - ys[ty.min()]) / step]
    np.save(os.path.join(os.path.dirname(OUT), f'{name}_opening_mask.npy'), tmpl)
    print(f'  saved outline template   : {tmpl.shape[1]}x{tmpl.shape[0]} cells '
          f'at {step*1000:.2f} mm/cell')

    return dict(
        work_size_m=[float(mesh.extents[0]), float(mesh.extents[1])],
        opening_template=f'{name}_opening_mask.npy',
        opening_template_step=float(step),
        opening_template_centre=tmpl_centre,
        opening_centroid_in_mesh=[float(cx), float(cy), float(z_top)],
        opening_size_m=[float(ex), float(ey)],
        long_axis_in_mesh=long_axis,
        flip_probe=dict(offset=HARDWARE_PROBE_OFFSET,
                        min_margin=HARDWARE_MIN_MARGIN),
    )


def main():
    out = {}
    for name, path in MESHES.items():
        if not os.path.exists(path):
            print(f'{name}: missing {path}')
            continue
        r = analyse(name, path)
        if r:
            r.update(PLATFORM_OVERRIDE.get(name, {}))
            out[name] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
