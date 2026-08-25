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
from py_gripper import depth_pose_lib as dpl

# Models live in the package's own cad/ directory so a fresh clone can rebuild
# the port table without hunting for files. A path outside the package is still
# accepted -- pass one on the command line -- but nothing here depends on one.
CAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cad')
MESHES = {
    'rj45_test': os.path.join(CAD_DIR, 'rj45_test.obj'),
    'server1': os.path.join(CAD_DIR, 'server1_all.STL'),
}

# Ports are told apart by how deep their cavity is -- a far cleaner signal than
# opening size, which only separates them by a millimetre or so. Measured on
# server1_all: USB 8.6mm, RJ45 12.8mm, HDMI 6.4mm.
PORT_TYPES = [('hdmi', 0.0064), ('usb', 0.0086), ('rj45', 0.0128)]
PORT_AREA_RANGE = (2e-5, 3e-4)      # excludes the handle slot (1092mm2)
OUT = os.path.join(os.path.dirname(__file__), '..', 'config', 'opening_reference.json')


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


def segment(D, step):
    """Same masking rule as depth_pose_lib.find_openings."""
    solid = np.isfinite(D)
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(solid, np.ones((7, 7))))
    mask = (foot & ~solid) | (solid & (D > 0.002))
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def analyse_ports(name, mesh, D, xs, ys, z_top, step):
    """Multi-opening part -> a table of named ports.

    Where a single-opening part has to squeeze a heading out of one small blob,
    a panel of ports carries it in the pattern itself, so what is needed here is
    just each port's centre, long axis and identity. Sizes barely separate the
    types (a millimetre between USB and HDMI), but cavity depth does, cleanly.
    """
    lab, n = ndimage.label(segment(D, step))
    cell = step * step
    ports = []
    for i in range(1, n + 1):
        blob = lab == i
        area = blob.sum() * cell
        if not (PORT_AREA_RANGE[0] <= area <= PORT_AREA_RANGE[1]):
            continue                      # skips noise and the handle slot
        xi, yi = np.where(blob)
        depth = float(np.nanmean(D[blob]))
        kind = min(PORT_TYPES, key=lambda t: abs(t[1] - depth))[0]
        ex = (xi.max() - xi.min() + 1) * step
        ey = (yi.max() - yi.min() + 1) * step
        ports.append(dict(
            kind=kind,
            centre=[float(xs[xi].mean()), float(ys[yi].mean()), float(z_top)],
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
    count = {}
    for p in ports:
        count[p['kind']] = count.get(p['kind'], 0) + 1
        p['name'] = f"{p['kind']}{count[p['kind']]}"

    print(f'--- {name} ---')
    print(f'  {len(ports)} ports (of {n} segmented regions; the rest are noise '
          f'or the handle)')
    print(f"  {'name':8s} {'x(mm)':>8s} {'y(mm)':>8s} {'size(mm)':>15s} {'depth':>7s} {'long':>5s}")
    for p in ports:
        print(f"  {p['name']:8s} {p['centre'][0]*1000:8.2f} {p['centre'][1]*1000:8.2f} "
              f"  {p['size'][0]*1000:6.2f} x {p['size'][1]*1000:5.2f} "
              f"{p['depth']*1000:7.2f} {'X' if p['long_axis'][0] else 'Y':>5s}")

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
    lab, n = ndimage.label(segment(D, step))
    if n == 0:
        print(f'{name}: no opening found')
        return None

    # A part with many openings takes the pattern route; the single-opening
    # analysis below exists only for parts that cannot.
    cell = step * step
    n_ports = sum(1 for i in range(1, n + 1)
                  if PORT_AREA_RANGE[0] <= (lab == i).sum() * cell <= PORT_AREA_RANGE[1])
    if n_ports >= 3:
        return analyse_ports(name, mesh, D, xs, ys, z_top, step)

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
            out[name] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
