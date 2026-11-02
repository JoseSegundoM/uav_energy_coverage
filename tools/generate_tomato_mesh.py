#!/usr/bin/env python3
"""Genera mallas OBJ low-poly de plantas de tomate para el gemelo digital.

Produce models/tomato_plants/meshes/:
  healthy_{0,1,2}.obj  — planta sana: tallo curvado, ramas, hojas verdes
                          en dos tonos, 2-4 tomates rojos/naranjas.
  stressed_{0,1}.obj   — planta con estrés hídrico: porte menor, hojas
                          caídas y cloróticas (amarillo-pardo), sin frutos.
  tomato.mtl           — materiales compartidos.

Cada planta son ~300-600 triángulos con normales — Gazebo 11 las carga vía
assimp y cada planta es UN solo visual (barato para gzclient).
RNG con semilla fija -> reproducible.
"""
import math
import os
import random

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(BASE, "src", "crop_uav_simulator", "models",
                   "tomato_plants", "meshes")

MTL = """# materiales planta de tomate
newmtl stem
Kd 0.28 0.42 0.16
Ka 0.10 0.15 0.06
newmtl stem_dry
Kd 0.45 0.38 0.18
Ka 0.16 0.13 0.06
newmtl leaf_a
Kd 0.10 0.42 0.10
Ka 0.03 0.14 0.03
newmtl leaf_b
Kd 0.16 0.55 0.13
Ka 0.05 0.18 0.04
newmtl leaf_dry_a
Kd 0.66 0.52 0.13
Ka 0.22 0.17 0.04
newmtl leaf_dry_b
Kd 0.55 0.38 0.11
Ka 0.18 0.12 0.03
newmtl tomato
Kd 0.82 0.10 0.05
Ka 0.27 0.03 0.02
newmtl tomato_green
Kd 0.55 0.62 0.22
Ka 0.18 0.20 0.07
newmtl leaf_pest_a
Kd 0.74 0.82 0.58
Ka 0.25 0.27 0.19
newmtl leaf_pest_b
Kd 0.63 0.72 0.48
Ka 0.21 0.24 0.16
newmtl spot
Kd 0.93 0.95 0.88
Ka 0.31 0.32 0.29
"""


class Mesh:
    def __init__(self):
        self.v, self.vn, self.faces = [], [], []   # faces: (mtl, [(vi,ni)x3])

    def add_tri(self, mtl, p0, p1, p2):
        ux = [p1[i] - p0[i] for i in range(3)]
        vx = [p2[i] - p0[i] for i in range(3)]
        n = [ux[1] * vx[2] - ux[2] * vx[1],
             ux[2] * vx[0] - ux[0] * vx[2],
             ux[0] * vx[1] - ux[1] * vx[0]]
        ln = math.sqrt(sum(c * c for c in n)) or 1.0
        n = [c / ln for c in n]
        i0 = len(self.v)
        self.v += [p0, p1, p2]
        self.vn.append(n)
        ni = len(self.vn)
        self.faces.append((mtl, [(i0 + 1, ni), (i0 + 2, ni), (i0 + 3, ni)]))

    def add_quad(self, mtl, p0, p1, p2, p3):
        self.add_tri(mtl, p0, p1, p2)
        self.add_tri(mtl, p0, p2, p3)

    def write(self, path):
        with open(path, "w") as fh:
            fh.write("mtllib tomato.mtl\n")
            for p in self.v:
                fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            for n in self.vn:
                fh.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            cur = None
            for mtl, idx in self.faces:
                if mtl != cur:
                    fh.write(f"usemtl {mtl}\n")
                    cur = mtl
                fh.write("f " + " ".join(f"{v}//{n}" for v, n in idx) + "\n")


def rotz(p, a):
    c, s = math.cos(a), math.sin(a)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2])


def tube(m, mtl, path_pts, r0, r1, sides=5):
    """Tubo poligonal a lo largo de una polilínea (tallo/rama)."""
    rings = []
    n = len(path_pts)
    for k, (x, y, z) in enumerate(path_pts):
        r = r0 + (r1 - r0) * k / max(n - 1, 1)
        ring = [(x + r * math.cos(2 * math.pi * i / sides),
                 y + r * math.sin(2 * math.pi * i / sides), z)
                for i in range(sides)]
        rings.append(ring)
    for k in range(n - 1):
        a, b = rings[k], rings[k + 1]
        for i in range(sides):
            j = (i + 1) % sides
            m.add_quad(mtl, a[i], a[j], b[j], b[i])


def leaf(m, mtl, base, yaw, droop, length, width, rng):
    """Hoja: rombo curvado de 4 triángulos apuntando (yaw) y caído (droop)."""
    tip_d = length
    tip_z = -math.sin(droop) * length
    mid_d = length * 0.45
    mid_z = -math.sin(droop) * length * 0.35 + 0.015
    pts_local = [
        (0.0, 0.0, 0.0),                       # base
        (mid_d, width / 2, mid_z),             # lado izq
        (tip_d * math.cos(droop), 0.0, tip_z),  # punta
        (mid_d, -width / 2, mid_z),            # lado der
    ]
    pts = []
    for p in pts_local:
        q = rotz(p, yaw)
        pts.append((base[0] + q[0], base[1] + q[1], base[2] + q[2]))
    b, l, t, r = pts
    c = ((b[0] + t[0]) / 2, (b[1] + t[1]) / 2, (b[2] + t[2]) / 2 + 0.01)
    m.add_tri(mtl, b, l, c)
    m.add_tri(mtl, l, t, c)
    m.add_tri(mtl, t, r, c)
    m.add_tri(mtl, r, b, c)
    # envés (doble cara para que no desaparezca según el ángulo)
    m.add_tri(mtl, c, l, b)
    m.add_tri(mtl, c, t, l)
    m.add_tri(mtl, c, r, t)
    m.add_tri(mtl, c, b, r)


def fruit(m, mtl, c, r):
    """Tomate: octaedro subdividido 1 vez (esfera low-poly)."""
    import itertools
    verts = [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
    octa = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    def norm(p):
        ln = math.sqrt(sum(x * x for x in p)) or 1.0
        return tuple(x / ln * r for x in p)
    for (a, b, d) in octa:
        pa, pb, pd = verts[a], verts[b], verts[d]
        ab = norm(tuple((pa[i] + pb[i]) / 2 for i in range(3)))
        bd = norm(tuple((pb[i] + pd[i]) / 2 for i in range(3)))
        da = norm(tuple((pd[i] + pa[i]) / 2 for i in range(3)))
        for tri in [(pa, ab, da), (ab, pb, bd), (da, bd, pd), (ab, bd, da)]:
            m.add_tri(mtl, *[(c[0] + p[0], c[1] + p[1], c[2] + p[2])
                             for p in tri])


def plant(seed, stressed, pest=False):
    rng = random.Random(seed)
    m = Mesh()
    h = rng.uniform(0.40, 0.55) if stressed else rng.uniform(0.60, 0.80)
    lean = rng.uniform(0.0, 0.10) + (0.08 if stressed else 0.0)
    lean_dir = rng.uniform(0, 2 * math.pi)
    stem_mtl = "stem_dry" if stressed else "stem"
    if pest:
        # infestada (mosca blanca/oídio): follaje pálido moteado
        leafs = ["leaf_pest_a", "leaf_pest_b"]
    elif stressed:
        leafs = ["leaf_dry_a", "leaf_dry_b"]
    else:
        leafs = ["leaf_a", "leaf_b"]

    # tallo curvado
    npts = 5
    path = []
    for k in range(npts):
        t = k / (npts - 1)
        dx = math.cos(lean_dir) * lean * t * t * h
        dy = math.sin(lean_dir) * lean * t * t * h
        path.append((dx, dy, t * h))
    tube(m, stem_mtl, path, 0.016, 0.007)

    # ramas con hojas
    n_br = rng.randint(3, 4) if stressed else rng.randint(13, 15)
    for b in range(n_br):
        t = 0.25 + 0.65 * b / max(n_br - 1, 1)
        bx = math.cos(lean_dir) * lean * t * t * h
        by = math.sin(lean_dir) * lean * t * t * h
        bz = t * h
        yaw = rng.uniform(0, 2 * math.pi)
        blen = rng.uniform(0.12, 0.22) * (0.75 if stressed else 1.0)
        down = rng.uniform(0.35, 0.75) if stressed else rng.uniform(0.05, 0.35)
        ex = bx + math.cos(yaw) * blen * math.cos(down)
        ey = by + math.sin(yaw) * blen * math.cos(down)
        ez = bz - math.sin(down) * blen * 0.6 + 0.02
        tube(m, stem_mtl, [(bx, by, bz), (ex, ey, ez)], 0.008, 0.004, sides=4)
        # hojas en la rama
        for klf in range(2 if stressed else 7):
            f = 0.28 + 0.72 * klf / 6
            lbase = (bx + (ex - bx) * f, by + (ey - by) * f, bz + (ez - bz) * f)
            droop = rng.uniform(0.55, 0.95) if stressed else rng.uniform(0.15, 0.45)
            lsz = (rng.uniform(0.10, 0.15) if stressed or pest
                   else rng.uniform(0.17, 0.26))
            leaf(m, rng.choice(leafs), lbase,
                 yaw + rng.uniform(-1.1, 1.1), droop,
                 lsz, lsz * rng.uniform(0.55, 0.7), rng)

    # corona de hojas arriba
    for k in range(2 if stressed else 12):
        yaw = rng.uniform(0, 2 * math.pi)
        droop = rng.uniform(0.5, 0.9) if stressed else rng.uniform(0.1, 0.4)
        top = path[-1]
        leaf(m, rng.choice(leafs), top, yaw, droop,
             rng.uniform(0.09, 0.14), rng.uniform(0.05, 0.09), rng)

    # moteado blanco (colonias) sobre plantas infestadas
    if pest:
        for k in range(rng.randint(6, 9)):
            t = rng.uniform(0.4, 0.95)
            yaw = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0.06, 0.16)
            c = (math.cos(yaw) * d, math.sin(yaw) * d,
                 t * h + rng.uniform(-0.03, 0.05))
            fruit(m, "spot", c, rng.uniform(0.014, 0.022))

    # RELLENO de copa: anillos de hojas grandes a varias alturas (dosel
    # denso, sin huecos visto desde el aire)
    if not stressed:
        for lvl in range(4):
            t = 0.40 + 0.20 * lvl
            for k in range(7):
                yaw = 2 * math.pi * (k + rng.uniform(-0.2, 0.2)) / 7
                d = rng.uniform(0.05, 0.14)
                lbase = (math.cos(yaw) * d, math.sin(yaw) * d,
                         t * h + rng.uniform(-0.03, 0.03))
                lsz2 = rng.uniform(0.16, 0.24)
                leaf(m, rng.choice(leafs), lbase, yaw,
                     rng.uniform(0.15, 0.5), lsz2,
                     lsz2 * rng.uniform(0.55, 0.7), rng)

    # frutos (solo plantas sanas): RACIMOS cargados y llamativos
    if not stressed and not pest:
        for cl in range(rng.randint(7, 9)):
            t = rng.uniform(0.30, 0.80)
            yaw = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0.07, 0.15)
            cx0 = math.cos(yaw) * d
            cy0 = math.sin(yaw) * d
            cz0 = t * h
            for k in range(rng.randint(4, 6)):
                c = (cx0 + rng.uniform(-0.035, 0.035),
                     cy0 + rng.uniform(-0.035, 0.035),
                     cz0 + rng.uniform(-0.045, 0.02))
                fruit(m, "tomato" if rng.random() < 0.85 else "tomato_green",
                      c, rng.uniform(0.040, 0.056))
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tomato.mtl"), "w") as fh:
        fh.write(MTL)
    specs = [("healthy_0", 101, False, False), ("healthy_1", 102, False, False),
             ("healthy_2", 103, False, False), ("stressed_0", 201, True, False),
             ("stressed_1", 202, True, False), ("infested_0", 301, False, True),
             ("infested_1", 302, False, True)]
    for name, seed, stressed, pest in specs:
        msh = plant(seed, stressed, pest)
        msh.write(os.path.join(OUT, f"{name}.obj"))
        print(f"{name}.obj: {len(msh.faces)} tris")
    # model.config para que model://tomato_plants resuelva
    root = os.path.dirname(OUT)
    with open(os.path.join(root, "model.config"), "w") as fh:
        fh.write("""<?xml version="1.0"?>
<model><name>tomato_plants</name><version>1.0</version>
<sdf version="1.6">model.sdf</sdf>
<description>Mallas low-poly de plantas de tomate (generadas).</description></model>
""")
    with open(os.path.join(root, "model.sdf"), "w") as fh:
        fh.write("""<?xml version="1.0" ?>
<sdf version="1.6"><model name="tomato_plants"><static>true</static>
<link name="unused"/></model></sdf>
""")


if __name__ == "__main__":
    main()
