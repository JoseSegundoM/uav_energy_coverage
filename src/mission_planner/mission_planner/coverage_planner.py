"""
coverage_planner.py — Planificador de cobertura tipo boustrophedon (zigzag)
para misiones de teledetección con UAV sobre polígonos de cultivo.

Nivel de rigor (maestría/doctorado):
  - Dirección de barrido ÓPTIMA: las líneas se alinean a la arista del polígono
    que minimiza el número de pasadas (resultado de Huang, W.H., 2001, ICRA,
    "Optimal line-sweep-based decompositions for coverage algorithms"). Para
    polígonos convexos la dirección óptima es paralela a una arista.
  - Espaciado de líneas derivado de la CÁMARA: GSD y huella de terreno con
    traslapes lateral/frontal configurables (ecuaciones estándar de fotogrametría).
  - Parametrización TEMPORAL con perfil de velocidad trapezoidal -> referencia
    continua (t -> pose) para el NMPC, no solo waypoints.
  - Métricas de cobertura: longitud, tiempo, pasadas, fracción cubierta, traslapes.

Fundamentos de cobertura: Choset & Pignon (1998) descomposición boustrophedon;
Galceran & Carreras (2013), "A survey on coverage path planning for robotics",
Robotics and Autonomous Systems. (VERIFICAR DOI/páginas antes de citar en tesis.)

Dependencias: numpy, shapely (>=2.0).
"""
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate as _shp_rotate
from shapely.validation import make_valid

EARTH_R = 6378137.0  # radio terrestre WGS-84 (m)

# --------------------------------------------------------------------------- #
# Energía: proxy explícito, no un modelo aerodinámico completo.
#
# En un multirrotor pequeño en vuelo lento de cobertura la potencia de
# sustentación domina, así que la energía de misión se aproxima por
# P_hover * t_total. P_hover sale de los MISMOS parámetros del vehículo que usa
# la planta del MIL (nmpc_model/uav_dynamics_casadi.py), no de un valor
# inventado: par por rotor Q = KM*w^2, potencia de eje P = Q*w = KM*w^3, por 4
# rotores y dividido por el rendimiento de propulsión.
#
# Lo que ESTE proxy captura: los giros salen caros porque el perfil trapezoidal
# por segmento frena a 0 en cada cabecera, y eso alarga t_total.
# Lo que NO captura: potencia inducida por traslación, viento y aceleraciones.
# Sirve para COMPARAR rutas del mismo vehículo, no para predecir autonomía.
# Para el modelo completo de vuelo de avance (inducida + perfil + parásita,
# con teoría de cantidad de movimiento) ver paper/scripts/power_model.py:
# a 2 m/s coincide con este proxy dentro del 0.2 %, pero diverge por encima de
# ~5 m/s, donde la relación de avance deja de ser pequeña.
_KM = 7.0e-7          # N*m*s^2/rad^2  (par reactivo)
_OMEGA_HOVER = 292.0  # rad/s
_ETA = 0.85           # rendimiento de propulsión
P_HOVER_W = 4.0 * _KM * _OMEGA_HOVER ** 3 / _ETA

XY = Tuple[float, float]


class PlanningError(RuntimeError):
    """The requested coverage cannot be planned without entering a no-fly zone."""


# --------------------------------------------------------------------------- #
# Conversión geodésica <-> local ENU (equirectangular; válida a escala de campo)
# --------------------------------------------------------------------------- #
def geodetic_to_local(lat: float, lon: float, lat0: float, lon0: float) -> XY:
    """(lat,lon) en grados -> (x_este, y_norte) en metros respecto a (lat0,lon0)."""
    lat0_r = math.radians(lat0)
    x = EARTH_R * math.radians(lon - lon0) * math.cos(lat0_r)
    y = EARTH_R * math.radians(lat - lat0)
    return x, y


def local_to_geodetic(x: float, y: float, lat0: float, lon0: float) -> XY:
    """(x_este,y_norte) en metros -> (lat,lon) en grados."""
    lat0_r = math.radians(lat0)
    lat = lat0 + math.degrees(y / EARTH_R)
    lon = lon0 + math.degrees(x / (EARTH_R * math.cos(lat0_r)))
    return lat, lon


# --------------------------------------------------------------------------- #
# Modelo de cámara: GSD y huella de terreno (fotogrametría)
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    """Parámetros de sensor para derivar espaciado de líneas y disparo."""
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int

    def gsd(self, altitude_m: float) -> float:
        """Ground Sampling Distance (m/px) a una altura dada."""
        return (self.sensor_width_mm * altitude_m) / (
            self.focal_length_mm * self.image_width_px)

    def footprint(self, altitude_m: float) -> XY:
        """Huella de terreno (ancho, alto) en metros."""
        w = self.sensor_width_mm * altitude_m / self.focal_length_mm
        h = self.sensor_height_mm * altitude_m / self.focal_length_mm
        return w, h


# --------------------------------------------------------------------------- #
# Resultado del planificador
# --------------------------------------------------------------------------- #
@dataclass
class CoveragePlan:
    waypoints_xy: List[XY]            # esquinas de la trayectoria (incl. giros)
    sweep_angle_rad: float            # dirección de barrido óptima
    track_spacing_m: float
    altitude_m: float
    metrics: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Planificador
# --------------------------------------------------------------------------- #
class CoveragePlanner:
    def __init__(
        self,
        polygon_xy: List[XY],
        altitude_m: float,
        speed_mps: float,
        *,
        track_spacing_m: Optional[float] = None,
        camera: Optional[Camera] = None,
        side_overlap: float = 0.70,
        forward_overlap: float = 0.80,
        accel_mps2: float = 2.0,
        yaw_mode: str = "course",     # "course" (alinea cámara al avance) | "fixed"
        fixed_yaw_rad: float = 0.0,
        obstacles_xy: Optional[List[List[XY]]] = None,
        obstacle_clearance_m: float = 0.0,
        cell_ordering: str = "exact",   # "exact" (Held-Karp) | "greedy"
        exact_max_cells: int = 14,
    ):
        if len(polygon_xy) < 3:
            raise ValueError("El polígono necesita >= 3 vértices.")
        shell = Polygon(polygon_xy)
        if not shell.is_valid:
            shell = shell.buffer(0)  # repara auto-intersecciones leves

        # --- Zonas de exclusión (obstáculos / no-fly) ---------------------
        # Se restan del área a cubrir Y se respetan en los tránsitos entre
        # celdas. La holgura se aplica UNA vez aquí: todo lo de aguas abajo
        # (barrido, visibilidad, métricas) usa ya la geometría inflada.
        self.cell_ordering = str(cell_ordering)
        self.exact_max_cells = int(exact_max_cells)
        self.clearance = float(obstacle_clearance_m)
        self.obstacles: List[Polygon] = []
        for ring in (obstacles_xy or []):
            if len(ring) < 3:
                raise ValueError("Cada obstáculo necesita >= 3 vértices.")
            o = Polygon(ring)
            if not o.is_valid:
                o = o.buffer(0)
            if self.clearance > 0:
                o = o.buffer(self.clearance, join_style=2)
            self.obstacles.append(o)

        self.shell = shell
        self.poly = shell
        for o in self.obstacles:
            self.poly = self.poly.difference(o)
        if self.poly.is_empty:
            raise ValueError("Las zonas de exclusión cubren todo el campo.")
        self.altitude = float(altitude_m)
        self.speed = float(speed_mps)
        self.accel = float(accel_mps2)
        self.side_overlap = float(side_overlap)
        self.forward_overlap = float(forward_overlap)
        self.camera = camera
        self.yaw_mode = yaw_mode
        self.fixed_yaw = float(fixed_yaw_rad)

        # --- Espaciado de líneas ---
        if track_spacing_m is not None:
            self.track_spacing = float(track_spacing_m)
            self.photo_interval = self.track_spacing  # sin cámara, asume cuadrado
        elif camera is not None:
            fw, fh = camera.footprint(self.altitude)
            self.track_spacing = fw * (1.0 - self.side_overlap)
            self.photo_interval = fh * (1.0 - self.forward_overlap)
        else:
            raise ValueError("Indica track_spacing_m o un objeto Camera.")
        if self.track_spacing <= 0:
            raise ValueError("track_spacing resultó <= 0 (revisa traslape/altura).")

    # ----- Dirección óptima (Huang 2001): minimiza extensión perpendicular ----
    def _optimal_sweep_angle(self) -> float:
        # Sobre el CONTORNO del campo: las zonas de exclusión no deben cambiar
        # la dirección de barrido (Huang minimiza la extensión perpendicular del
        # recinto, no la del área libre, que puede ser un MultiPolygon).
        coords = list(self.shell.exterior.coords)[:-1]
        best_height = math.inf
        best_angle = 0.0
        cx, cy = self.shell.centroid.x, self.shell.centroid.y
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            ang = math.atan2(y2 - y1, x2 - x1)
            rp = _shp_rotate(self.shell, -math.degrees(ang),
                             origin=(cx, cy), use_radians=False)
            miny, maxy = rp.bounds[1], rp.bounds[3]
            height = maxy - miny
            if height < best_height - 1e-9:
                best_height, best_angle = height, ang
        return best_angle

    @staticmethod
    def _segments_from_geom(geom) -> List[Tuple[XY, XY]]:
        segs: List[Tuple[XY, XY]] = []
        if geom.is_empty:
            return segs
        gt = geom.geom_type
        if gt == "LineString":
            cs = list(geom.coords)
            segs.append((cs[0], cs[-1]))
        elif gt in ("MultiLineString", "GeometryCollection"):
            for g in geom.geoms:
                segs.extend(CoveragePlanner._segments_from_geom(g))
        return segs

    @staticmethod
    def _rotate_pts(pts: List[XY], ang: float, cx: float, cy: float) -> List[XY]:
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            out.append((cx + ca * dx - sa * dy, cy + sa * dx + ca * dy))
        return out

    # ----- Descomposición boustrophedon en celdas (Choset & Pignon 1998) -----
    # Sin obstáculos hay UNA celda y el resultado es el zigzag clásico. Con
    # zonas de exclusión, cada línea de barrido se parte en varios tramos; los
    # tramos de líneas contiguas que se solapan en X pertenecen a la misma
    # celda. Donde el número de tramos cambia hay un PUNTO CRÍTICO (el
    # obstáculo empieza o termina): allí se cierran celdas y se abren otras.
    # Esto evita el error de encadenar tramos de una misma línea, que haría
    # volar en línea recta A TRAVÉS del obstáculo.
    def _cells_rot(self, rp) -> List[List[Tuple[float, float, float]]]:
        """Celdas en el marco girado. Cada celda es [(y, x0, x1), ...]."""
        minx, miny, maxx, maxy = rp.bounds
        scan_rows: List[Tuple[float, List[Tuple[float, float]]]] = []
        y = miny + self.track_spacing / 2.0
        while y <= maxy:
            scan = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
            segs = self._segments_from_geom(rp.intersection(scan))
            spans = sorted((min(a[0], b[0]), max(a[0], b[0])) for a, b in segs)
            # descarta tramos más cortos que un píxel de huella: son artefactos
            # de tangencia con el borde, no superficie que valga la pena cubrir
            spans = [s for s in spans if s[1] - s[0] > 1e-6]
            # A row lying exactly on an inflated obstacle edge intersects the
            # (closed) free area in collinear pieces that TOUCH, split at the
            # hole's vertices. They are one flyable span; left apart they read
            # as a critical point, open extra cells and force stops mid-row
            # (the published field had 9 cells instead of 6 because of this).
            merged: List[Tuple[float, float]] = []
            for s0, s1 in spans:
                if merged and s0 - merged[-1][1] <= 1e-9:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], s1))
                else:
                    merged.append((s0, s1))
            spans = merged
            if spans:
                scan_rows.append((y, spans))
            y += self.track_spacing

        closed: List[List[Tuple[float, float, float]]] = []
        open_cells: List[List[Tuple[float, float, float]]] = []
        prev_spans: List[Tuple[float, float]] = []

        for y, spans in scan_rows:
            # conectividad por solape en X con la última fila de cada celda
            links = [[] for _ in spans]           # span -> celdas abiertas
            back = [[] for _ in prev_spans]       # celda abierta -> spans
            for i, (a0, a1) in enumerate(spans):
                for j, (b0, b1) in enumerate(prev_spans):
                    if min(a1, b1) - max(a0, b0) > 1e-9:
                        links[i].append(j)
                        back[j].append(i)

            new_open: List[Optional[List[Tuple[float, float, float]]]] = \
                [None] * len(spans)
            for i, (a0, a1) in enumerate(spans):
                # continúa la celda solo si la correspondencia es 1:1; en un
                # split o un merge la celda anterior termina en el punto crítico
                if len(links[i]) == 1 and len(back[links[i][0]]) == 1:
                    cell = open_cells[links[i][0]]
                    cell.append((y, a0, a1))
                    new_open[i] = cell
                else:
                    new_open[i] = [(y, a0, a1)]
            # cierra las celdas que no continúan
            for j, cell in enumerate(open_cells):
                if not any(new_open[i] is cell for i in range(len(spans))):
                    closed.append(cell)
            open_cells = [c for c in new_open if c is not None]
            prev_spans = spans

        closed.extend(open_cells)
        return [c for c in closed if c]

    # ----- Tránsito entre celdas: grafo de visibilidad + Dijkstra -----
    def _blocked(self, p: XY, q: XY) -> bool:
        seg = LineString([p, q])
        for o in self.obstacles:
            # se prueba contra el obstáculo ENCOGIDO: así una arista que corre
            # pegada al contorno (que es justo por donde queremos rodearlo) no
            # cuenta como bloqueada
            if seg.intersection(o.buffer(-1e-6)).length > 1e-6:
                return True
        return not seg.within(self.shell.buffer(1e-6))

    def _vis_nodes_static(self) -> List[XY]:
        """Vértices candidatos del grafo de visibilidad, calculados UNA vez.

        La inflación mínima usa junta en INGLETE: con la junta redondeada por
        defecto, shapely sustituye cada esquina por decenas de vértices y el
        grafo pasa de ~10 a ~150 nodos, lo que multiplica por ~200 el coste de
        las comprobaciones de visibilidad.
        """
        if getattr(self, "_vis_cache", None) is None:
            nodes: List[XY] = []
            for o in self.obstacles:
                nodes.extend(
                    list(o.buffer(1e-3, join_style=2).exterior.coords)[:-1])
            if not self.shell.convex_hull.equals(self.shell):
                nodes.extend(
                    list(self.shell.buffer(-1e-3, join_style=2).exterior.coords)[:-1])
            self._vis_cache = nodes
            # visibilidad entre nodos estáticos: no cambia entre llamadas
            m = len(nodes)
            self._vis_adj = [[] for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    if not self._blocked(nodes[i], nodes[j]):
                        w = math.dist(nodes[i], nodes[j])
                        self._vis_adj[i].append((j + 2, w))
                        self._vis_adj[j].append((i + 2, w))
        return self._vis_cache

    def _transit(self, p: XY, q: XY) -> List[XY]:
        """Camino libre de p a q. [] si el tramo recto ya es válido."""
        if not self._blocked(p, q):
            return []
        static = self._vis_nodes_static()
        nodes: List[XY] = [p, q] + list(static)

        n = len(nodes)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        # aristas estáticas ya precalculadas (índices desplazados en 2)
        for i, lst in enumerate(self._vis_adj):
            adj[i + 2].extend(lst)
        # solo faltan las aristas que salen de p y de q
        for i in (0, 1):
            for j in range(n):
                if j == i:
                    continue
                if not self._blocked(nodes[i], nodes[j]):
                    w = math.dist(nodes[i], nodes[j])
                    adj[i].append((j, w))
                    adj[j].append((i, w))

        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-12:
                continue
            if u == 1:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[1] == math.inf:
            # No free route: the free area is split (an inflated obstacle
            # meets the boundary or another obstacle). Emitting the straight
            # segment would cross the no-fly zone, by up to 1.9 m in the
            # cross-review's random fields, so the plan is refused instead.
            raise PlanningError(f"no collision-free route from {p} to {q}: "
                                "the free area is disconnected")
        path, u = [], 1
        while u != -1:
            path.append(nodes[u])
            u = prev[u]
        return path[::-1][1:-1]            # sin extremos (ya están en la ruta)

    # ----- Orden de visita de celdas -----
    def _transit_cost(self, p: XY, q: XY) -> float:
        """Longitud del tránsito p->q rodeando obstáculos (no la euclídea)."""
        bridge = self._transit(p, q)
        pts = [p] + bridge + [q]
        return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def _order_cells(self, cells: List[List[XY]]) -> List[Tuple[int, bool]]:
        """Devuelve [(indice_celda, invertida), ...].

        Cada celda puede recorrerse en dos sentidos, así que el problema es un
        TSP de camino abierto sobre 2n estados. Con `cell_ordering="exact"` y
        pocas celdas se resuelve con Held-Karp, que da el ÓPTIMO en
        milisegundos; por encima del umbral se cae a vecino más cercano
        refinado con 2-opt, porque la tabla de Held-Karp crece como 2^n.
        """
        n = len(cells)
        if n <= 1:
            return [(0, False)] if n else []

        # coste[(j,o1)][(k,o2)] = tránsito del final de una celda al inicio de otra
        ends = [(cells[j][-1], cells[j][0]) for j in range(n)]      # (fin, inicio)

        def start_pt(j, o):
            return cells[j][-1] if o else cells[j][0]

        def end_pt(j, o):
            return cells[j][0] if o else cells[j][-1]

        cost = {}

        def c(j, o1, k, o2):
            key = (j, o1, k, o2)
            if key not in cost:
                cost[key] = self._transit_cost(end_pt(j, o1), start_pt(k, o2))
            return cost[key]

        use_exact = (self.cell_ordering == "exact" and n <= self.exact_max_cells)
        if use_exact:
            INF = float("inf")
            full = 1 << n
            dp = [[[INF, INF] for _ in range(n)] for _ in range(full)]
            par = {}
            for j in range(n):
                for o in (0, 1):
                    dp[1 << j][j][o] = 0.0          # arranque libre
            for mask in range(full):
                for j in range(n):
                    if not (mask >> j) & 1:
                        continue
                    for o1 in (0, 1):
                        base = dp[mask][j][o1]
                        if base == INF:
                            continue
                        for k in range(n):
                            if (mask >> k) & 1:
                                continue
                            nm = mask | (1 << k)
                            for o2 in (0, 1):
                                v = base + c(j, o1, k, o2)
                                if v < dp[nm][k][o2] - 1e-12:
                                    dp[nm][k][o2] = v
                                    par[(nm, k, o2)] = (mask, j, o1)
            best, endst = INF, None
            for j in range(n):
                for o in (0, 1):
                    if dp[full - 1][j][o] < best:
                        best, endst = dp[full - 1][j][o], (j, o)
            seq, mask, (j, o) = [], full - 1, endst
            while True:
                seq.append((j, bool(o)))
                prev = par.get((mask, j, o))
                if prev is None:
                    break
                mask, j, o = prev
            return seq[::-1]

        # --- vecino más cercano + 2-opt ---
        pending = list(range(n))
        order: List[Tuple[int, int]] = []
        cur_j, cur_o = None, None
        while pending:
            best = None
            for idx in pending:
                for o in (0, 1):
                    d = 0.0 if cur_j is None else c(cur_j, cur_o, idx, o)
                    if best is None or d < best[0]:
                        best = (d, idx, o)
            _, idx, o = best
            order.append((idx, o))
            cur_j, cur_o = idx, o
            pending.remove(idx)

        def total(seq):
            return sum(c(seq[i][0], seq[i][1], seq[i + 1][0], seq[i + 1][1])
                       for i in range(len(seq) - 1))

        improved, best_cost = True, total(order)
        while improved:
            improved = False
            for i in range(len(order) - 1):
                for k in range(i + 1, len(order)):
                    cand = order[:i] + [(j, not o) for j, o in
                                        reversed(order[i:k + 1])] + order[k + 1:]
                    tc = total(cand)
                    if tc < best_cost - 1e-9:
                        order, best_cost, improved = cand, tc, True
        return [(j, bool(o)) for j, o in order]

    def plan(self) -> CoveragePlan:
        angle = self._optimal_sweep_angle()
        cx, cy = self.shell.centroid.x, self.shell.centroid.y

        # Gira el área libre para barrer paralelo al eje X
        rp = _shp_rotate(self.poly, -math.degrees(angle),
                         origin=(cx, cy), use_radians=False)
        cells_rot = self._cells_rot(rp)

        # zigzag dentro de cada celda, ya en el marco del mundo
        cells: List[List[XY]] = []
        n_rows_total = 0
        for cell in cells_rot:
            pts_rot: List[XY] = []
            for k, (y, x0, x1) in enumerate(cell):
                pts_rot.extend([(x0, y), (x1, y)] if k % 2 == 0
                               else [(x1, y), (x0, y)])
            n_rows_total += len(cell)
            pts = self._rotate_pts(pts_rot, angle, cx, cy)
            # The link between the end of one row and the start of the next is
            # straight only if it is free: with an oblique sweep or a rotated
            # obstacle it can cross the no-fly polygon, so it is routed like
            # an inter-cell transit.  Rows themselves are free by construction.
            routed: List[XY] = pts[:2]
            for i in range(2, len(pts), 2):
                routed.extend(self._transit(routed[-1], pts[i]))
                routed.extend(pts[i:i + 2])
            cells.append(routed)

        # orden de visita de las celdas (ver _order_cells): el coste de cada
        # enlace es la longitud REAL del tránsito, que rodea los obstáculos, no
        # la distancia euclídea. Ordenar celdas es un TSP de camino abierto con
        # dos orientaciones por celda; para los tamaños que produce una parcela
        # agrícola se resuelve exacto.
        order = self._order_cells(cells)
        waypoints: List[XY] = []
        transit_len = 0.0
        cur: Optional[XY] = None
        for idx, rev in order:
            seq = cells[idx][::-1] if rev else cells[idx]
            if cur is not None:
                bridge = self._transit(cur, seq[0])
                waypoints.extend(bridge)
                pts = [cur] + bridge + [seq[0]]
                transit_len += sum(math.dist(pts[i], pts[i + 1])
                                   for i in range(len(pts) - 1))
            waypoints.extend(seq)
            cur = seq[-1]

        plan = CoveragePlan(
            waypoints_xy=waypoints,
            sweep_angle_rad=angle,
            track_spacing_m=self.track_spacing,
            altitude_m=self.altitude,
        )
        plan.metrics = self._metrics(waypoints, n_rows_total)
        plan.metrics["cells"] = float(len(cells))
        plan.metrics["transit_length_m"] = round(transit_len, 2)
        return plan

    # ----- Métricas de cobertura -----
    def _metrics(self, wps: List[XY], n_passes: int) -> Dict[str, float]:
        if len(wps) < 2:
            return {}
        length = sum(math.dist(wps[i], wps[i + 1]) for i in range(len(wps) - 1))
        # Tiempo con el trapezoide POR SEGMENTO — el mismo perfil que emite
        # time_parameterize(). El trapezoide global anterior suponía un único
        # tramo recto e ignoraba que el vehículo frena a 0 en cada giro, así
        # que subestimaba la misión justo en lo que cuesta: las cabeceras.
        t = sum(self._segment_time(math.dist(wps[i], wps[i + 1]))
                for i in range(len(wps) - 1))
        cov = self._coverage_breakdown(wps)
        coverage = cov["coverage_accessible"]
        return {
            **{k: round(v, 4) for k, v in cov.items()},
            "passes": float(n_passes),
            "turns": float(max(0, len(wps) - 2)),
            "path_length_m": round(length, 2),
            "est_time_s": round(t, 1),
            "est_time_min": round(t / 60.0, 2),
            "energy_wh": round(P_HOVER_W * t / 3600.0, 3),
            "track_spacing_m": round(self.track_spacing, 3),
            "photo_interval_m": round(self.photo_interval, 3),
            "field_area_ha": round(self.poly.area / 1e4, 3),
            "coverage_fraction": round(coverage, 3),
            "gsd_m_per_px": round(self.camera.gsd(self.altitude), 4)
            if self.camera else float("nan"),
        }

    def _coverage_breakdown(self, wps: List[XY]) -> Dict[str, float]:
        """Cobertura desglosada en tres cifras, no una sola.

        Una única cifra de "cobertura" es ambigua cuando hay zonas de exclusión:
        mezcla el área que el vehículo NO PUDO cubrir con la que se le PROHIBIÓ
        cubrir. Se reportan por separado:

          coverage_field       barrido / área bruta del campo. Baja siempre que
                               hay exclusiones, por definición: es el dato
                               agronómico (qué fracción de la parcela se
                               inspeccionó).
          coverage_accessible  barrido / área LIBRE. Mide al planificador: si no
                               es ~1.0, el barrido está dejando huecos que sí
                               podía cubrir.
          excluded_fraction    área prohibida / área bruta. El contexto que hace
                               interpretable a las otras dos.
        """
        if len(wps) < 2:
            return {"coverage_field": 0.0, "coverage_accessible": 0.0,
                    "excluded_fraction": 0.0}
        half = (self.camera.footprint(self.altitude)[0] / 2.0
                if self.camera else self.track_spacing / 2.0)
        swath = LineString(wps).buffer(half, cap_style=2)  # 2 = extremos planos
        if not swath.is_valid:      # a routed polyline can self-overlap
            swath = make_valid(swath)
        a_field = max(self.shell.area, 1e-6)
        a_free = max(self.poly.area, 1e-6)
        return {
            "coverage_field": min(1.0, swath.intersection(self.shell).area / a_field),
            "coverage_accessible": min(1.0, swath.intersection(self.poly).area / a_free),
            "excluded_fraction": (a_field - a_free) / a_field,
        }

    def _coverage_fraction(self, wps: List[XY]) -> float:
        """Compatibilidad: la fracción sobre el área ACCESIBLE."""
        return self._coverage_breakdown(wps)["coverage_accessible"]

    def _segment_time(self, L: float) -> float:
        """Duración de un tramo recto con trapezoide de v=0 a v=0."""
        if L <= 1e-9:
            return 0.0
        v, a = self.speed, self.accel
        d_acc = v * v / (2 * a)
        if L < 2 * d_acc:                 # no da tiempo a alcanzar crucero
            return 2.0 * math.sqrt(L / a)
        return 2.0 * (v / a) + (L - 2 * d_acc) / v

    def _mean_pass_length(self, wps: List[XY]) -> float:
        if len(wps) < 2:
            return 0.0
        seg = [math.dist(wps[i], wps[i + 1]) for i in range(len(wps) - 1)]
        return float(np.mean(seg)) if seg else 0.0

    # ----- Parametrización temporal (perfil trapezoidal POR SEGMENTO) -----
    # NOTA (integración SIL, P2/P4): el perfil original era trapezoidal GLOBAL
    # (aceleraba al inicio del camino y frenaba solo al final), de modo que la
    # referencia tomaba cada giro de 180 grados a velocidad de crucero: cambio
    # de dirección instantáneo = aceleración infinita, inseguible por el UAV
    # (medido en Gazebo: tumbos con tilt de 40-179 grados en cada esquina).
    # Ahora cada segmento del boustrophedon tiene su propio trapezoide con
    # v=0 en ambos extremos (parar -> girar -> acelerar), el patrón estándar
    # de los drones de survey. La referencia resultante es dinámicamente
    # factible y el seguimiento queda suave. (Extensión natural: giros Dubins
    # con velocidad de esquina > 0 — ver "extensiones" en docs/TECHNICAL_SPEC.md.)
    def time_parameterize(self, plan: CoveragePlan, dt: float = 0.05
                          ) -> List[Tuple[float, float, float, float, float]]:
        """Devuelve [(t, x, y, z, yaw), ...] muestreado cada dt segundos."""
        wps = plan.waypoints_xy
        if len(wps) < 2:
            return []

        v, a = self.speed, self.accel
        out = []
        t0 = 0.0
        for i in range(1, len(wps)):
            p0, p1 = wps[i - 1], wps[i]
            L = math.dist(p0, p1)
            if L < 1e-9:
                continue
            if self.yaw_mode == "fixed":
                yaw = self.fixed_yaw
            else:
                yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

            # trapezoide del segmento (v0 = vf = 0)
            d_acc = v * v / (2 * a)
            if L < 2 * d_acc:                      # perfil triangular
                t_acc = math.sqrt(L / a)
                v_pk = a * t_acc
                T = 2 * t_acc
                t_cru = 0.0
            else:
                t_acc = v / a
                v_pk = v
                t_cru = (L - 2 * d_acc) / v
                T = 2 * t_acc + t_cru

            def s_at(tt: float) -> float:
                if tt < t_acc:
                    return 0.5 * a * tt * tt
                if tt < t_acc + t_cru:
                    return 0.5 * a * t_acc ** 2 + v_pk * (tt - t_acc)
                td = tt - t_acc - t_cru
                return (0.5 * a * t_acc ** 2 + v_pk * t_cru
                        + v_pk * td - 0.5 * a * td * td)

            # muestreo alineado a la rejilla global de dt
            t = (math.floor(t0 / dt) + 1) * dt
            while t <= t0 + T + 1e-9:
                r = min(max(s_at(t - t0) / L, 0.0), 1.0)
                out.append((round(t, 4),
                            p0[0] + r * (p1[0] - p0[0]),
                            p0[1] + r * (p1[1] - p0[1]),
                            self.altitude, yaw))
                t += dt
            t0 += T
        return out

    # ----- Export geodésico -----
    def to_geodetic(self, plan: CoveragePlan, lat0: float, lon0: float
                    ) -> List[Tuple[float, float, float]]:
        """waypoints (x,y) -> [(lat, lon, alt), ...]."""
        return [(*local_to_geodetic(x, y, lat0, lon0), self.altitude)
                for x, y in plan.waypoints_xy]
