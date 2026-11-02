#!/usr/bin/env python3
"""Genera el mundo agrícola realista (v3) del gemelo digital.

Produce:
  1. models/field_ground/  — modelo de suelo con TEXTURA PROCEDURAL de tierra
     arada (surcos, ruido, variación tonal) generada con PIL. Las zonas de
     estrés hídrico van COCIDAS en la textura como manchas resecas claras.
  2. worlds/agricultural_field.world — hileras de cultivo continuas tipo dosel
     (cada surco = capas de "seto" verde con variación + calvas cloróticas en
     las zonas de estrés + tomates puntuales). Pocos visuales (~250): gzclient
     en WSL2 se ahoga con >800.

El estrés hídrico se representa de forma agronómica: dosel clorótico
(amarillo-pardo, porte menor, calvas) sobre suelo reseco — y la cámara
"térmica" (mono8) lo ve más claro -> más caliente -> CWSI alto.

RNG con semilla fija -> reproducible.  Uso: python3 tools/generate_field_world.py
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(BASE, "src", "crop_uav_simulator")
GROUND = os.path.join(PKG, "models", "field_ground")
WORLD = os.path.join(PKG, "worlds", "agricultural_field.world")

# Campo de la misión por defecto: x [0,20], y [-14,0]
FIELD_W, FIELD_H = 20.0, 14.0
GROUND_W, GROUND_H = 26.0, 20.0          # suelo texturizado (margen)
ROW_Y = [-0.8 - 1.05 * i for i in range(13)]
STRESS_ZONES = [(5.0, -3.5, 2.6, 1.8), (14.5, -7.0, 3.0, 2.2),
                (8.0, -11.5, 2.4, 1.6)]
# Focos de plaga: clusters de plantas infestadas (follaje pálido moteado).
# El detector sim-color los encuentra desde el aire -> pipeline completo.
PEST_CLUSTERS = [(17.0, -3.0, 1.9), (3.5, -9.6, 1.7),
                 (10.5, -5.0, 1.2)]

rng = random.Random(11)


def in_stress(x, y, margin=0.0):
    return any(((x - cx) / (rx + margin)) ** 2 + ((y - cy) / (ry + margin)) ** 2
               <= 1.0 for (cx, cy, rx, ry) in STRESS_ZONES)


def in_pest(x, y):
    return any((x - cx) ** 2 + (y - cy) ** 2 <= r * r
               for (cx, cy, r) in PEST_CLUSTERS)


# --------------------------------------------------------------------- suelo
def make_soil_texture(path: str, px_per_m: int = 40) -> None:
    """Tierra arada: base marrón + ruido + surcos paralelos + zonas resecas."""
    W, H = int(GROUND_W * px_per_m), int(GROUND_H * px_per_m)
    rs = np.random.RandomState(11)
    # base marrón con ruido multiescala
    base = np.stack([np.full((H, W), 96.0), np.full((H, W), 64.0),
                     np.full((H, W), 38.0)], axis=-1)
    for scale, amp in [(4, 18), (16, 12), (64, 8)]:
        n = rs.randn(H // scale + 1, W // scale + 1) * amp
        n = np.kron(n, np.ones((scale, scale)))[:H, :W]
        base += n[..., None] * np.array([1.0, 0.85, 0.6])
    # surcos de arado: franjas horizontales (paralelas a las hileras en x)
    yy = np.arange(H)[:, None]
    period = int(0.35 * px_per_m)
    furrow = 14.0 * np.sin(2 * np.pi * yy / period)
    base += furrow[..., None] * np.array([1.0, 0.9, 0.7])

    # mapeo mundo->píxel: x [-3, 23] -> [0,W], y [3, -17] -> [0,H]
    def to_px(x, y):
        return int((x + 3.0) / GROUND_W * W), int((3.0 - y) / GROUND_H * H)

    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    # zonas resecas: parches claros irregulares (suelo seco agrietado)
    for (cx, cy, rx_m, ry_m) in STRESS_ZONES:
        px, py = to_px(cx, cy)
        rx, ry = int(rx_m * px_per_m), int(ry_m * px_per_m)
        for k in range(60):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, 1) ** 0.5
            bx = px + int(math.cos(a) * r * rx)
            by = py + int(math.sin(a) * r * ry)
            br = rng.randint(6, 18)
            tone = rng.randint(140, 175)
            dr.ellipse([bx - br, by - br, bx + br, by + br],
                       fill=(tone, int(tone * 0.82), int(tone * 0.55)))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def write_ground_model() -> None:
    tex_dir = os.path.join(GROUND, "materials", "textures")
    scr_dir = os.path.join(GROUND, "materials", "scripts")
    os.makedirs(scr_dir, exist_ok=True)
    make_soil_texture(os.path.join(tex_dir, "soil.png"))
    with open(os.path.join(scr_dir, "field.material"), "w") as fh:
        fh.write("""material Field/Soil
{
  technique
  {
    pass
    {
      texture_unit
      {
        texture soil.png
        filtering anisotropic
      }
    }
  }
}
""")
    with open(os.path.join(GROUND, "model.config"), "w") as fh:
        fh.write("""<?xml version="1.0"?>
<model><name>field_ground</name><version>1.0</version>
<sdf version="1.6">model.sdf</sdf>
<description>Suelo agrícola texturizado (tierra arada, generado).</description></model>
""")
    with open(os.path.join(GROUND, "model.sdf"), "w") as fh:
        fh.write(f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="field_ground">
    <static>true</static>
    <link name="link">
      <visual name="soil">
        <pose>10 -7 0.01 0 0 0</pose>
        <geometry><box><size>{GROUND_W} {GROUND_H} 0.02</size></box></geometry>
        <material>
          <script>
            <uri>model://field_ground/materials/scripts</uri>
            <uri>model://field_ground/materials/textures</uri>
            <name>Field/Soil</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")



# -------------------------------------------------------------------- plantas
HEALTHY = [f"healthy_{i}" for i in range(3)]
STRESSED = [f"stressed_{i}" for i in range(2)]
INFESTED = [f"infested_{i}" for i in range(2)]
PLANT_DX = 0.8


def crop_visuals():
    """Plantas de tomate como MALLAS (1 visual c/u, ~150-360 tris)."""
    out, idx = [], 0
    for y in ROW_Y:
        x = 0.7
        while x <= FIELD_W - 0.7:
            jx = x + rng.uniform(-0.12, 0.12)
            jy = y + rng.uniform(-0.10, 0.10)
            yaw = rng.uniform(0, 2 * math.pi)
            if in_pest(jx, jy):
                mesh = rng.choice(INFESTED)    # foco de plaga
                s = rng.uniform(0.95, 1.2)
            elif in_stress(jx, jy, margin=0.15):
                if rng.random() < 0.30:        # calva (planta muerta)
                    x += PLANT_DX
                    continue
                mesh = rng.choice(STRESSED)
                s = rng.uniform(0.85, 1.05)
            else:
                mesh = rng.choice(HEALTHY)
                s = rng.uniform(0.9, 1.25)
            out.append(
                f'        <visual name="p{idx}"><pose>{jx:.2f} {jy:.2f} 0 0 0 {yaw:.2f}</pose>\n'
                f'          <geometry><mesh>'
                f'<uri>model://tomato_plants/meshes/{mesh}.obj</uri>'
                f'<scale>{s:.2f} {s:.2f} {s:.2f}</scale></mesh></geometry></visual>')
            idx += 1
            x += PLANT_DX
    return out


def main():
    write_ground_model()
    crop = crop_visuals()

    posts = []
    for k, (x, y) in enumerate([(-1.2, 0.8), (21.2, 0.8),
                                (-1.2, -14.8), (21.2, -14.8)]):
        posts.append(
            f'        <visual name="post_{k}"><pose>{x} {y} 0.9 0 0 0</pose>\n'
            f'          <geometry><cylinder><radius>0.10</radius><length>1.8</length></cylinder></geometry>\n'
            f'          <material><ambient>0.28 0.19 0.10 1</ambient>'
            f'<diffuse>0.36 0.25 0.13 1</diffuse></material></visual>')

    nl = "\n"
    world = f'''<?xml version="1.0" ?>
<!-- GENERADO por tools/generate_field_world.py — editar el generador, no este
     archivo. Campo de tomate 20x14 m: suelo texturizado (tierra arada) +
     plantas de tomate (mallas) con {len(STRESS_ZONES)} zonas de estrés hídrico
     (clorosis + calvas). ~{len(crop)} visuales de cultivo. -->
<sdf version="1.6">
  <world name="agricultural_field">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://field_ground</uri></include>

    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <scene>
      <ambient>0.55 0.55 0.5 1</ambient>
      <background>0.55 0.7 0.9 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- cámara inicial de la GUI: encuadra el campo y el dron -->
    <gui>
      <camera name="user_camera">
        <pose>10 -26 11 0 0.40 1.5708</pose>
      </camera>
    </gui>

    <!-- cultivo: dosel continuo por hileras -->
    <model name="tomato_crop">
      <static>true</static>
      <link name="rows">
{nl.join(crop)}
      </link>
    </model>

    <!-- linderos -->
    <model name="boundary_posts">
      <static>true</static>
      <link name="posts">
{nl.join(posts)}
      </link>
    </model>
  </world>
</sdf>
'''
    with open(WORLD, "w") as fh:
        fh.write(world)
    print(f"{WORLD}: {len(crop)} visuales de cultivo + suelo texturizado")


if __name__ == "__main__":
    main()
