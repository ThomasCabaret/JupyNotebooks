from enum import Enum, auto
import pandas as pd
import plotly.express as px
import math
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720

EXPORTS = {
    "hd": (1920, 1080),
    "4k": (3840, 2160)
}

#------------------------------------------------------------------------------------
class DecayMode(Enum):
    STABLE = "STBL"
    BETA_MINUS = "B-"
    BETA_PLUS = "B+"
    DOUBLE_BETA_MINUS = "2B-"
    ALPHA = "A"
    PROTON = "p"
    NEUTRON = "n"
    TWO_PROTON = "2p"
    TWO_NEUTRON = "2n"
    TRITON = "t"
    DEUTERON = "d"
    ELECTRON_CAPTURE = "EC"
    ISOMERIC_TRANSITION = "IT"
    SPONTANEOUS_FISSION = "SF"
    UNKNOWN = "?"

#------------------------------------------------------------------------------------
class DecayBranch:
    def __init__(self, mode: DecayMode, intensity: float = None):
        self.mode = mode
        self.intensity = intensity  # in %, optional (can be None if unknown)

    def __repr__(self):
        return f"{self.mode.value}={self.intensity}%" if self.intensity is not None else f"{self.mode.value}"

#------------------------------------------------------------------------------------
class NuclearSpecies:
    def __init__(self, Z: int, N: int):
        self.Z = Z
        self.N = N
        self.A = Z + N
        self.symbol: str = None
        self.mass_excess_keV: float = None
        self.binding_energy_per_A_keV: float = None
        self.atomic_mass_micro_u: float = None
        self.half_life_s: float = None
        self.half_life_unit: str = None  # optional: raw text unit from file
        self.decay_branches: list[DecayBranch] = []
        self.spin_parity: str = None
        self.excited_energy_keV: float = None
        self.excited_energy_unc_keV: float = None
        self.is_isomer: bool = False
        self.raw_sources: dict[str, str] = {}  # optional debug storage

    def add_decay_mode(self, mode: DecayMode, intensity: float = None):
        self.decay_branches.append(DecayBranch(mode, intensity))

    def get_dominant_decay(self) -> DecayBranch | None:
        if not self.decay_branches:
            return None
        stable_branch = next((b for b in self.decay_branches if b.mode == DecayMode.STABLE), None)
        if stable_branch:
            return stable_branch
        return max((b for b in self.decay_branches if b.intensity is not None), key=lambda x: x.intensity, default=None)
        def __repr__(self):
            return f"{self.symbol or '?'}-{self.A} (Z={self.Z}, N={self.N})"

    def __str__(self):
        lines = [f"{self.symbol or '?'}-{self.A} (Z={self.Z}, N={self.N})"]
        if self.half_life_s is not None:
            lines.append(f"  Half-life: {self.half_life_s} s [{self.half_life_unit}]")
        if self.decay_branches:
            modes = ', '.join(str(b) for b in self.decay_branches)
            lines.append(f"  Decay modes: {modes}")
        if self.spin_parity:
            lines.append(f"  Spin/parity: {self.spin_parity}")
        if self.excited_energy_keV is not None:
            lines.append(f"  Excitation energy: {self.excited_energy_keV} keV")
        return '\n'.join(lines)

# NUBASE PARSER #################################################################################
def parse_nubase_typed(filepath):
    fields = [
        ("AAA", 0, 3, int),
        ("ZZZi", 4, 8, str),
        ("A_El", 11, 16, str),
        ("s", 16, 17, str),
        ("Mass", 18, 31, float),
        ("dMass", 31, 42, float),
        ("Exc", 42, 54, float),
        ("dExc", 54, 65, float),
        ("Orig", 65, 67, str),
        ("IsomUnc", 67, 68, str),
        ("IsomInv", 68, 69, str),
        ("HalfLife", 69, 78, float),
        ("HalfLifeUnit", 78, 80, str),
        ("dT", 81, 88, float),
        ("Jpi", 88, 102, str),
        ("ENSDF_Year", 102, 104, str),
        ("DiscoveryYear", 114, 118, int),
        ("DecayModes", 119, 209, str)
    ]
    def try_parse(value, target_type, field_name):
        raw = value.strip().replace('#', '').replace('*', '').replace('~', '')
        if raw.lower() in {'', 'non-exist', 'none'}:
            return None
        if field_name == "HalfLife":
            if raw.lower() == "stbl":
                return float('inf')
            if raw.lower() == "p-unst":
                return None
        if field_name == "DecayModes":
            return parse_decay_modes(raw)
        if raw.startswith(('>', '<')):
            raw = raw[1:]
        if target_type is float:
            import re
            raw_clean = raw.replace('~', '').replace('#', '').replace('*', '').strip()
            float_match = re.search(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', raw_clean)
            if not float_match:
                print(f"[ERROR] Failed to find float in '{raw}'")
                return None
            number = float_match.group(0)
            unit_start = float_match.end()
            unit = raw_clean[unit_start:].strip()
            try:
                value = float(number)
            except ValueError:
                print(f"[ERROR] Failed to convert numeric part of '{raw}' to float")
                return None
            if unit:
                return convert_half_life_to_seconds(value, unit)
            else:
                return value
        try:
            return target_type(raw)
        except Exception:
            print(f"[ERROR] Failed to parse field '{field_name}' ? expected {target_type.__name__}, got raw='{raw}'")
            return None
    def parse_decay_modes(raw):
        modes = []
        entries = raw.split(';')
        for entry in entries:
            entry = entry.strip()
            if '=' not in entry:
                continue
            key, val = entry.split('=', 1)
            key = key.strip()
            val = val.strip().rstrip('%')
            if key == 'IS':
                continue
            submodes = key.split('+')
            for subkey in submodes:
                subkey = subkey.strip()
                if not subkey:
                    continue
                if subkey == 'e+':
                    decay = DecayMode.BETA_PLUS
                elif subkey == '3H':
                    decay = DecayMode.TRITON
                elif subkey == '3He':
                    decay = DecayMode.UNKNOWN
                    print(f"[ERROR] Unknown decay mode '{subkey}' in DecayModes")
                elif subkey in {"14C", "20O", "24Ne", "26Ne", "22Ne", "23F", "28Mg", "34Si", "20Ne", "24Ne+26Ne"}:
                    decay = DecayMode.SPONTANEOUS_FISSION
                elif subkey == '2B-':
                    decay = DecayMode.DOUBLE_BETA_MINUS
                elif subkey == 'B':
                    decay = DecayMode.BETA_PLUS
                elif subkey == 'e':
                    decay = DecayMode.BETA_PLUS
                elif subkey.startswith('B-'):
                    decay = DecayMode.BETA_MINUS
                elif subkey.startswith('B+'):
                    decay = DecayMode.BETA_PLUS
                elif subkey == 'EC':
                    decay = DecayMode.ELECTRON_CAPTURE
                elif subkey == 'IT':
                    decay = DecayMode.ISOMERIC_TRANSITION
                elif subkey == 'A':
                    decay = DecayMode.ALPHA
                elif subkey == 'p':
                    decay = DecayMode.PROTON
                elif subkey == '2p':
                    decay = DecayMode.TWO_PROTON
                elif subkey == 'n':
                    decay = DecayMode.NEUTRON
                elif subkey == '2n':
                    decay = DecayMode.TWO_NEUTRON
                elif subkey == 't':
                    decay = DecayMode.TRITON
                elif subkey == 'd':
                    decay = DecayMode.DEUTERON
                elif subkey == 'SF':
                    decay = DecayMode.SPONTANEOUS_FISSION
                else:
                    decay = DecayMode.UNKNOWN
                    print(f"[ERROR] Unknown decay mode '{subkey}' in DecayModes")
                try:
                    intensity = float(val)
                except ValueError:
                    intensity = None
                modes.append((decay, intensity))
        return modes
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            row = {}
            for name, start, end, typ in fields:
                raw = line[start:end]
                row[name] = try_parse(raw, typ, name)
            data.append(row)
    return pd.DataFrame(data)

#------------------------------------------------------------------------------------
def convert_half_life_to_seconds(value: float | None, unit: str | None) -> float | None:
    if math.isinf(value):
        return value
    if value is None or unit is None:
        return None
    scale = {
        "ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12,
        "ns": 1e-9,  "us": 1e-6,  "ms": 1e-3,  "s": 1,      "m": 60,
        "h": 3600,   "d": 86400,  "y": 3.15576e7,
        "ky": 3.15576e10, "My": 3.15576e13, "Gy": 3.15576e16,
        "Ty": 3.15576e19, "Py": 3.15576e22, "Ey": 3.15576e25, "Zy": 3.15576e28, "Yy": 3.15576e31
    }
    unit = unit.strip()
    factor = scale.get(unit, None)
    if factor is None:
        print(f"[ERROR] Unknown time unit '{unit}' for half-life conversion")
        return None
    return value * factor

#------------------------------------------------------------------------------------
def df_to_species(df: pd.DataFrame) -> dict[tuple[int, int], NuclearSpecies]:
    species_dict = {}
    for _, row in df.iterrows():
        z_str = row["ZZZi"]
        if not z_str or len(z_str) < 3:
            continue
        try:
            Z = int(z_str[:3])
        except ValueError:
            continue
        A = row["AAA"]
        if pd.isna(A):
            continue
        N = A - Z
        key = (Z, N)
        if key in species_dict:
            continue  # one species per (Z,N) only
        sp = NuclearSpecies(Z, N)
        sp.symbol = row["A_El"].strip() if row["A_El"] else None
        sp.mass_excess_keV = row["Mass"]
        sp.excited_energy_keV = row["Exc"]
        sp.excited_energy_unc_keV = row["dExc"]
        sp.half_life_s = convert_half_life_to_seconds(row["HalfLife"], row["HalfLifeUnit"])
        sp.half_life_unit = row["HalfLifeUnit"]
        sp.spin_parity = row["Jpi"]
        sp.is_isomer = row["s"] in {'m', 'n', 'p', 'q', 'r', 'i', 'j', 'x'}
        if sp.half_life_s == float('inf'):
            sp.add_decay_mode(DecayMode.STABLE)
        decay_info = row["DecayModes"]
        if decay_info:
            for mode, intensity in decay_info:
                sp.add_decay_mode(mode, intensity)
        species_dict[key] = sp
    return species_dict

#------------------------------------------------------------------------------------
def parse_nubase_species(filepath: str) -> dict[tuple[int, int], NuclearSpecies]:
    df = parse_nubase_typed(filepath)
    df = df[df["s"].isna() | ~df["s"].isin(['m', 'n', 'p', 'q', 'r', 'i', 'j', 'x'])]
    return df_to_species(df)

# PLOTLY ADAPTER #################################################################################

#------------------------------------------------------------------------------------
def plot_half_life_nz(species_dict: dict[tuple[int, int], "NuclearSpecies"], basename="half_life_nz"):
    shapes = []; texts = []; colors = []; xs = []; ys = []
    for (Z, N), sp in species_dict.items():
        if sp.half_life_s is None or sp.half_life_s <= 0:
            continue
        log_hl = math.log10(sp.half_life_s)
        x0, x1 = N - 0.5, N + 0.5
        y0, y1 = Z - 0.5, Z + 0.5
        xs.append([x0, x1, x1, x0, x0])
        ys.append([y0, y0, y1, y1, y0])
        colors.append(log_hl)
        texts.append(f"{sp.symbol or '?'}-{sp.A}<br>log10(T1/2)={log_hl:.2f}")
    if not xs:
        print("[WARN] No valid half-life data to plot.")
        return
    color_min = min(colors)
    color_max = max(colors)
    colorscale = px.colors.sequential.Viridis
    norm = lambda v: (v - color_min) / (color_max - color_min) if color_max > color_min else 0.5
    fig = go.Figure()
    for x_coords, y_coords, c, text in zip(xs, ys, colors, texts):
        fill_color = sample_colorscale(colorscale, norm(c), low=color_min, high=color_max)[0]
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill='toself',
            fillcolor=fill_color,
            line=dict(width=0),
            mode='none',
            hoverinfo='text',
            hovertext=text,
            showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color=[color_min, color_max], colorscale=colorscale, cmin=color_min, cmax=color_max, colorbar=dict(title="log10(Half-life [s])")),
        showlegend=False
    ))
    fig.update_layout(
        title="N-Z Diagram: log10(Half-life in seconds)",
        xaxis=dict(title="Neutron Number (N)", scaleanchor="y", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10)),
        yaxis=dict(title="Proton Number (Z)", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10)),
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        margin=dict(t=60, b=60),
        template="plotly_white"
    )
    out_html = f"{basename}.html"
    out_png = f"{basename}.png"
    fig.write_html(out_html)
    print(f"[INFO] Plot saved to {out_html}")
    for suffix, (w, h) in EXPORTS.items():
        scale = w / NATIVE_WIDTH
        output_file = f"{out_png[:-4]}_{suffix}.png"
        try:
            fig.write_image(output_file, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, scale=scale)
            print(f"[INFO] Saved {output_file} at scale {scale:.2f}")
        except Exception as e:
            print(f"[WARN] Failed to write {output_file}: {e}")

#------------------------------------------------------------------------------------
def plot_dominant_decay_nz(species_dict: dict[tuple[int, int], "NuclearSpecies"], basename="dominant_decay_nz"):
    data = []
    for (Z, N), sp in species_dict.items():
        dom = sp.get_dominant_decay()
        if dom:
            data.append({"Z": Z, "N": N, "Symbol": sp.symbol or "?", "DecayMode": dom.mode.value})
    if not data:
        print("[WARN] No species with dominant decay mode to plot.")
        return
    df = pd.DataFrame(data)
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3
    unique_modes = sorted(df["DecayMode"].unique())
    color_map = {mode: palette[i % len(palette)] for i, mode in enumerate(unique_modes)}
    fig = go.Figure()
    for mode in unique_modes:
        sub = df[df["DecayMode"] == mode]
        for _, row in sub.iterrows():
            Z, N = row["Z"], row["N"]
            x0, x1 = N - 0.5, N + 0.5
            y0, y1 = Z - 0.5, Z + 0.5
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill='toself',
                fillcolor=color_map[mode],
                line=dict(width=0),
                mode='none',
                name=mode,
                hoverinfo='text',
                hovertext=f"{row['Symbol']}-{Z+N}<br>{mode}",
                showlegend=False
            ))
    for mode in unique_modes:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color_map[mode]),
            name=mode
        ))
    fig.update_layout(
        title="N-Z Diagram by Dominant Decay Mode",
        xaxis=dict(title="Neutron Number (N)", scaleanchor="y", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10)),
        yaxis=dict(title="Proton Number (Z)", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10)),
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        margin=dict(t=60, b=60),
        template="plotly_white"
    )
    out_html = f"{basename}.html"
    out_png = f"{basename}.png"
    fig.write_html(out_html)
    print(f"[INFO] Plot saved to {out_html}")
    for suffix, (w, h) in EXPORTS.items():
        scale = w / NATIVE_WIDTH
        output_file = f"{out_png[:-4]}_{suffix}.png"
        try:
            fig.write_image(output_file, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, scale=scale)
            print(f"[INFO] Saved {output_file} at scale {scale:.2f}")
        except Exception as e:
            print(f"[WARN] Failed to write {output_file}: {e}")

# PYGAME ADAPTER #################################################################################
import random
import os, pygame

#------------------------------------------------------------------------------------
class RendererAdapter:
    def __init__(self, species_dict: dict[tuple[int, int], "NuclearSpecies"]):
        self.species_dict = species_dict
        self.color_map = self._generate_color_map()
        self.icon_cache = {}
    def _generate_color_map(self):
        decay_modes = list(DecayMode)
        palette = [(random.randint(50,255), random.randint(50,255), random.randint(50,255)) for _ in decay_modes]
        return {mode: palette[i % len(palette)] for i, mode in enumerate(decay_modes)}
    def _load_icon(self, mode: DecayMode):
        filename = f"{mode.name.lower()}_icon.png"
        if os.path.isfile(filename):
            print("Loaded icon file", filename)
            return pygame.image.load(filename).convert_alpha()
        else:
            print("Missing icon file", filename)
        return None
    def _get_icon(self, mode: DecayMode, zoom: float):
        key = (mode, round(zoom,3))
        if key in self.icon_cache:
            return self.icon_cache[key]
        icon = self._load_icon(mode)
        if icon is None:
            self.icon_cache[key] = None
            return None
        size = int(SQUARE_SIZE * zoom * 0.9)
        scaled_icon = pygame.transform.smoothscale(icon, (size, size))
        self.icon_cache[key] = scaled_icon
        return scaled_icon
    def get_render_data(self):
        render_data = []
        for (Z, N), species in self.species_dict.items():
            dominant_decay = species.get_dominant_decay()
            if dominant_decay:
                color = self.color_map.get(dominant_decay.mode, (200,200,200))
                icon_mode = dominant_decay.mode
            else:
                color = (100,100,100)
                icon_mode = None
            render_data.append({
                "pos": (N, Z),
                "size": (0.9, 0.9),
                "color": color,
                "symbol": species.symbol or "?",
                "icon_mode": icon_mode
            })
        return render_data

# APP EXPORT AND VIEW #################################################################################
import pygame
import threading
from typing import Tuple, List
import time
import sys
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60
BACKGROUND_COLOR = (30,30,30)
GRID_COLOR = (60,60,60)
SQUARE_SIZE = 20
zoom_level = 1.0
offset_x = 0
offset_y = 0
running = True
lock = threading.Lock()

#------------------------------------------------------------------------------------
class Viewer:
    def __init__(self, render_data, adapter):
        self.render_data = render_data
        self.adapter = adapter
        self.zoom_level = 1.0
        self.offset_x = 0  # camera center x (world coordinates)
        self.offset_y = 0  # camera center y (world coordinates)
        self.running = True
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("N-Z Diagram Viewer")
    def render(self):
        clock = pygame.time.Clock()
        while self.running:
            with lock:
                self.screen.fill(BACKGROUND_COLOR)
                self.draw_objects()
                pygame.display.flip()
            clock.tick(FPS)
        pygame.quit()
    def draw_objects(self):
        scale = SQUARE_SIZE * self.zoom_level
        min_n = self.offset_x - WINDOW_WIDTH/(2*scale)
        max_n = self.offset_x + WINDOW_WIDTH/(2*scale)
        min_z = self.offset_y - WINDOW_HEIGHT/(2*scale)
        max_z = self.offset_y + WINDOW_HEIGHT/(2*scale)
        for obj in self.render_data:
            N, Z = obj["pos"]
            if not (min_n <= N <= max_n and min_z <= Z <= max_z):
                continue
            self.draw_single_object(obj)
    def draw_single_object(self, obj):
        N, Z = obj["pos"]
        color = obj["color"]
        scale = SQUARE_SIZE * self.zoom_level
        x = int((N - self.offset_x) * scale + WINDOW_WIDTH/2)
        y = int((self.offset_y - Z) * scale + WINDOW_HEIGHT/2)
        size = int(scale * obj["size"][0])
        if -size < x < WINDOW_WIDTH and -size < y < WINDOW_HEIGHT:
            if obj.get("icon_mode") is not None:
                icon_surface = self.adapter._get_icon(obj["icon_mode"], self.zoom_level)
                if icon_surface is not None:
                    self.screen.blit(icon_surface, (x - size//2, y - size//2))
                else:
                    pygame.draw.rect(self.screen, color, pygame.Rect(x - size//2, y - size//2, size, size))
            else:
                pygame.draw.rect(self.screen, color, pygame.Rect(x - size//2, y - size//2, size, size))
    def handle_input(self):
        dragging = False
        last_mouse_pos = None
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                        last_mouse_pos = pygame.mouse.get_pos()
                    elif event.button == 4:
                        self.zoom_level *= 1.1
                    elif event.button == 5:
                        self.zoom_level /= 1.1
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                if event.type == pygame.MOUSEMOTION:
                    if dragging and last_mouse_pos:
                        current_mouse_pos = pygame.mouse.get_pos()
                        scale = SQUARE_SIZE * self.zoom_level
                        dx = (current_mouse_pos[0] - last_mouse_pos[0]) / scale
                        dy = (current_mouse_pos[1] - last_mouse_pos[1]) / scale
                        self.offset_x -= dx
                        self.offset_y += dy
                        last_mouse_pos = current_mouse_pos
            time.sleep(0.01)

#------------------------------------------------------------------------------------
if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    species = parse_nubase_species("nubase_4.mas20.txt")
    adapter = RendererAdapter(species)
    render_data = adapter.get_render_data()
    viewer = Viewer(render_data, adapter)
    render_thread = threading.Thread(target=viewer.render, daemon=True)
    render_thread.start()
    viewer.handle_input()
    render_thread.join()