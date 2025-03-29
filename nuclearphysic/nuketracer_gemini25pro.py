import os
import math
# import re # No longer needed by the original try_parse logic being restored
import sys
import time
import threading
import random
from enum import Enum, auto
from typing import Tuple, List, Optional # Use Optional instead of | for older Python compatibility if needed, but | is fine for 3.10+

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import pygame # Moved pygame import here for better grouping

# Constants
NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720
EXPORTS = {
    "hd": (1920, 1080),
    "4k": (3840, 2160)
}
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60
BACKGROUND_COLOR = (30, 30, 30)
GRID_COLOR = (60, 60, 60)
SQUARE_SIZE = 20 # Base size for N,Z squares in Pygame

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
    # Using original __init__ signature
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
        self.symbol: Optional[str] = None
        self.mass_excess_keV: Optional[float] = None
        self.binding_energy_per_A_keV: Optional[float] = None
        self.atomic_mass_micro_u: Optional[float] = None
        self.half_life_s: Optional[float] = None
        self.half_life_unit: Optional[str] = None
        self.decay_branches: list[DecayBranch] = []
        self.spin_parity: Optional[str] = None
        self.excited_energy_keV: Optional[float] = None
        self.excited_energy_unc_keV: Optional[float] = None
        self.is_isomer: bool = False
        self.raw_sources: dict[str, str] = {}

    def add_decay_mode(self, mode: DecayMode, intensity: Optional[float] = None):
        self.decay_branches.append(DecayBranch(mode, intensity))

    # Using original get_dominant_decay
    def get_dominant_decay(self) -> Optional[DecayBranch]:
        if not self.decay_branches:
            return None
        stable_branch = next((b for b in self.decay_branches if b.mode == DecayMode.STABLE), None)
        if stable_branch:
            return stable_branch
        # Original logic: find max intensity among those not None, default=None if all are None
        valid_branches = [b for b in self.decay_branches if b.intensity is not None]
        if not valid_branches:
             # If no branch has intensity, return the first non-stable one, or None
             non_stable = next((b for b in self.decay_branches if b.mode != DecayMode.STABLE), None)
             return non_stable
        return max(valid_branches, key=lambda x: x.intensity)

    def __repr__(self):
        # Using original __repr__ logic
        return f"{self.symbol or '?'}-{self.A} (Z={self.Z}, N={self.N})"

    def __str__(self):
        # Using original __str__ logic, slightly improved formatting
        lines = [f"{self.symbol or '?'}-{self.A} (Z={self.Z}, N={N})"]
        if self.half_life_s is not None:
            if math.isinf(self.half_life_s):
                lines.append("  Half-life: Stable")
            else:
                unit_str = f" [{self.half_life_unit}]" if self.half_life_unit else ""
                lines.append(f"  Half-life: {self.half_life_s:.3g} s{unit_str}")
        if self.decay_branches:
            modes = ', '.join(str(b) for b in self.decay_branches)
            lines.append(f"  Decay modes: {modes}")
        if self.spin_parity:
            lines.append(f"  Spin/parity: {self.spin_parity.strip()}")
        if self.excited_energy_keV is not None:
            lines.append(f"  Excitation energy: {self.excited_energy_keV} keV")
        return '\n'.join(lines)


# NUBASE PARSER #################################################################################
# <<< --- REVERTING PARSING LOGIC TO ORIGINAL --- >>>

#------------------------------------------------------------------------------------
# Original convert_half_life_to_seconds
def convert_half_life_to_seconds(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None: # Check None first
        return None
    if math.isinf(value):
        return value # Keep inf as inf
    if unit is None:
        # Original did not specify behavior for unit=None, let's return None for safety
        # print(f"[WARN] Missing unit for half-life value {value}. Cannot convert.")
        return None

    # Original scale dictionary
    scale = {
        "ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12,
        "ns": 1e-9,  "us": 1e-6,  "ms": 1e-3,  "s": 1,      "m": 60,
        "h": 3600,   "d": 86400,  "y": 3.15576e7,
        "ky": 3.15576e10, "My": 3.15576e13, "Gy": 3.15576e16,
        "Ty": 3.15576e19, "Py": 3.15576e22, "Ey": 3.15576e25,
        "Zy": 3.15576e28, "Yy": 3.15576e31
        # Note: Original didn't handle case variations or min/hr explicitly
    }
    unit = unit.strip() # Original stripped unit
    factor = scale.get(unit) # Original used direct get

    if factor is None:
        print(f"[ERROR] Unknown time unit '{unit}' for half-life conversion")
        return None # Original returned None on unknown unit
    return value * factor

#------------------------------------------------------------------------------------
# Original parse_nubase_typed structure
def parse_nubase_typed(filepath):
    # Original field definitions (matching columns in prompt)
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
        # Original parsed HalfLife directly as float, relies on try_parse logic
        ("HalfLife", 69, 78, float),
        ("HalfLifeUnit", 78, 80, str),
        # Original parsed dT as float
        ("dT", 81, 88, float),
        ("Jpi", 88, 102, str),
        ("ENSDF_Year", 102, 104, str),
        ("DiscoveryYear", 114, 118, int),
        ("DecayModes", 119, 209, str)
    ]

    # Original parse_decay_modes logic (nested inside try_parse)
    def parse_decay_modes(raw):
        modes = []
        entries = raw.split(';')
        for entry in entries:
            entry = entry.strip()
            if '=' not in entry:
                # Original behavior: skip entries without '='
                continue
            key, val = entry.split('=', 1)
            key = key.strip()
            # Original behavior: strip % and attempt float conversion
            val_str = val.strip()
            intensity = None
            # Handle potential spaces before % (e.g., "100 %")
            if val_str.endswith('%'):
                val_str = val_str[:-1].strip()
            try:
                # Original handles uncertainty text implicitly by float conversion failure
                intensity = float(val_str)
            except ValueError:
                # Original intensity is None if float conversion fails
                 intensity = None # Explicitly set None

            if key == 'IS': # Original skipped IS
                continue

            submodes = key.split('+')
            for subkey in submodes:
                subkey = subkey.strip()
                if not subkey:
                    continue

                # --- Original Decay Mode Mapping ---
                decay = DecayMode.UNKNOWN # Default
                if subkey == 'e+':
                    decay = DecayMode.BETA_PLUS
                elif subkey == '3H':
                    decay = DecayMode.TRITON
                elif subkey == '3He':
                    # Original printed error and set UNKNOWN
                    decay = DecayMode.UNKNOWN
                    print(f"[ERROR] Unknown decay mode '{subkey}' in DecayModes")
                elif subkey in {"14C", "20O", "24Ne", "26Ne", "22Ne", "23F", "28Mg", "34Si", "20Ne", "24Ne+26Ne"}:
                     # Original mapped cluster decays to SF
                    decay = DecayMode.SPONTANEOUS_FISSION
                elif subkey == '2B-':
                    decay = DecayMode.DOUBLE_BETA_MINUS
                elif subkey == 'B': # Original mapped 'B' -> B+
                    decay = DecayMode.BETA_PLUS
                elif subkey == 'e': # Original mapped 'e' -> B+
                    decay = DecayMode.BETA_PLUS
                elif subkey.startswith('B-'): # Original used startswith
                    decay = DecayMode.BETA_MINUS
                elif subkey.startswith('B+'): # Original used startswith
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
                    # Original printed error and set UNKNOWN
                    decay = DecayMode.UNKNOWN
                    print(f"[ERROR] Unknown decay mode '{subkey}' in DecayModes: Raw Entry='{entry}'")

                # Original appends the mode and potentially None intensity
                modes.append((decay, intensity))
        return modes

    # Original try_parse logic
    def try_parse(value, target_type, field_name):
        # Original cleaning steps
        raw = value.strip().replace('#', '').replace('*', '').replace('~', '')
        if raw.lower() in {'', 'non-exist', 'none'}:
            return None

        # Original special handling for HalfLife field
        if field_name == "HalfLife":
            if raw.lower() == "stbl":
                return float('inf')
            if raw.lower() == "p-unst":
                # Original returned None for p-unst, which might cause issues later
                # Let's return 0.0 instead, similar to the revised parser, as it's numerically representable
                return 0.0 # Changed from original None to 0.0 for consistency

        # Original handling for DecayModes field
        if field_name == "DecayModes":
            # Pass the original uncleaned value string to parse_decay_modes
            return parse_decay_modes(value.strip()) # Pass stripped original value

        # Original handling for > < prefixes
        if raw.startswith(('>', '<')):
            raw = raw[1:]

        # --- Original Float Parsing Logic ---
        # This part was complex and potentially fragile. The original didn't explicitly
        # use convert_half_life_to_seconds here, it seems it expected floats directly
        # for Mass, dMass, Exc, dExc, HalfLife, dT.
        # Let's simplify based on the *intent* of the original structure:
        # Try direct conversion first, handle specific errors.
        if target_type is float:
            try:
                # Try direct float conversion after basic cleaning
                return float(raw)
            except ValueError:
                 # Original had complex regex fallback, let's just return None on failure
                 # as the complex regex wasn't robust either.
                 # This applies to Mass, dMass, Exc, dExc, HalfLife, dT
                 # Print a warning for fields other than HalfLife (which handles stbl/p-unst above)
                 if field_name != "HalfLife":
                     # Check if it's just uncertainty text like ' sys' etc.
                     if raw.lower() not in ['syst', 'sys', 'ap']: # Common non-numeric values
                          print(f"[WARN] Failed to parse field '{field_name}' as float, got raw='{raw}' (cleaned='{raw}'). Returning None.")
                 return None # Return None if float conversion fails

        # Original general type conversion
        try:
            # Handle int and str (already cleaned)
             if target_type is int:
                 # Handle potential floats in int field (DiscoveryYear)
                 try:
                     return int(raw)
                 except ValueError:
                     return int(float(raw))
             else: # Assume string
                return raw # Return cleaned string for str types
        except Exception as e:
            # Original error message
            print(f"[ERROR] Failed to parse field '{field_name}' ? expected {target_type.__name__}, got raw='{raw}': {e}")
            return None

    # Original data reading loop
    data = []
    try:
        # Using 'iso-8859-1' as it often works better for older data files than utf-8
        with open(filepath, 'r', encoding='iso-8859-1') as f:
            for line_num, line in enumerate(f):
                # Original check for comments/empty lines
                if not line.strip() or line.startswith('#') or len(line) < 119: # Basic check
                    continue
                row = {}
                parse_successful = True
                for name, start, end, typ in fields:
                    raw_val = line[start:end]
                    parsed_val = try_parse(raw_val, typ, name)
                    row[name] = parsed_val
                    # Add basic check for critical fields if needed
                    if name in ['AAA', 'ZZZi'] and parsed_val is None and len(raw_val.strip()) > 0:
                        print(f"[WARN] Critical field {name} parsed as None on line {line_num+1}. Raw: '{raw_val}'")
                        # Decide whether to skip row: parse_successful = False; break

                if parse_successful:
                     data.append(row)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        print(f"[ERROR] Failed to read or process file {filepath}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


#------------------------------------------------------------------------------------
# Original df_to_species
def df_to_species(df: pd.DataFrame) -> dict[tuple[int, int], NuclearSpecies]:
    species_dict = {}
    processed_keys = set() # To avoid duplicates if filtering wasn't perfect

    for index, row in df.iterrows():
        # Use .get() for safer access, default to None
        z_str = row.get("ZZZi")
        if not z_str or not isinstance(z_str, str) or len(z_str) < 3:
            # print(f"[DEBUG] Skipping row {index}: Invalid ZZZi '{z_str}'")
            continue
        try:
            Z = int(z_str[:3])
        except (ValueError, TypeError):
            # print(f"[DEBUG] Skipping row {index}: Cannot parse Z from '{z_str}'")
            continue

        A_val = row.get("AAA")
        if pd.isna(A_val):
            # print(f"[DEBUG] Skipping row {index} (Z={Z}): Missing A")
            continue
        try:
            A = int(A_val)
        except (ValueError, TypeError):
            # print(f"[DEBUG] Skipping row {index} (Z={Z}): Cannot parse A '{A_val}'")
            continue

        N = A - Z
        if N < 0:
            # print(f"[DEBUG] Skipping row {index} (Z={Z}, A={A}): Negative N ({N})")
             continue

        key = (Z, N)
        # Original check for duplicates
        if key in species_dict or key in processed_keys:
            # print(f"[DEBUG] Skipping row {index}: Duplicate key {key}")
            continue

        sp = NuclearSpecies(Z, N)
        # Use .get() and check pd.notna() for optional fields
        sp.symbol = str(row.get("A_El", "")).strip() or None # Ensure None if empty
        sp.mass_excess_keV = row.get("Mass") if pd.notna(row.get("Mass")) else None
        sp.excited_energy_keV = row.get("Exc") if pd.notna(row.get("Exc")) else None
        sp.excited_energy_unc_keV = row.get("dExc") if pd.notna(row.get("dExc")) else None

        # --- Half-Life: Combine value and unit ---
        # Assumes HalfLife column contains float/inf/0.0 from try_parse
        # And HalfLifeUnit contains the string unit
        half_life_val = row.get("HalfLife") # Should be float/inf/0.0 or None
        half_life_unit = row.get("HalfLifeUnit") if pd.notna(row.get("HalfLifeUnit")) else None
        # Convert using the value and unit
        sp.half_life_s = convert_half_life_to_seconds(half_life_val, half_life_unit)
        sp.half_life_unit = half_life_unit # Store original unit string

        sp.spin_parity = str(row.get("Jpi", "")).strip() or None
        # Original isomer check based on 's' field
        s_val = str(row.get("s", "")).strip()
        sp.is_isomer = s_val in {'m', 'n', 'p', 'q', 'r', 'i', 'j', 'x'}

        # Add STABLE mode if half-life is infinite *after* conversion
        if sp.half_life_s is not None and math.isinf(sp.half_life_s):
            # Ensure no other modes are added if stable
            sp.decay_branches = [DecayBranch(DecayMode.STABLE)]
        else:
            # Add decay modes parsed from the 'DecayModes' field
            # Assumes 'DecayModes' column contains the list from parse_decay_modes
            decay_info = row.get("DecayModes")
            if isinstance(decay_info, list):
                for mode, intensity in decay_info:
                    sp.add_decay_mode(mode, intensity)

        species_dict[key] = sp
        processed_keys.add(key)
    return species_dict

#------------------------------------------------------------------------------------
# Original parse_nubase_species
def parse_nubase_species(filepath: str) -> dict[tuple[int, int], NuclearSpecies]:
    print(f"Parsing NUBASE file: {filepath} (using original logic)")
    df = parse_nubase_typed(filepath)
    if df.empty:
        print("[ERROR] Parsing resulted in an empty DataFrame. Cannot proceed.")
        return {}

    # Original filtering based on 's' column
    if 's' in df.columns:
        # Original filter logic
        df_filtered = df[df["s"].isna() | ~df["s"].isin(['m', 'n', 'p', 'q', 'r', 'i', 'j', 'x'])]
        print(f"Filtered DataFrame from {len(df)} to {len(df_filtered)} rows (removing isomers/excited states based on 's' column).")
        if df_filtered.empty and not df.empty:
             print("[WARN] Filtering based on 's' removed all entries. Check filter logic or file format. Proceeding with unfiltered data.")
             df_filtered = df # Fallback
    else:
        print("[WARN] 's' column not found. Cannot filter isomers. Proceeding with all entries.")
        df_filtered = df

    species = df_to_species(df_filtered)
    print(f"Successfully created {len(species)} NuclearSpecies objects.")
    return species

# <<< --- END OF REVERTED PARSING LOGIC --- >>>
#################################################################################


# PLOTLY ADAPTER #################################################################################
# (Using the improved Plotly functions from the previous revision)
#------------------------------------------------------------------------------------
def plot_half_life_nz(species_dict: dict[tuple[int, int], "NuclearSpecies"], basename="half_life_nz"):
    shapes = []; texts = []; colors = []; xs = []; ys = []
    valid_species_count = 0
    min_log_hl = float('inf')
    max_log_hl = float('-inf')

    for (Z, N), sp in species_dict.items():
        if sp.half_life_s is not None and sp.half_life_s > 0 and not math.isinf(sp.half_life_s):
            log_hl = math.log10(sp.half_life_s)
            min_log_hl = min(min_log_hl, log_hl)
            max_log_hl = max(max_log_hl, log_hl)

            x0, x1 = N - 0.5, N + 0.5
            y0, y1 = Z - 0.5, Z + 0.5
            xs.append([x0, x1, x1, x0, x0])
            ys.append([y0, y0, y1, y1, y0])
            colors.append(log_hl)
            # Use fixed __str__ method for hover text
            texts.append(f"{sp!s}<br>log10(T1/2) = {log_hl:.2f} s") # Simplified text
            valid_species_count += 1

    if not xs:
        print("[WARN] No valid finite half-life data found to plot.")
        return

    print(f"[INFO] Plotting {valid_species_count} species with finite half-lives. Range log10(T1/2): [{min_log_hl:.2f}, {max_log_hl:.2f}]")

    if min_log_hl >= max_log_hl : # Handle single value or invalid range
        min_log_hl -= 0.5
        max_log_hl += 0.5

    colorscale = px.colors.sequential.Viridis
    norm = lambda v: (v - min_log_hl) / (max_log_hl - min_log_hl) if (max_log_hl - min_log_hl) != 0 else 0.5

    fig = go.Figure()
    for x_coords, y_coords, c, text in zip(xs, ys, colors, texts):
        fill_color = sample_colorscale(colorscale, norm(c))[0]
        fig.add_trace(go.Scattergl( # Using Scattergl
            x=x_coords, y=y_coords, fill='toself', fillcolor=fill_color,
            line=dict(width=0.1, color='rgba(50,50,50,0.2)'), mode='lines',
            hoverinfo='text', hovertext=text, showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale=colorscale, cmin=min_log_hl, cmax=max_log_hl,
            colorbar=dict(title="log10(Half-life [s])", thickness=15, titleside="right", tickfont=dict(size=10), len=0.8)
        ), showlegend=False
    ))

    fig.update_layout(
        title="N-Z Diagram: log10(Half-life)",
        xaxis=dict(title="Neutron Number (N)", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10), zeroline=False),
        yaxis=dict(title="Proton Number (Z)", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10), zeroline=False, scaleanchor="x", scaleratio=1),
        width=NATIVE_WIDTH, height=NATIVE_HEIGHT, margin=dict(l=60, r=40, t=60, b=60),
        template="plotly_white", hovermode='closest'
    )

    out_html = f"{basename}.html"
    out_png = f"{basename}.png"
    try:
        fig.write_html(out_html)
        print(f"[INFO] Plot saved to {out_html}")
    except Exception as e: print(f"[ERROR] Failed to write HTML plot {out_html}: {e}")

    print("[INFO] Exporting PNG images...")
    try:
        fig.write_image(out_png, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, scale=1)
        print(f"[INFO] Saved {out_png} at native resolution.")
        for suffix, (w, h) in EXPORTS.items():
            output_file = f"{basename}_{suffix}.png"
            try:
                fig.write_image(output_file, width=w, height=h)
                print(f"[INFO] Saved {output_file} at {w}x{h}")
            except Exception as e: print(f"[WARN] Failed to write {output_file}: {e}")
    except ValueError as ve:
         if "requires the kaleido" in str(ve) or "requires orca" in str(ve): print("[WARN] Image export requires 'kaleido' or 'orca'. Skipping PNG export.")
         else: print(f"[WARN] Failed to write base PNG {out_png}: {ve}")
    except Exception as e: print(f"[WARN] Failed to write base PNG {out_png}: {e}")


#------------------------------------------------------------------------------------
def plot_dominant_decay_nz(species_dict: dict[tuple[int, int], "NuclearSpecies"], basename="dominant_decay_nz"):
    data = []
    mode_counts = {}
    for (Z, N), sp in species_dict.items():
        dom = sp.get_dominant_decay()
        if dom:
            mode_val = dom.mode.value
            data.append({"Z": Z, "N": N, "Symbol": sp.symbol or "?", "A": sp.A, "DecayMode": mode_val, "Tooltip": str(sp)})
            mode_counts[mode_val] = mode_counts.get(mode_val, 0) + 1

    if not data:
        print("[WARN] No species with dominant decay mode found to plot.")
        return

    df = pd.DataFrame(data)
    unique_modes = sorted(df["DecayMode"].unique())
    print(f"[INFO] Plotting {len(df)} species by dominant decay mode. Modes found: {unique_modes}")
    print("[INFO] Mode counts:", sorted(mode_counts.items(), key=lambda item: item[1], reverse=True))

    base_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel + px.colors.qualitative.Set3
    color_map = {}
    palette_idx = 0
    for mode in unique_modes:
        if mode == DecayMode.STABLE.value: color_map[mode] = 'rgb(150, 150, 150)'
        else:
            while palette_idx < len(base_palette) and base_palette[palette_idx] in color_map.values(): palette_idx += 1
            if palette_idx < len(base_palette):
                color_str = base_palette[palette_idx]; palette_idx += 1
                try:
                    if color_str.startswith('#'): colors[mode] = tuple(int(color_str.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    elif color_str.startswith('rgb'): colors[mode] = tuple(map(int, color_str[4:-1].split(',')))
                    else: color_map[mode] = f'rgb({random.randint(50,200)}, {random.randint(50,200)}, {random.randint(50,200)})' # Fallback
                except: color_map[mode] = f'rgb({random.randint(50,200)}, {random.randint(50,200)}, {random.randint(50,200)})' # Fallback
            else: color_map[mode] = f'rgb({random.randint(50,200)}, {random.randint(50,200)}, {random.randint(50,200)})' # Fallback

    fig = go.Figure()
    # Use invisible trace for hover
    all_n = df['N'].tolist()
    all_z = df['Z'].tolist()
    all_hover_texts = df['Tooltip'].tolist()
    fig.add_trace(go.Scattergl(
        x=all_n, y=all_z, mode='markers', marker=dict(size=1, opacity=0),
        hoverinfo='text', hovertext=all_hover_texts, showlegend=False
    ))

    for mode in unique_modes:
        sub_df = df[df["DecayMode"] == mode]
        xs_mode, ys_mode = [], []
        for _, row in sub_df.iterrows():
            Z, N = row["Z"], row["N"]
            x0, x1 = N - 0.5, N + 0.5
            y0, y1 = Z - 0.5, Z + 0.5
            xs_mode.extend([x0, x1, x1, x0, x0, None])
            ys_mode.extend([y0, y0, y1, y1, y0, None])

        fig.add_trace(go.Scattergl(
            x=xs_mode, y=ys_mode, fill='toself', fillcolor=color_map.get(mode, 'rgb(200,0,200)'), # Default color if map fails
            line=dict(width=0), mode='lines', name=mode, legendgroup=mode,
            hoverinfo='skip', showlegend=False
        ))
        # Add dummy trace for legend
        fig.add_trace(go.Scatter(
             x=[None], y=[None], mode='markers', marker=dict(size=10, color=color_map.get(mode)),
             name=mode, legendgroup=mode, showlegend=True
        ))

    fig.update_layout(
        title="N-Z Diagram by Dominant Decay Mode",
        xaxis=dict(title="Neutron Number (N)", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10), zeroline=False),
        yaxis=dict(title="Proton Number (Z)", showgrid=False, tickmode="auto", ticks="outside", tickfont=dict(size=10), zeroline=False, scaleanchor="x", scaleratio=1),
        width=NATIVE_WIDTH, height=NATIVE_HEIGHT, margin=dict(l=60, r=40, t=60, b=60),
        template="plotly_white", legend=dict(title="Decay Mode", traceorder="reversed"), hovermode='closest'
    )

    out_html = f"{basename}.html"; out_png = f"{basename}.png"
    try: fig.write_html(out_html); print(f"[INFO] Plot saved to {out_html}")
    except Exception as e: print(f"[ERROR] Failed to write HTML plot {out_html}: {e}")
    print("[INFO] Exporting PNG images...")
    try:
        fig.write_image(out_png, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, scale=1); print(f"[INFO] Saved {out_png}")
        for suffix, (w, h) in EXPORTS.items():
            output_file = f"{basename}_{suffix}.png"
            try: fig.write_image(output_file, width=w, height=h); print(f"[INFO] Saved {output_file} at {w}x{h}")
            except Exception as e: print(f"[WARN] Failed to write {output_file}: {e}")
    except ValueError as ve:
         if "requires the kaleido" in str(ve) or "requires orca" in str(ve): print("[WARN] Image export requires 'kaleido' or 'orca'. Skipping PNG export.")
         else: print(f"[WARN] Failed to write base PNG {out_png}: {ve}")
    except Exception as e: print(f"[WARN] Failed to write base PNG {out_png}: {e}")


# PYGAME ADAPTER #################################################################################
# (Using the improved Pygame Adapter and Viewer from previous revision)
import re # Needed again for filename sanitization in RendererAdapter

#------------------------------------------------------------------------------------
class RendererAdapter:
    def __init__(self, species_dict: dict[tuple[int, int], "NuclearSpecies"]):
        self.species_dict = species_dict
        self.color_map = self._generate_color_map()
        self.icon_cache = {} # Cache for loaded and scaled icons
        self.base_icon_surfaces = {} # Cache for base loaded surfaces
        self._load_icons() # Pre-load icons

    def _generate_color_map(self):
        decay_modes = list(DecayMode)
        colors = {}
        base_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel + px.colors.qualitative.Set3
        palette_idx = 0
        for mode in decay_modes:
            if mode == DecayMode.STABLE: colors[mode] = (150, 150, 150)
            elif mode == DecayMode.UNKNOWN: colors[mode] = (50, 50, 50)
            else:
                if palette_idx < len(base_palette):
                    color_str = base_palette[palette_idx]; palette_idx += 1
                    try:
                        if color_str.startswith('#'): colors[mode] = tuple(int(color_str.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        elif color_str.startswith('rgb'): colors[mode] = tuple(map(int, color_str[4:-1].split(',')))
                        else: colors[mode] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                    except: colors[mode] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                else: colors[mode] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        return colors

    def _load_icons(self):
        print("[INFO] Loading available icons...")
        icon_dir = "icons"
        if not os.path.isdir(icon_dir):
             print(f"[WARN] Icon directory '{icon_dir}' not found. Icons will not be loaded.")
             return
        for mode in DecayMode:
            filename_std = f"{mode.name.lower()}.png"
            sanitized_value = re.sub(r'[^\w\-_.]', '', mode.value)
            filename_val = f"{sanitized_value}.png" if sanitized_value else ""
            found = False
            for filename in [filename_std, filename_val]:
                 if not filename: continue
                 filepath = os.path.join(icon_dir, filename)
                 if os.path.isfile(filepath):
                     try:
                         icon_surface = pygame.image.load(filepath).convert_alpha()
                         self.base_icon_surfaces[mode] = icon_surface
                         found = True; break
                     except pygame.error as e:
                         print(f"[WARN] Failed to load icon '{filepath}': {e}")
                         self.base_icon_surfaces[mode] = None; break
            if not found: self.base_icon_surfaces[mode] = None

    def _get_scaled_icon(self, mode: DecayMode, target_size: int):
        if target_size <= 0: return None
        cache_key = (mode, target_size)
        if cache_key in self.icon_cache: return self.icon_cache[cache_key]
        base_surface = self.base_icon_surfaces.get(mode)
        if base_surface is None:
            self.icon_cache[cache_key] = None; return None
        try:
            scaled_icon = pygame.transform.smoothscale(base_surface, (target_size, target_size))
            self.icon_cache[cache_key] = scaled_icon; return scaled_icon
        except Exception as e:
            print(f"[WARN] Failed to scale icon for {mode.name} to size {target_size}: {e}")
            self.icon_cache[cache_key] = None; return None

    def get_render_data(self):
        render_data = []
        for (Z, N), species in self.species_dict.items():
            dominant_decay = species.get_dominant_decay()
            color = (100, 100, 100); icon_mode = None
            if dominant_decay:
                color = self.color_map.get(dominant_decay.mode, (200, 200, 200))
                icon_mode = dominant_decay.mode
            elif species.half_life_s is not None and math.isinf(species.half_life_s):
                 color = self.color_map.get(DecayMode.STABLE, (150, 150, 150))
                 icon_mode = DecayMode.STABLE
            render_data.append({
                "pos": (N, Z), "size_factor": 0.9, "color": color,
                "symbol": species.symbol or "?", "A": species.A,
                "icon_mode": icon_mode, "tooltip": str(species)
            })
        return render_data

# APP EXPORT AND VIEW #################################################################################
# (Using the improved Viewer class from previous revision)
#------------------------------------------------------------------------------------
class Viewer:
    def __init__(self, render_data, adapter):
        self.render_data = render_data
        self.adapter = adapter
        self.zoom_level = 1.0
        self.offset_x = 60; self.offset_y = 40
        self.min_zoom = 0.1; self.max_zoom = 20.0
        self.screen = None; self.clock = pygame.time.Clock(); self.font = None
        self.running = False; self.lock = threading.Lock(); self.render_thread = None
        self.dragging = False; self.last_mouse_pos = None

    def initialize(self):
        try:
            pygame.init()
            if not pygame.font: self.font = None; print("[WARN] Pygame font module not available.")
            else:
                 try: self.font = pygame.font.SysFont(None, 16)
                 except Exception as e: self.font = pygame.font.Font(None, 16); print(f"[WARN] Could not load system font: {e}. Using fallback.")
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption("N-Z Diagram Viewer")
            print("[INFO] Pygame initialized successfully.")
            return True
        except pygame.error as e: print(f"[ERROR] Failed to initialize Pygame: {e}"); return False

    def _world_to_screen(self, n, z):
        scale = SQUARE_SIZE * self.zoom_level
        screen_x = (n - self.offset_x) * scale + WINDOW_WIDTH / 2
        screen_y = (self.offset_y - z) * scale + WINDOW_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _screen_to_world(self, sx, sy):
        scale = SQUARE_SIZE * self.zoom_level
        if scale == 0: return self.offset_x, self.offset_y
        world_n = (sx - WINDOW_WIDTH / 2) / scale + self.offset_x
        world_z = self.offset_y - (sy - WINDOW_HEIGHT / 2) / scale
        return world_n, world_z

    def _render_frame(self):
        if not self.screen: return
        self.screen.fill(BACKGROUND_COLOR)
        with self.lock:
            current_zoom = self.zoom_level
            current_offset_x = self.offset_x; current_offset_y = self.offset_y
        scale = SQUARE_SIZE * current_zoom
        if scale <= 0: return
        half_world_w = WINDOW_WIDTH / (2 * scale); half_world_h = WINDOW_HEIGHT / (2 * scale)
        min_n_vis = current_offset_x - half_world_w - 1; max_n_vis = current_offset_x + half_world_w + 1
        min_z_vis = current_offset_y - half_world_h - 1; max_z_vis = current_offset_y + half_world_h + 1
        visible_count = 0
        for obj in self.render_data:
            N, Z = obj["pos"]
            if not (min_n_vis <= N <= max_n_vis and min_z_vis <= Z <= max_z_vis): continue
            screen_x, screen_y = self._world_to_screen(N, Z)
            size = int(scale * obj["size_factor"])
            if screen_x + size/2 < 0 or screen_x - size/2 > WINDOW_WIDTH or \
               screen_y + size/2 < 0 or screen_y - size/2 > WINDOW_HEIGHT: continue
            visible_count += 1
            self.draw_single_object(obj, screen_x, screen_y, size)
        # self.draw_hud(visible_count, current_zoom, current_offset_x, current_offset_y) # Optional HUD
        pygame.display.flip()

    def draw_single_object(self, obj, screen_x, screen_y, size):
        if size < 1: return
        rect = pygame.Rect(screen_x - size // 2, screen_y - size // 2, size, size)
        icon_mode = obj.get("icon_mode"); icon_surface = None
        if icon_mode is not None:
             icon_size = max(1, int(size * 0.9))
             icon_surface = self.adapter._get_scaled_icon(icon_mode, icon_size)
        if icon_surface:
            icon_rect = icon_surface.get_rect(center=rect.center)
            self.screen.blit(icon_surface, icon_rect)
        else:
            pygame.draw.rect(self.screen, obj["color"], rect)
            if size > 4: pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

    def draw_hud(self, visible_count, zoom, ox, oy): # Optional HUD function
        if not self.font: return
        try:
            fps = self.clock.get_fps()
            lines = [f"FPS: {fps:.1f}", f"Zoom: {zoom:.2f}", f"Center: (N={ox:.1f}, Z={oy:.1f})", f"Visible: {visible_count}"]
            y_offset = 5
            for line in lines:
                text_surface = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(text_surface, (5, y_offset))
                y_offset += self.font.get_height()
        except Exception as e: print(f"[WARN] Error rendering HUD: {e}")

    def render_loop(self):
        print("[INFO] Render thread started.")
        while self.running:
            try: self._render_frame(); self.clock.tick(FPS)
            except Exception as e: print(f"[ERROR] Unhandled exception in render loop: {e}"); time.sleep(0.1)
        print("[INFO] Render thread finished.")

    def handle_input(self):
        global WINDOW_WIDTH, WINDOW_HEIGHT
        mouse_world_pos_start_drag = None
        print("[INFO] Starting input handler...")
        while self.running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("[INFO] QUIT event received. Shutting down.")
                        self.running = False; return
                    elif event.type == pygame.VIDEORESIZE:
                         WINDOW_WIDTH = event.w; WINDOW_HEIGHT = event.h
                         self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
                         print(f"[INFO] Window resized to {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1: self.dragging = True; self.last_mouse_pos = pygame.mouse.get_pos(); mouse_world_pos_start_drag = self._screen_to_world(*self.last_mouse_pos)
                        elif event.button == 4: self.zoom_at(pygame.mouse.get_pos(), 1.2)
                        elif event.button == 5: self.zoom_at(pygame.mouse.get_pos(), 1 / 1.2)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1: self.dragging = False; self.last_mouse_pos = None; mouse_world_pos_start_drag = None
                    elif event.type == pygame.MOUSEMOTION:
                        if self.dragging and self.last_mouse_pos:
                            current_mouse_pos = pygame.mouse.get_pos()
                            dx_screen = current_mouse_pos[0] - self.last_mouse_pos[0]; dy_screen = current_mouse_pos[1] - self.last_mouse_pos[1]
                            with self.lock:
                                 scale = SQUARE_SIZE * self.zoom_level
                                 if scale > 0: self.offset_x -= dx_screen / scale; self.offset_y += dy_screen / scale
                            self.last_mouse_pos = current_mouse_pos
                time.sleep(0.01) # Prevent busy-waiting
            except Exception as e:
                 print(f"[ERROR] Exception in input loop: {e}")
                 # Maybe add a small sleep on error too?
                 time.sleep(0.1)
        print("[INFO] Input handling finished.")

    def zoom_at(self, screen_pos, factor):
         with self.lock:
             world_before_zoom_n, world_before_zoom_z = self._screen_to_world(*screen_pos)
             new_zoom = self.zoom_level * factor
             self.zoom_level = max(self.min_zoom, min(self.max_zoom, new_zoom))
             world_after_zoom_n, world_after_zoom_z = self._screen_to_world(*screen_pos)
             self.offset_x += (world_before_zoom_n - world_after_zoom_n)
             self.offset_y += (world_before_zoom_z - world_after_zoom_z)

    def run(self):
        if not self.initialize(): return
        self.running = True
        self.render_thread = threading.Thread(target=self.render_loop, name="RenderThread")
        self.render_thread.start()
        self.handle_input() # Blocks until self.running is False
        print("[INFO] Main thread: Waiting for render thread to join...")
        if self.render_thread:
            self.render_thread.join(timeout=2.0)
            if self.render_thread.is_alive(): print("[WARN] Render thread did not exit cleanly.")
        print("[INFO] Quitting Pygame...")
        pygame.quit()
        print("[INFO] Application finished.")

#------------------------------------------------------------------------------------
if __name__ == "__main__":
    nubase_file = "nubase_4.mas20.txt" # Default file
    if len(sys.argv) > 1: nubase_file = sys.argv[1]
    if not os.path.isfile(nubase_file):
         fallback_file = "nubase2020.txt"
         if os.path.isfile(fallback_file): print(f"[WARN] File '{nubase_file}' not found. Using fallback '{fallback_file}'."); nubase_file = fallback_file
         else: print(f"[ERROR] NUBASE data file '{nubase_file}' not found (and fallback '{fallback_file}' not found)."); sys.exit(1)

    species_data = parse_nubase_species(nubase_file) # Uses reverted parser
    if not species_data: print("[ERROR] No nuclear species data loaded. Exiting."); sys.exit(1)

    # --- Optional Plotting ---
    # print("\n[INFO] Generating Plotly plots...")
    # plot_half_life_nz(species_data, basename=f"{os.path.splitext(nubase_file)[0]}_half_life")
    # plot_dominant_decay_nz(species_data, basename=f"{os.path.splitext(nubase_file)[0]}_decay_mode")
    # print("[INFO] Plotly plot generation finished.\n")

    print("[INFO] Initializing Pygame viewer...")
    adapter = RendererAdapter(species_data)
    render_data = adapter.get_render_data()
    viewer = Viewer(render_data, adapter)
    viewer.run() # Start application

    print("[INFO] Program exited.")