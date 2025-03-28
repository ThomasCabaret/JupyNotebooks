from enum import Enum, auto
import pandas as pd
import plotly.express as px
import math

NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720

EXPORTS = {
    "hd": (1920, 1080),
    "4k": (3840, 2160)
}

#------------------------------------------------------------------------------------
class DecayMode(Enum):
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

#------------------------------------------------------------------------------------
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
        if field_name == "HalfLife" and raw.lower() in {'stbl', 'p-unst'}:
            return None
        if field_name == "DecayModes":
            return parse_decay_modes(raw)
        if raw.startswith(('>', '<')):
            #print(f"[WARN] Truncated/limit value in field '{field_name}': raw='{raw}'")
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
                    #print(f"[WARN] Interpreting ambiguous 'B' as BETA_PLUS")
                elif subkey == 'e':
                    decay = DecayMode.BETA_PLUS
                    #print(f"[WARN] Interpreting ambiguous 'e' as BETA_PLUS")
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

#------------------------------------------------------------------------------------
import plotly.express as px
import math
import pandas as pd

#------------------------------------------------------------------------------------
NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720

EXPORTS = {
    "hd": (1920, 1080),
    "4k": (3840, 2160)
}

#------------------------------------------------------------------------------------
def plot_half_life_nz(species_dict: dict[tuple[int, int], "NuclearSpecies"], basename="half_life_nz"):
    data = []
    for (Z, N), sp in species_dict.items():
        if sp.half_life_s is not None and sp.half_life_s > 0:
            log_hl = math.log10(sp.half_life_s)
            data.append({
                "Z": Z,
                "N": N,
                "Symbol": sp.symbol or "?",
                "HalfLife_s": sp.half_life_s,
                "log10_HalfLife": log_hl
            })
    if not data:
        print("[WARN] No species with defined half-life to plot")
        return
    df = pd.DataFrame(data)
    fig = px.scatter(
        df,
        x="N",
        y="Z",
        color="log10_HalfLife",
        color_continuous_scale="Viridis",
        hover_name="Symbol",
        hover_data={"HalfLife_s": True, "log10_HalfLife": True, "N": False, "Z": False},
        title="N-Z Diagram Colored by log10(Half-life in seconds)"
    )
    fig.update_traces(marker=dict(size=6, symbol="square"))
    fig.update_layout(
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        xaxis_title="Neutron Number (N)",
        yaxis_title="Proton Number (Z)",
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
            data.append({
                "Z": Z,
                "N": N,
                "Symbol": sp.symbol or "?",
                "DecayMode": dom.mode.value
            })
    if not data:
        print("[WARN] No species with identifiable dominant decay mode to plot")
        return
    df = pd.DataFrame(data)
    fig = px.scatter(
        df,
        x="N",
        y="Z",
        color="DecayMode",
        symbol="DecayMode",
        hover_name="Symbol",
        hover_data={"DecayMode": True, "N": False, "Z": False},
        title="N-Z Diagram Colored by Dominant Decay Mode"
    )
    fig.update_traces(marker=dict(size=6, symbol="square"))
    fig.update_layout(
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        xaxis_title="Neutron Number (N)",
        yaxis_title="Proton Number (Z)",
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
if __name__ == "__main__":
    species = parse_nubase_species("nubase_4.mas20.txt")
    plot_half_life_nz(species, basename="nuclear_half_life")
    plot_dominant_decay_nz(species, basename="nuclear_decay_mode")