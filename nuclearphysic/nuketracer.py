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
            return DecayBranch(DecayMode.UNKNOWN, 0.0)
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
#     column   quantity   format      description
#       1: 3   AAA           a3       Mass Number (AAA)
#       5: 8   ZZZi          a4       Atomic Number (ZZZ); i=0 (gs); i=1,2 (isomers); i=3,4 (levels); i=5 (resonance); i=8,9 (IAS)
#                                     i=3,4,5,6 can also indicate isomers (when more than two isomers are presented in a nuclide)
#     12: 16   A El          a5       A Element 
#     17: 17   s             a1       s=m,n (isomers); s=p,q (levels); s=r (reonance); s=i,j (IAS); 
#                                     s=p,q,r,x can also indicate isomers (when more than two isomers are presented in a nuclide)
#     19: 31   Mass #     f13.6       Mass Excess in keV (# from systematics)
#     32: 42   dMass #    f11.6       Mass Excess uncertainty in keV (# from systematics)
#     43: 54   Exc #      f12.6       Isomer Excitation Energy in keV (# from systematics)
#     55: 65   dE #       f11.6       Isomer Excitation Energy uncertainty in keV (# from systematics)
#     66: 67   Orig          a2       Origin of Excitation Energy  
#     68: 68   Isom.Unc      a1       Isom.Unc = *  (gs and isomer ordering is uncertain) 
#     69: 69   Isom.Inv      a1       Isom.Inv = &  (the ordering of gs and isomer is reversed compared to ENSDF) 
#     70: 78   T #         f9.4       Half-life (# from systematics); stbl=stable; p-unst=particle unstable
#     79: 80   unit T        a2       Half-life unit 
#     82: 88   dT            a7       Half-life uncertainty 
#     89:102   Jpi */#/T=    a14      Spin and Parity (* directly measured; # from systematics; T=isospin) 
#    103:104   Ensdf year    a2       Ensdf update year 
#    115:118   Discovery     a4       Year of Discovery 
#    120:209   BR            a90      Decay Modes and their Intensities and Uncertanties in %; IS = Isotopic Abundance in %
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#001 0000   1n       8071.3181     0.0004                              609.8    s 0.6    1/2+*         06          1932 B-=100
#001 0010   1H       7288.971064   0.000013                            stbl              1/2+*         06          1920 IS=99.9855 78
#002 0010   2H      13135.722895   0.000015                            stbl              1+*           03          1932 IS=0.0145 78
#...

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
                    decay = DecayMode.ELECTRON_CAPTURE
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

# FROM REST API ##################################################################################

import requests
import os
import json
import io # Required for StringIO
import csv # Required for CSV parsing
from typing import Dict, Tuple, List, Optional # Added Optional
import collections

# --- Basic Z to Element Symbol Mapping ---
# (Add more elements as needed or use a library like mendeleev)
Z_TO_SYMBOL = {
    0: "n", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr",
    25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
    33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd",
    49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
    57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf",
    73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra",
    89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm",
    97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr",
    # Add elements 104+ if needed
}

# --- Helper Function to Safely Convert API String to Float ---
def safe_float(value_str: Optional[str]) -> Optional[float]:
    if value_str is None or value_str.strip() == "":
        return None
    try:
        # Handle potential "inf" or similar for stable isotopes if API uses it
        if value_str.strip().lower() == 'infinity' or value_str.strip().lower() == 'inf':
             return float('inf')
        return float(value_str)
    except ValueError:
        print(f"[Warning] Could not convert '{value_str}' to float.")
        return None

# --- Decay Mode String Mapping ---
# Map strings returned by API's decay_X fields to your DecayMode enum
API_DECAY_MAP = {
    "B-": DecayMode.BETA_MINUS,
    "EC": DecayMode.ELECTRON_CAPTURE,
    "B+": DecayMode.BETA_PLUS,
    "EC+B+": DecayMode.BETA_PLUS, # Often EC and B+ are combined in evaluations
    "A": DecayMode.ALPHA,
    "IT": DecayMode.ISOMERIC_TRANSITION,
    "SF": DecayMode.SPONTANEOUS_FISSION,
    "P": DecayMode.PROTON,
    "N": DecayMode.NEUTRON,
    # Add mappings for 2B-, B-N, B-2N, EC P, EC A, 2P, 2N, Cluster, etc.
    # based on actual strings observed in the API output for decay_1, decay_2, decay_3
    "STABLE": DecayMode.STABLE, # If the API uses 'STABLE'
    "": None, # Handle empty decay fields
}

# --- API Loader Class ---
API_URL = "https://nds.iaea.org/relnsd/v1/data" # Use v1 explicitly
CACHE_FILE = "nuclear_cache_iaea.json" # Use a distinct cache filename

class APILoader:
    def __init__(self):
        # Use instance cache, not global
        self.cache: Dict[Tuple[int, int], dict] = {}
        self.load_cache()

    def _get_element_symbol(self, Z: int) -> Optional[str]:
        return Z_TO_SYMBOL.get(Z)

    def _get_nuclide_str(self, Z: int, N: int) -> Optional[str]:
        symbol = self._get_element_symbol(Z)
        if symbol is None:
            print(f"[Error] No element symbol found for Z={Z}")
            return None
        A = Z + N
        # Ensure neutron is handled correctly if Z=0
        return f"{A}{symbol}" if Z > 0 else f"{A}n"

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    # Convert string keys "Z-N" back to tuple keys (Z, N)
                    raw_cache = json.load(f)
                    self.cache = {
                        (int(k.split('-')[0]), int(k.split('-')[1])): v
                        for k, v in raw_cache.items()
                        if '-' in k # Basic check for valid key format
                    }
                print(f"Loaded {len(self.cache)} entries from cache {CACHE_FILE}")
            except (json.JSONDecodeError, ValueError, IndexError, KeyError) as e:
                 print(f"[Error] Failed to load or parse cache file {CACHE_FILE}: {e}. Starting with empty cache.")
                 self.cache = {} # Start fresh if cache is corrupt
        else:
            print("Cache file not found, will query API.")

    def save_cache(self):
        try:
            with open(CACHE_FILE, 'w') as f:
                 # Convert tuple keys (Z, N) to string keys "Z-N" for JSON
                 serializable_cache = {
                     f"{k[0]}-{k[1]}": v
                     for k, v in self.cache.items()
                 }
                 json.dump(serializable_cache, f, indent=2) # Add indent for readability
            # print(f"Saved {len(self.cache)} entries to cache {CACHE_FILE}") # Reduce verbosity
        except IOError as e:
             print(f"[Error] Failed to save cache to {CACHE_FILE}: {e}")


    def retrieve_species_data(self, Z: int, N: int) -> Optional[dict]:
        key = (Z, N)
        if key in self.cache:
            # print(f"[CACHE HIT] Found cached data for Z={Z}, N={N}")
            return self.cache[key]

        nuclide_str = self._get_nuclide_str(Z, N)
        if not nuclide_str:
            return None # Cannot form request

        params = {"fields": "ground_states", "nuclides": nuclide_str}
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0'
            # Use a reasonable user agent string
        }

        print(f"[API CALL] Requesting ground_state data for {nuclide_str} (Z={Z}, N={N})")

        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=30) # Add timeout

            # Check for API specific error codes (returned as plain text)
            if not response.ok:
                if response.status_code == 403:
                     print(f"[API ERROR 403] Forbidden for {nuclide_str}. Check User-Agent or potential IP block.")
                     return None
                # Try reading potential numeric error code
                try:
                    error_code = int(response.text.strip())
                    print(f"[API ERROR {response.status_code}] Received API error code: {error_code} for {nuclide_str}")
                except ValueError:
                    print(f"[API ERROR {response.status_code}] Received non-numeric error message for {nuclide_str}: {response.text[:200]}")
                return None # General HTTP error

            # Check if response content looks like CSV (heuristic)
            content = response.text
            if not content or "z,n,symbol" not in content.splitlines()[0]: # Check header
                 # Check for numeric error code 0 (valid request, no data)
                try:
                    if int(content.strip()) == 0:
                         print(f"[API Info] No ground_state data found for {nuclide_str} (API code 0).")
                         # Cache the fact that there's no data? Or just return None. Let's return None.
                         return None
                    else:
                         print(f"[API Warning] Unexpected empty or non-CSV response for {nuclide_str}: {content[:200]}")
                         return None
                except ValueError:
                     print(f"[API Warning] Unexpected non-CSV response for {nuclide_str}: {content[:200]}")
                     return None

            # Parse CSV
            csv_file = io.StringIO(content)
            reader = csv.DictReader(csv_file)
            data_list = list(reader)

            if not data_list:
                print(f"[API Info] No data rows returned in CSV for {nuclide_str}.")
                # Maybe cache this "no data" result? For now, return None.
                return None

            # For ground_states and specific nuclide, expect only one row
            if len(data_list) > 1:
                 print(f"[API Warning] Expected 1 row for ground_state {nuclide_str}, got {len(data_list)}. Using first row.")

            species_data = data_list[0]
            self.cache[key] = species_data # Cache the dictionary
            self.save_cache() # Save after successful fetch
            return species_data

        except requests.Timeout:
            print(f"[ERROR] API request timed out for {nuclide_str}.")
            return None
        except requests.RequestException as e:
            print(f"[ERROR] Failed API request for {nuclide_str}: {e}")
            return None
        except Exception as e:
            # Catch unexpected errors during processing
            print(f"[ERROR] Unexpected error processing data for {nuclide_str}: {e}")
            return None

    def parse_species(self, data: dict, Z: int, N: int) -> Optional[NuclearSpecies]:
        """ Parses the dictionary data (one row from ground_states CSV) into a NuclearSpecies object. """
        if not data:
             print(f"[Parse Error] No data provided for Z={Z}, N={N}")
             return None

        try:
            species = NuclearSpecies(Z, N)

            # Basic properties
            species.symbol = data.get("symbol", "").strip() or None
            species.spin_parity = data.get("jp", "").strip() or None

            # Mass excess
            species.mass_excess_keV = safe_float(data.get("mass_excess"))
            species.mass_excess_unc_keV = safe_float(data.get("unc_me")) # Assuming 'unc_me' is the correct field

            # Half-life
            species.half_life_s = safe_float(data.get("half_life_sec"))
            species.half_life_unc_s = safe_float(data.get("unc_hls"))
            hl_val = data.get("half_life", "").strip()
            hl_unit = data.get("unit_hl", "").strip()
            species.half_life_text = f"{hl_val} {hl_unit}".strip() or None

            # Check for stability based on half-life
            is_stable = species.half_life_s == float('inf') or species.half_life_text.lower() == 'stable'
            if is_stable:
                 species.add_decay_mode(DecayMode.STABLE, 100.0)
                 # API might still list decay modes even if stable (e.g. theoretical SF)
                 # Let's parse them anyway but ensure STABLE is primary if HL suggests it.

            # Decay Modes (from decay_1, decay_2, decay_3)
            for i in [1, 2, 3]:
                mode_str = data.get(f"decay_{i}", "").strip()
                intensity_str = data.get(f"decay_{i}_%", "").strip()

                if not mode_str: # Stop if no more decay modes listed
                    break

                decay_mode = API_DECAY_MAP.get(mode_str, DecayMode.UNKNOWN)
                intensity = safe_float(intensity_str)

                if decay_mode == DecayMode.UNKNOWN:
                     print(f"[Warning] Unknown decay mode string '{mode_str}' for Z={Z}, N={N}. Mapping to UNKNOWN.")
                
                if decay_mode:
                    # Avoid adding STABLE again if already added based on half-life
                    if not (is_stable and decay_mode == DecayMode.STABLE):
                         species.add_decay_mode(decay_mode, intensity)

            # Add other fields if needed (Q-values, binding energy, etc.)
            # species.q_beta_minus = safe_float(data.get("qbm"))
            # species.q_alpha = safe_float(data.get("qa"))
            # ... etc

            return species

        except KeyError as e:
             print(f"[Parse Error] Missing expected key {e} in data for Z={Z}, N={N}. Data: {data}")
             return None
        except Exception as e:
             print(f"[Parse Error] Unexpected error parsing data for Z={Z}, N={N}: {e}")
             return None

    def get_species_list(self, nz_list: List[Tuple[int, int]]) -> Dict[Tuple[int, int], NuclearSpecies]:
        """ Fetches and parses data for a list of (Z, N) pairs. """
        species_dict = {}
        total = len(nz_list)
        for i, (Z, N) in enumerate(nz_list):
            print(f"Processing {i+1}/{total}: Z={Z}, N={N}")
            data = self.retrieve_species_data(Z, N) # Fetch raw dict data
            if not data:
                print(f"  -> No data retrieved.")
                continue # Skip if retrieval failed or no data found

            species = self.parse_species(data, Z, N) # Parse the dict
            if species:
                species_dict[(Z, N)] = species
                # print(f"  -> Success: {species.symbol}") # Optional success message
            else:
                 print(f"  -> Failed to parse data.")

        print(f"Finished processing. Successfully obtained data for {len(species_dict)} out of {total} requested nuclides.")
        return species_dict

    def get_all_species(self) -> Dict[Tuple[int, int], NuclearSpecies]:
        species_dict: Dict[Tuple[int, int], NuclearSpecies] = {}
        queried: Set[Tuple[int, int]] = set() # Track nuclides processed or in queue
        queue: Deque[Tuple[int, int]] = collections.deque()
        # Initialize
        start_nuclide = (1, 0)
        queue.append(start_nuclide)
        while queue:
            Z, N = queue.popleft()
            # Check cache first (retrieve_species_data handles this)
            data = self.retrieve_species_data(Z, N) # Fetches raw dict data (or None)
            if data:
                # Data found, parse it
                species = self.parse_species(data, Z, N)
                if species:
                    species_dict[(Z, N)] = species
                    # Explore valid neighbors only if data was found for current nuclide
                    neighbors = [
                        (Z + 1, N),
                        (Z, N + 1),
                    ]
                    for next_Z, next_N in neighbors:
                        # Basic validity checks
                        if next_Z < 0 or next_N < 0 or (next_Z == 0 and next_N == 0):
                            continue
                        neighbor_key = (next_Z, next_N)
                        if neighbor_key not in queried:
                             queried.add(neighbor_key)
                             queue.append(neighbor_key)
                # else: parsing failed, don't explore neighbors from this node
            # else: retrieve_species_data returned None (no data/API error)
            #      -> Do not add neighbors, effectively pruning this branch
        print(f"Successfully obtained data for {len(species_dict)} nuclides.")
        return species_dict

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
    # Pygame initialization should ideally be inside the Viewer's __init__,
    # but keeping it here as requested to minimize changes to this block.
    pygame.init()
    # Setting display mode here might be okay if Viewer doesn't also do it.
    # If Viewer.__init__ handles display setup, this line should be removed.
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    # --- Data Loading --- from NUBASE
    # Consider adding file existence checks here as done previously
#     nubase_filepath = "nubase_4.mas20.txt" # Default or use sys.argv
#     # (Optional: Add file check logic from previous examples if desired)
#     print(f"Loading species from: {nubase_filepath}")
#     species = parse_nubase_species(nubase_filepath)
#     if not species:
#          print("ERROR: No species loaded. Exiting.")
#          sys.exit(1) # Exit if loading failed
#     print(f"Loaded {len(species)} species.")

    # --- Data Loading --- from API
    loader = APILoader()
    print("Starting data retrieval (may take a long time)...")
    species = loader.get_all_species() # This performs caching/API calls
    print(f"Finished. Retrieved data for {len(species)} nuclides.")

    # --- Setup Viewer and Adapter ---
    adapter = RendererAdapter(species)
    render_data = adapter.get_render_data()
    # Assuming Viewer.__init__ does NOT re-initialize pygame or screen if done above
    viewer = Viewer(render_data, adapter)

    # --- Threading ---
    # 3. Start render thread as NON-DAEMON so join() will wait for it
    render_thread = threading.Thread(target=viewer.render, daemon=False, name="RenderThread")
    render_thread.start()

    # --- Run Input Loop (Main Thread) ---
    # Assuming viewer.handle_input() now blocks until viewer._running is False
    # and that viewer._running is set to False on QUIT event.
    print("Starting input handler...")
    try:
        viewer.handle_input()
    except KeyboardInterrupt: # Handle Ctrl+C gracefully
        print("KeyboardInterrupt received, signaling viewer to stop.")
        # Assuming Viewer has a method to signal stop, e.g., viewer.stop()
        # If not, the handle_input setting _running=False on QUIT is sufficient
        # viewer.stop() # Call this if viewer has such a method
        pass # Allow loop in handle_input to terminate naturally

    # --- Shutdown ---
    # 3. Wait for the render thread to finish *after* the input loop has ended
    print("Main thread waiting for render thread to join...")
    render_thread.join()
    print("Render thread joined.")

    # 4. Quit Pygame *after* the render thread has finished cleanly
    print("Quitting Pygame.")
    pygame.quit()

    print("Program finished.")
    # No sys.exit() needed here, program exits when main thread finishes.