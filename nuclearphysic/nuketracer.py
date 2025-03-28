import pandas as pd
import numpy as np
import plotly.express as px

def parse_mass20(filepath):
    columns = [
        ('cc', 0, 1),
        ('NZ', 1, 4),
        ('N', 4, 9),
        ('Z', 9, 14),
        ('A', 14, 19),
        ('el', 19, 23),
        ('o', 23, 27),
        ('mass_excess', 28, 42),
        ('mass_excess_unc', 42, 54),
        ('binding_per_A', 55, 68),
        ('binding_per_A_unc', 68, 79),
        ('decay_mode', 80, 82),
        ('beta_decay_energy', 83, 96),
        ('beta_decay_unc', 96, 107),
        ('unknown_field', 108, 111),
        ('atomic_mass', 112, 125),
        ('atomic_mass_unc', 125, 137),
    ]
    def parse_line(line):
        row = {}
        for name, start, end in columns:
            raw = line[start:end].strip()
            if name in ['el', 'o', 'decay_mode']:
                row[name] = raw
            else:
                try:
                    if '#' in raw or '*' in raw or raw == '':
                        row[name] = None
                    else:
                        row[name] = float(raw)
                except ValueError:
                    row[name] = None
        return row
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(('0', '1')) and len(line.strip()) > 50:
                try:
                    row = parse_line(line)
                    data.append(row)
                except Exception:
                    continue
    return pd.DataFrame(data)

def parse_nubase2020(filepath):
    columns = [
        ('A', 0, 3),
        ('Z_flag', 4, 8),
        ('A_el', 11, 16),
        ('state_id', 16, 17),
        ('mass_excess', 18, 31),
        ('mass_excess_unc', 31, 42),
        ('exc_energy', 42, 54),
        ('exc_energy_unc', 54, 65),
        ('exc_origin', 65, 67),
        ('isom_unc', 67, 68),
        ('isom_inv', 68, 69),
        ('half_life', 69, 78),
        ('half_life_unit', 78, 80),
        ('half_life_unc', 81, 88),
        ('J_pi', 88, 102),
        ('ensdf_year', 102, 104),
        ('discovery_year', 114, 118),
        ('decay_modes', 119, 209)
    ]
    def parse_line(line):
        row = {}
        for name, start, end in columns:
            raw = line[start:end].strip()
            if name in ['A_el', 'state_id', 'exc_origin', 'isom_unc', 'isom_inv', 'half_life_unit', 'J_pi', 'decay_modes']:
                row[name] = raw
            else:
                try:
                    if '#' in raw or '*' in raw or raw.lower() in ['stbl', 'p-unst'] or raw == '':
                        row[name] = None
                    else:
                        row[name] = float(raw)
                except ValueError:
                    row[name] = raw
        return row
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            try:
                row = parse_line(line)
                data.append(row)
            except Exception:
                continue
    return pd.DataFrame(data)

if __name__ == '__main__':
    df_mass = parse_mass20("mass_1.mas20.txt")
    df_nubase = parse_nubase2020("nubase_4.mas20.txt")
    df = df_nubase[df_nubase['A'].notna()]
    df['Z_str'] = df['Z_flag'].astype(str).str[:3]
    df = df[df['Z_str'].str.match(r'^\d{1,3}$')]
    df['Z'] = df['Z_str'].astype(int)
    df['A'] = df['A'].astype(int)
    df['N'] = df['A'] - df['Z']
    df['half_life'] = pd.to_numeric(df['half_life'], errors='coerce')
    df = df[df['half_life'] > 0]
    df['log_half_life'] = np.log10(df['half_life'])
    fig = px.scatter(
        df,
        x='Z',
        y='N',
        color='log_half_life',
        hover_data=['A_el', 'half_life', 'half_life_unit', 'decay_modes'],
        title='N-Z Chart Colored by log10(Half-life)',
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(symbol='square', size=8))
    fig.write_html("chart_half_life.html")
    fig.write_image("chart_half_life.png", engine="kaleido")
