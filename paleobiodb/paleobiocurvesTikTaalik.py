import os
import warnings
import requests
import pandas as pd
from io import StringIO
import plotly.graph_objects as go

# Suppress pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Native resolution (logical layout)
NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720

# Target resolutions
EXPORTS = {
    "hd": (1920, 1080),
    "4k": (3840, 2160),
}

# Taxa sets and local cache filenames
TAXA_SETS = {
    "tetrapod": [
        ("Tetrapoda", "tetrapoda.csv")
    ],
    "pre_tiktaalik": [
        ("Dipnoi", "dipnoi.csv"),
        ("Actinistia", "actinistia.csv"),
        ("Osteolepiformes", "osteolepiformes.csv")
    ]
}

# Tiktaalik interval in Ma
TIKTAALIK_MIN = 373
TIKTAALIK_MAX = 377

# Chronostratigraphic scale
chronostrat = [
    ("Cambrian", 541, 485, "period"),
    ("Ordovician", 485, 444, "period"),
    ("Silurian", 444, 419, "period"),
    ("Devonian", 419, 359, "period"),
    ("Carboniferous", 359, 299, "period"),
    ("Permian", 299, 252, "period"),
    ("Paleozoic", 541, 252, "era"),
    ("Triassic", 252, 201, "period"),
    ("Jurassic", 201, 145, "period"),
    ("Cretaceous", 145, 66, "period"),
    ("Mesozoic", 252, 66, "era"),
    ("Paleogene", 66, 23, "period"),
    ("Neogene", 23, 2.6, "period"),
    ("Quaternary", 2.6, 0, "period"),
    ("Cenozoic", 66, 0, "era")
]

# Function to download or load from cache, or refresh if required columns are missing
def fetch_data(taxon, filename):
    def load_and_check(fpath):
        df = pd.read_csv(fpath)
        required_cols = {"max_ma", "min_ma"}
        return df, required_cols.issubset(df.columns)

    if os.path.exists(filename):
        print(f"[Cache] Using existing file: {filename}")
        df, valid = load_and_check(filename)
        if valid:
            return df
        else:
            print(f"[Update] Cached file missing required columns, refreshing: {filename}")

    print(f"[Download] Fetching data for {taxon}...")
    url = f"https://paleobiodb.org/data1.2/occs/list.csv?base_name={taxon}&show=phylo,time&limit=all"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed for {taxon} with status {response.status_code}")
    data = response.text
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)
    return pd.read_csv(StringIO(data))

# Bin configuration
def compute_bins(min_ma, max_ma, bin_size=5):
    start = int((max_ma + bin_size) // bin_size * bin_size)
    end = int(min_ma // bin_size * bin_size)
    return list(range(end, start + bin_size, bin_size))

# Plotting function
def generate_plot(label, resolved_level, df, bins):
    min_ma_data = df["mid_ma"].min()
    max_ma_data = df["mid_ma"].max()
    grouped = df.groupby([resolved_level, "bin"], observed=True).size().unstack(fill_value=0).sort_index()

    fig = go.Figure()
    
    taxa_grouped = df.groupby([resolved_level, "bin"], observed=True).size()
    
    for taxon in sorted(df[resolved_level].unique()):
        taxon_data = taxa_grouped.loc[taxon] if taxon in taxa_grouped.index else pd.Series()
        if taxon_data.empty:
            continue
        taxon_data = taxon_data[taxon_data > 0]
        fig.add_trace(go.Scatter(
            x=[int(b) for b in taxon_data.index],
            y=taxon_data.values,
            mode='lines+markers',
            name=str(taxon)
        ))

    period_colors = ["#e0f7fa", "#b2ebf2"]
    color_idx = 0
    for name, start, end, level_type in chronostrat:
        if level_type == "period" and start >= min_ma_data and end <= max_ma_data:
            fig.add_shape(
                type="rect",
                x0=end, x1=start, y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor=period_colors[color_idx % 2],
                layer="below",
                line_width=0,
                opacity=0.4
            )
            if start - end > 5:
                fig.add_annotation(
                    x=(start + end) / 2,
                    y=1.02,
                    text=name,
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    font=dict(size=10)
                )
            color_idx += 1

    era_colors = ["#ffe0b2", "#ffcc80", "#ffb74d"]
    era_color_idx = 0
    for name, start, end, level_type in chronostrat:
        if level_type == "era" and start >= min_ma_data and end <= max_ma_data:
            fig.add_shape(
                type="rect",
                x0=end, x1=start, y0=-0.1, y1=0,
                xref="x", yref="paper",
                fillcolor=era_colors[era_color_idx % len(era_colors)],
                layer="below",
                line_width=0,
                opacity=0.8
            )
            fig.add_annotation(
                x=(start + end) / 2,
                y=-0.08,
                text=f"<b>{name}</b>",
                showarrow=False,
                xref="x",
                yref="paper",
                font=dict(size=12)
            )
            era_color_idx += 1

    # Highlight Tiktaalik's fossil window
    fig.add_shape(
        type="rect",
        x0=TIKTAALIK_MIN,
        x1=TIKTAALIK_MAX,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="red",
        opacity=0.2,
        layer="below",
        line_width=0
    )
    fig.add_annotation(
        x=(TIKTAALIK_MIN + TIKTAALIK_MAX) / 2,
        y=1.05,
        text="Tiktaalik window",
        showarrow=False,
        xref="x",
        yref="paper",
        font=dict(size=11)
    )

    fig.update_layout(
        title="Fossil Occurrence Distributions with Geological Timescale",
        xaxis_title="Age (Ma)",
        yaxis_title="Number of Occurrences",
        xaxis=dict(
            autorange=False,
            range=[450, 300],
            tickmode="linear",
            tick0=500,
            dtick=20
        ),
        yaxis=dict(range=[-0.15, None]),
        template="plotly_white",
        margin=dict(t=60, b=60, r=100),
        legend=dict(
            orientation="v",
            x=1.00,
            xanchor="left",
            y=1,
            font=dict(size=10),
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.7)"
        )
    )

    html_file = f"fossil_plot_{label}_{resolved_level}.html"
    fig.write_html(html_file, auto_open=False)
    print(f"[Output] HTML saved to {html_file}")

    for suffix, (w, h) in EXPORTS.items():
        scale = w / NATIVE_WIDTH
        filename = f"fossil_plot_{label}_{resolved_level}_{suffix}.png"
        fig.write_image(filename, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, scale=scale)
        print(f"[Output] Saved {filename} at scale {scale:.2f}")

    print(f"\n[Bin counts for: {label}]")
    for taxon in sorted(df[resolved_level].unique()):
        taxon_data = taxa_grouped.loc[taxon] if taxon in taxa_grouped.index else pd.Series()
        if taxon_data.empty:
            continue
        taxon_data = taxon_data[taxon_data > 0]
        print(f"\nTaxon: {taxon}")
        for bin_label, count in taxon_data.items():
            bin_start = int(bin_label)
            bin_end = bin_start + 5
            print(f"  {bin_start}-{bin_end} Ma : {count} occurrence(s)")

    return True

def process_custom_plot(label, taxa_pairs):
    print(f"\n--- Processing custom plot: {label} ---")
    all_data = []
    for taxon, file in taxa_pairs:
        df = fetch_data(taxon, file)
        df["source_taxon"] = taxon
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    df = df[df["occurrence_no"] != 694739]  #false positiv
    df = df[df["max_ma"].notnull() & df["min_ma"].notnull()]
    df["mid_ma"] = (df["max_ma"] + df["min_ma"]) / 2

    # Print Tetrapoda outliers above 400 Ma
    tetrapoda_outliers = df[(df["source_taxon"] == "Tetrapoda") & (df["mid_ma"] > 400)]
    if not tetrapoda_outliers.empty:
        print("\n[Warning] Tetrapoda occurrences > 400 Ma:")
        print(tetrapoda_outliers[[
            "occurrence_no", "collection_no", "max_ma", "min_ma", "mid_ma",
            "early_interval", "late_interval", "identified_name"
        ]])

    # Apply time window
    df = df[(df["mid_ma"] >= 300) & (df["mid_ma"] <= 450)]
    min_ma = df["mid_ma"].min()
    max_ma = df["mid_ma"].max()
    bins = compute_bins(min_ma, max_ma)
    df["bin"] = pd.cut(df["mid_ma"], bins=bins, labels=bins[:-1])
    df = df[df["source_taxon"].notnull()]
    generate_plot(label, "source_taxon", df, bins)

# Main loop
def process_taxa_set(label, taxa):
    print(f"\n--- Processing sequence: {label} ---")
    all_data = []
    for taxon, file in taxa:
        df = fetch_data(taxon, file)
        df["source_taxon"] = taxon

        if "order_name" not in df.columns or df["order_name"].isnull().all():
            df["order_name"] = taxon

        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[full_df["max_ma"].notnull() & full_df["min_ma"].notnull()]

    full_df["mid_ma"] = (full_df["max_ma"] + full_df["min_ma"]) / 2
    min_ma = full_df["mid_ma"].min()
    max_ma = full_df["mid_ma"].max()
    bins = compute_bins(min_ma, max_ma)

    df = full_df.copy()
    df = df[(df["mid_ma"] >= 300) & (df["mid_ma"] <= 450)]
    df["bin"] = pd.cut(df["mid_ma"], bins=bins, labels=bins[:-1])
    df = df[df["order_name"].notnull()]

    generate_plot(label, "source_taxon", df, bins)

# Execute processing
process_custom_plot("dipnoi_vs_tetrapoda", [
    ("Tetrapoda", "tetrapoda.csv"),
    ("Dipnoi", "dipnoi.csv")
])