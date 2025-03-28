import os
import requests
import pandas as pd
from io import StringIO
import plotly.io as pio
import plotly.graph_objects as go

# Native resolution (logical layout)
NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720

# Target resolutions
EXPORTS = {
    "fossil_plot_1920.png": (1920, 1080),
    "fossil_plot_4k.png": (3840, 2160),
}

# Taxa and local cache files
taxa = {
    "Orthocerida": "orthocerida.csv",
    "Goniatitida": "goniatitida.csv",
    "Ceratitida": "ceratitida.csv",
    "Ammonitida": "ammonitida.csv"
}

# Function to download or load from cache
def fetch_data(taxon, filename):
    if os.path.exists(filename):
        print(f"[Cache] Using existing file: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read()
    else:
        print(f"[Download] Fetching data for {taxon}...")
        url = f"https://paleobiodb.org/data1.2/occs/list.csv?base_name={taxon}&show=time&limit=all"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Request failed for {taxon} with status {response.status_code}")
        data = response.text
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)
    return pd.read_csv(StringIO(data))

# Bin configuration
bin_size = 5
bins = range(0, 550, bin_size)
dfs = {}
for taxon, file in taxa.items():
    df = fetch_data(taxon, file)
    df["mid_ma"] = (df["max_ma"] + df["min_ma"]) / 2
    df = df[df["mid_ma"].notnull()]
    df["bin"] = pd.cut(df["mid_ma"], bins=bins, labels=bins[:-1])
    count_series = df["bin"].value_counts().sort_index()
    dfs[taxon] = count_series

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

# Build the figure once at native resolution
fig = go.Figure()

for taxon, counts in dfs.items():
    fig.add_trace(go.Scatter(
        x=[int(b) for b in counts.index],
        y=counts.values,
        mode='lines+markers',
        name=taxon
    ))

period_colors = ["#e0f7fa", "#b2ebf2"]
color_idx = 0
for name, start, end, level in chronostrat:
    if level == "period" and name != "Quaternary":
        fig.add_shape(
            type="rect",
            x0=end, x1=start, y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=period_colors[color_idx % 2],
            layer="below",
            line_width=0,
            opacity=0.4
        )
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
for name, start, end, level in chronostrat:
    if level == "era":
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

fig.update_layout(
    title="Fossil Occurrence Distributions with Geological Timescale",
    xaxis_title="Age (Ma)",
    yaxis_title="Number of Occurrences",
    xaxis=dict(autorange="reversed"),
    yaxis=dict(range=[-0.15, None]),
    template="plotly_white",
    margin=dict(t=60, b=60)
)

# HTML preview
fig.write_html("fossil_plot.html", auto_open=False)
print("HTML saved to fossil_plot.html")

# PNG exports (scaling from native resolution)
for filename, (target_width, target_height) in EXPORTS.items():
    scale = target_width / NATIVE_WIDTH
    fig.write_image(filename, width=NATIVE_WIDTH, height=NATIVE_HEIGHT, scale=scale)
    print(f"Saved {filename} at scale {scale:.2f}")
