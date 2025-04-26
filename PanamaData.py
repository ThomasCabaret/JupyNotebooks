import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fossil-calibrated divergence data from O'Dea et al. (2016) Supplementary Table S3
data = [
    {'Pair': 'Chaetodon humeralis vs. C. ocellatus', 'Median': 3.4, 'Low95': 1.8, 'High95': 5.4},
    {'Pair': 'Mycteroperca jordani vs. M. bonaci/venenosa', 'Median': 3.58, 'Low95': 1.9, 'High95': 5.51},
    {'Pair': 'Scarus hoefleri vs. S. perrico', 'Median': 4.82, 'Low95': 2.46, 'High95': 7.7},
    {'Pair': 'Balistes capriscus vs. B. polylepis', 'Median': 4.97, 'Low95': 3.14, 'High95': 7.13},
    {'Pair': 'Chromis alta vs. C. enchrysura', 'Median': 5.57, 'Low95': 3.03, 'High95': 8.81},
    {'Pair': 'Stegastes rectifraenum vs. S. imbricatus', 'Median': 6.19, 'Low95': 3.17, 'High95': 9.63},
    {'Pair': 'Sargocentron suborbitalis vs. S. vexillarium', 'Median': 6.2, 'Low95': 3.9, 'High95': 10.8},
    {'Pair': 'Microspathodon chrysurus vs. M. dorsalis', 'Median': 6.56, 'Low95': 2.75, 'High95': 11.53},
    {'Pair': 'Calamus brachysomus vs. C. nodosus', 'Median': 8.0, 'Low95': 2.0, 'High95': 16.0},
    {'Pair': 'Plectrypops lima vs. P. retrospinis', 'Median': 8.1, 'Low95': 4.8, 'High95': 11.6},
    {'Pair': 'Pomacanthus zonipectus vs. P. paru/arcuatus', 'Median': 8.6, 'Low95': 5.0, 'High95': 12.3},
    {'Pair': 'Stegastes flavilatus vs. complex', 'Median': 9.61, 'Low95': 6.56, 'High95': 13.18},
    {'Pair': 'Littoraria irrorata vs. L. variegata', 'Median': 8.5, 'Low95': 4.5, 'High95': 12.5},
    {'Pair': 'Arcopsis adamsi vs. A. solida', 'Median': 8.87, 'Low95': 3.78, 'High95': 15.36},
    {'Pair': 'Conus perplexus vs. C. puncticulatus', 'Median': 8.9, 'Low95': 5.0, 'High95': 19.0},
    {'Pair': 'Conus regius vs. complex', 'Median': 9.2, 'Low95': 5.0, 'High95': 14.9},
    {'Pair': 'Barbatia candida vs. B. reeveana', 'Median': 9.54, 'Low95': 3.39, 'High95': 16.62},
    {'Pair': '(Echinolittorina apicina, E. paytensis) vs. E. risei', 'Median': 10.36, 'Low95': 6.83, 'High95': 13.96},
    {'Pair': '(Echinolittorina aspera...) vs. E. interrupta', 'Median': 10.75, 'Low95': 7.39, 'High95': 14.35},
    {'Pair': '(Echinolittorina modesta...) vs. E. ziczac', 'Median': 10.87, 'Low95': 7.52, 'High95': 14.42},
    {'Pair': 'Barbatia illota vs. B. tenera', 'Median': 11.23, 'Low95': 5.09, 'High95': 18.87},
    {'Pair': 'Trachycardium egmontianum vs. Phlogocardium belcheri', 'Median': 12.0, 'Low95': 5.5, 'High95': 20.0},
    {'Pair': 'Laevicardium pictum vs. L. elenense', 'Median': 12.5, 'Low95': 5.0, 'High95': 22.5},
    {'Pair': 'Papyridea aspersa vs. P. semisulcata', 'Median': 16.0, 'Low95': 8.0, 'High95': 24.5},
    {'Pair': '(Bulla gouldiana...) vs. B. mabillei', 'Median': 16.8, 'Low95': 7.2, 'High95': 31.7},
    {'Pair': 'Nerita scabricosta vs. complex', 'Median': 19.29, 'Low95': 16.0, 'High95': 24.24},
    {'Pair': 'Anadara chemnitzii vs. A. nux', 'Median': 20.95, 'Low95': 10.69, 'High95': 32.21},
    {'Pair': 'Amercardia media vs. complex', 'Median': 21.0, 'Low95': 9.0, 'High95': 30.0},
    {'Pair': 'Echinolittorina galapagensis vs. complex', 'Median': 21.31, 'Low95': 16.39, 'High95': 26.09},
    {'Pair': 'Nerita funiculata vs. complex', 'Median': 21.8, 'Low95': 17.13, 'High95': 28.16},
    {'Pair': 'Littoraria rosewateri vs. L. tessellata', 'Median': 24.2, 'Low95': 15.0, 'High95': 32.5},
    {'Pair': 'Littoraria nebulosa vs. complex', 'Median': 25.7, 'Low95': 18.0, 'High95': 33.0},
    {'Pair': 'Arca mutabilis vs. A. imbricata', 'Median': 27.5, 'Low95': 12.38, 'High95': 44.42},
    {'Pair': 'Mellita quinquiesperforata vs. complex', 'Median': 3.21, 'Low95': 2.51, 'High95': 3.91},
    {'Pair': '(Mellita longifissa...) vs. complex', 'Median': 5.46, 'Low95': 4.4, 'High95': 6.5},
    {'Pair': 'Conus gladiator vs. C. mus', 'Median': 4.10, 'Low95': 2.90, 'High95': 5.80},
    {'Pair': 'Conus ermineus vs. C. purpurascens', 'Median': 4.20, 'Low95': 1.50, 'High95': 7.50},
    {'Pair': 'Dallocardia senticosum vs. Acrosterigma magnum', 'Median': 6.00, 'Low95': 2.00, 'High95': 13.00}
]

def compute_density(data, t_min=0, t_max=50, num_points=1000):
    time_grid = np.linspace(t_min, t_max, num_points)
    density_sum = np.zeros_like(time_grid)
    
    for entry in data:
        median = entry["Median"]
        low95 = entry["Low95"]
        high95 = entry["High95"]
        std = (high95 - low95) / (2 * 1.96)
        if std > 0:
            density_sum += norm.pdf(time_grid, loc=median, scale=std)
    return time_grid, density_sum

# Compute the density
time, density = compute_density(data)

# Define common plot function
def make_plot(with_density=True, filename="output.png"):
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    
    if with_density:
        ax.plot(time, density, color='navy', linewidth=2.5, label="Sum of probability densities")
    
    for entry in data:
        ax.axvline(x=entry["Median"], color='#4E97D1', linestyle='--', alpha=0.8, linewidth=2)

    ax.set_xlim(0, 40)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("Time before present (Ma)")
    ax.set_ylabel("Relative divergence density")
    ax.set_title("Estimated divergence density over time (fossil-calibrated pairs)")
    ax.grid(True)
    
    if with_density:
        ax.legend()
    
    plt.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)

# Create and save the two figures
make_plot(with_density=True, filename="divergence_density_with_curve.png")
make_plot(with_density=False, filename="divergence_density_without_curve.png")
