from models.utils.feature_engineering import load_data
from models.utils.visualization import Plotter
import matplotlib.pyplot as plt

def plot():
    data = load_data("../data")
    corr_plot = Plotter.corr_plot(data)
    plt.savefig("../images/corr.png", dpi=150)

if __name__ == "__main__":
    plot()