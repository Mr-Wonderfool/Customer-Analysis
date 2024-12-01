import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Plotter():
    _PALETTE_ = sns.color_palette("muted")
    def __init__(self, ):
        pass
    @classmethod
    def pair_plot(cls, data, to_plot, hue):
        return sns.pairplot(data[to_plot], hue=hue, palette=cls._PALETTE_[:len(data[hue].value_counts())])
    @classmethod
    def corr_plot(cls, data: pd.DataFrame):
        numeric_columns = data.select_dtypes(include=['int','float']).columns
        corr = data[numeric_columns].corr()
        plt.figure(figsize=(20, 20))
        return sns.heatmap(corr, cmap="Reds")