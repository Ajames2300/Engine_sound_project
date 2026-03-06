import os 
import sys
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


if __name__ == "__main__":

    from data.data_loader import EngineDataSet
    from features.feature import FreqFeatureExtractor, StaticFeatureExtractor

    # ---------------------------------------------------- #
    data = EngineDataSet('data/raw/train_cut/')
    data_good = [item for item in data.sample if item[1] == data.class_idx['engine1_good']]
    data_broken = [item for item in data.sample if item[1] == data.class_idx['engine2_broken']]
    data_heavy = [item for item in data.sample if item[1] == data.class_idx['engine3_heavyload']]

    #  Construct static datasets 
    static_features_good = StaticFeatureExtractor(data_good)
    static_features_broken = StaticFeatureExtractor(data_broken)
    static_features_heavy = StaticFeatureExtractor(data_heavy)

    #  features of good engine   
    entire_good = static_features_good.make_static_features()
    entire_broken = static_features_broken.make_static_features()
    entire_heavy = static_features_heavy.make_static_features()

    # ==== Frequency domain features ==== #
    freq_feature_good = FreqFeatureExtractor(data_good)
    freq_feature_broken = FreqFeatureExtractor(data_broken) 
    freq_feature_heavy = FreqFeatureExtractor(data_heavy)

    # PSD calculation for each class
    psd_good = freq_feature_good.psd(nfft = 2048, window = "hann", scaling = "density")
    psd_broken = freq_feature_broken.psd(nfft = 2048, window = "hann", scaling = "density") 
    psd_heavy = freq_feature_heavy.psd(nfft = 2048, window = "hann", scaling = "density")   
    
    # === Subplot layout === # 
    fig = sp.make_subplots(
                            rows=2,
                            cols=2,
                            subplot_titles=(
                                "Power Spectrum",
                                "STD Distribution",
                                "RMS Distribution",
                                "Peak Distribution"
                            ),
                            specs=[
                                [{}, {}],
                                [{}, {}]
                            ]
                        )
                                        
    # ===== PSD ===== # 
    fig.add_trace(
        go.Scatter(
            x=psd_good[0][0],
            y=psd_good[0][1],
            mode="lines",
            name="PSD"
        ),
        row=1,
        col=1
    )

    # ===== Mean ===== #
    fig.add_trace(
        go.Histogram(
            x=[feature["mean"] for feature in entire_good],
            nbinsx=20,
            name="Mean"
        ),
        row=1,
        col=2
    )

    # ===== STD ===== #
    fig.add_trace(
        go.Histogram(
            x=[feature["std"] for feature in entire_good],
            nbinsx=20,
            name="STD"
        ),
        row=2,
        col=1
    )

    # ===== RMS ===== #
    fig.add_trace(
        go.Histogram(
            x=[feature["rms"] for feature in entire_good],
            nbinsx=20,
            name="RMS"
        ),
        row=2,
        col=2
    )

    # ===== layout style ===== #
    fig.update_layout(
        height=800,
        width=1000,
        template="plotly_white",
        title_font_size=24,
        title_x=0.5,  # Center the title 
        title="Engine Signal Feature Analysis Dashboard",
        showlegend=False
    )

    fig.show()

    