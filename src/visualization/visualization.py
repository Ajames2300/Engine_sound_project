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
    EngineData = EngineDataSet('data/raw/train_cut/')
    data_good = [EngineData.sample[i] for i in range(len(EngineData.sample)) if EngineData.label[i] == EngineData.class_idx['engine1_good']]
    data_broken = [EngineData.sample[i] for i in range(len(EngineData.sample)) if EngineData.label[i] == EngineData.class_idx['engine2_broken']]
    data_heavy = [EngineData.sample[i] for i in range(len(EngineData.sample)) if EngineData.label[i] == EngineData.class_idx['engine3_heavyload']]

    #  Construct with different classes object
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
    # psd_good = freq_feature_good.psd(nfft = 2048, window = "hann", scaling = "density")
    # psd_broken = freq_feature_broken.psd(nfft = 2048, window = "hann", scaling = "density") 
    # psd_heavy = freq_feature_heavy.psd(nfft = 2048, window = "hann", scaling = "density")   
    
    # TODO: 3 Loops 
    static_group_names = ["entire_good", "entire_broken", "entire_heavy"]
    freq_group_names = ["freq_feature_good", "freq_feature_broken", "freq_feature_heavy"]

    for groupf in range(len(EngineData.class_idx)):

        # Corresponding the group name
        psd_ = eval(freq_group_names[groupf])
        psd_feature_f, psd_feature_pxx = psd_.psd(sample_rate= EngineData.sample_rate[0], 
                                                  nfft = 2048, 
                                                  window = "hann", 
                                                  scaling = "density")
        static_feature = eval(static_group_names[groupf])
    
        # 2. 建立初始畫布 
        fig1 = go.Figure()
                        
        # ===== PSD with ranger slider ===== # 
        fig1.add_trace(
            go.Scatter(
                x=psd_feature_f[0],
                y=psd_feature_pxx[0],
                mode="lines",
                name="PSD"
            )
        )

        frames = []
        for i in range(len(psd_feature_f)):
            frames.append(go.Frame(
                data = [
                    go.Scatter(
                        x=psd_feature_f[i],
                        y=psd_feature_pxx[i],
                        name=f"PSD - Sample {i+1}"
                    )
                ],
                name=f"frame{i+1}"
            ))    
        fig1.frames = frames

        # Update axis titles for subplot[0,0]
        fig1.update_xaxes(title_text="Frequency (Hz)")
        fig1.update_yaxes(title_text="Power Spectral Density")

        fig1.update_layout(
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Sample Index: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "args": [[f"frame{i+1}"], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                        "label": f"{i+1}",
                        "method": "animate"
                    } for i in range(len(psd_feature_f))
                ]
            }]
        )

        # Combine fig1 and fig2 side by side
        fig_combined = sp.make_subplots(
            rows=2, cols=3,
            specs=[[{"rowspan":2}, {}, {}], [None, {}, {}]],
            subplot_titles=("PSD", "Mean", "Variance", "STD", "RMS")
        )

        # Add fig1's trace
        for trace in fig1.data:
            fig_combined.add_trace(trace, row=1, col=1)

        # Mean
        fig_combined.add_trace(
            go.Histogram(
                x=[feature["mean"] for feature in static_feature],
                nbinsx=20,
                name="Mean"
            ),
            row=1,
            col=2
        )

        # STD
        fig_combined.add_trace(
            go.Histogram(
                x=[feature["std"] for feature in static_feature],
                nbinsx=20,
                name="STD"
            ),
            row=2,
            col=2
        )

        # Variance
        fig_combined.add_trace(
            go.Histogram(
                x=[feature["variance"] for feature in static_feature],
                nbinsx=20,
                name="Variance"
            ),
            row=1,
            col=3
        )

        # RMS
        fig_combined.add_trace(
            go.Histogram(
                x=[feature["rms"] for feature in static_feature],
                nbinsx=20,
                name="RMS"
            ),
            row=2,
            col=3
        )

        # Copy frames and sliders from fig1
        fig_combined.frames = fig1.frames
        fig_combined.update_layout(
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Sample Index: "},
                "pad": {"t": 50},
                "steps": fig1.layout.sliders[0].steps,
                "len": 0.3,  # Cover only the left part
                "x": 0,
                "xanchor": "left"
            }]
        )

        # Update axes for fig1
        fig_combined.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig_combined.update_yaxes(title_text="Power Spectral Density", row=1, col=1)

        # Update overall layout
        fig_combined.update_layout(
            height=800,
            width=1600,  # Wider for 3 columns
            template="plotly_white",
            title=f"Combined PSD and Feature Analysis Dashboard({static_group_names[groupf].split('_')[1].capitalize()} Engine)",
            title_x=0.5
        )

        fig_combined.show()

        del fig1, fig_combined, frames, static_feature

    