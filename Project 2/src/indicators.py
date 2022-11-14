import numpy as np
import pandas as pd

data = {'Mean Velocity'     : [0.02053689278, 0.01781118973, 0.01281222929, 0.01497316292],
        'Max Velocity'     : [0.1080186109, 0.108095226, 0.1102518943, 0.06124246794],
        'Curl'     : [0.1629225835, 0.1144479677, 0.1474560044, 0.1673521209],
        'Abs Curl'     : [0.2676441094, 0.2283399985, 0.1981550701 , 0.2423502585 ],
        'Zero Vel'     : [0.1876336914, 0.1488634601, 0.4394330839, 0.3556718005],
        #'Quant 03'     : [0.0001532020397, 2.217085041e-05, 1.189276663e-06, 8.52224204e-05],
        'Index Title'  : ["CC", "CW", "WS", "CL"]}
df = pd.DataFrame(data)
df = df.set_index('Index Title')
# %%
mean_ = df.mean()
std_ = df.std()

normalized_df = (df - mean_) / std_

# %%

negative_indicator =  normalized_df['Abs Curl'] + normalized_df['Zero Vel']
                      #+ normalized_df['Mean Velocity'] + normalized_df['Max Velocity']
indicator2 = normalized_df['Curl'] - normalized_df['Mean Velocity']
indicator3 = normalized_df['Curl'] - normalized_df['Mean Velocity'] + normalized_df['Zero Vel']
