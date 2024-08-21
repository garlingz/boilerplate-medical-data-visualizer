import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv', index_col= 'id')

def calculate_BMI(row):
    #   Convert to Meters
    height_meters = row['height'] * 0.01
    bmi = round((row['weight'] / (height_meters ** 2)), 2)
    return bmi

#   Formatting a few categories
df['BMI'] = df.apply(calculate_BMI, axis= 1)
df['overweight'] = df['BMI'] > 25
df['overweight'] = df['overweight'].astype(int)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
df = df.reset_index()
# Drawing the first plot
def draw_cat_plot():
    df_cat = pd.melt(df,
                        id_vars= 'cardio', 
                        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
                        var_name= 'variable',
                        value_name= 'value'
                        )
    df_grouped = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name= 'count')

    fig = sns.catplot(data= df_grouped, x= 'variable', y= 'count', col= 'cardio', hue= 'value', kind= 'bar')
    fig.savefig('catplot.png')
    return fig

# Drawing the second plot
def draw_heat_map():
    lower_height = df['height'].quantile(0.025)
    upper_height = df['height'].quantile(0.975)
    lower_weight = df['weight'].quantile(0.025)
    upper_weight = df['weight'].quantile(0.975)

    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= lower_height) &
        (df['height'] <= upper_height) &
        (df['weight'] >= lower_weight) &
        (df['weight'] <= upper_weight)
    ]
    df_heat = df_heat.drop('BMI', axis= 1)

    corr = df_heat.corr().round(1)
    mask = np.triu(np.ones_like(corr, dtype= bool))

    fig, ax = plt.subplots(figsize= (10, 8))
    sns.heatmap(data= corr, mask= mask, fmt= ".1f", annot= True, ax= ax)

    fig.savefig('heatmap.png')
    return fig
