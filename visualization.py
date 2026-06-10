import plotly.express as px
import pandas as pd


def plot_probability_chart(labels, probabilities):
    fig = px.bar(
        x=labels,
        y=probabilities,
        labels={'x': 'Emotion', 'y': 'Probability'},
        title='Emotion Probability Distribution',
        color=labels,
        template='plotly_white',
    )
    fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
    return fig


def plot_distribution_chart(history_df: pd.DataFrame):
    if history_df.empty:
        return px.bar(title='No predictions yet')
    fig = px.histogram(
        history_df,
        x='prediction',
        color='prediction',
        title='Prediction Distribution',
        labels={'prediction': 'Emotion'},
        template='plotly_white',
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_history_table(history_entries):
    if not history_entries:
        return pd.DataFrame(columns=['text', 'prediction', 'confidence'])
    history_df = pd.DataFrame(history_entries)
    history_df['confidence'] = history_df['confidence'].astype(float)
    return history_df
