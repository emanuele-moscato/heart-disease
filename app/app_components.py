import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import pickle

inputs = [
    html.Div([
        dcc.Input(
            placeholder='age',
            id='age-input'
        ),
        dcc.Input(
            placeholder='sex',
            id='sex-input'
        ),
        dcc.Input(
            placeholder='cp',
            id='cp-input'
        ),
        dcc.Input(
            placeholder='trestbps',
            id='trestbps-input'
        ),
        dcc.Input(
            placeholder='chol',
            id='chol-input'
        )
    ]),
    html.Div([
        dcc.Input(
            placeholder='fbs',
            id='fbs-input'
        ),
        dcc.Input(
            placeholder='restecg',
            id='restecg-input'
        ),
        dcc.Input(
            placeholder='thalach',
            id='thalach-input'
        ),
        dcc.Input(
            placeholder='exang',
            id='exang-input'
        ),
        dcc.Input(
            placeholder='oldpeak',
            id='oldpeak-input'
        )
    ]),
    html.Div([
        dcc.Input(
            placeholder='slop',
            id='slop-input'
        ),
        dcc.Input(
            placeholder='ca',
            id='ca-input'
        ),
        dcc.Input(
            placeholder='thal',
            id='thal-input'
        )
    ])
]

buttons = [
    html.Button(
        "Fill!",
        id='fill-button'
    ),
    html.Button(
        "Predict!",
        id='predict_button',
        style={'marginLeft': '30px'}
    )
]

with open("../data/X_m.pkl", "rb") as f:
    X_m = pickle.load(f)
    
with open("../data/X_f.pkl", "rb") as f:
    X_f = pickle.load(f)
    
fig_proba = go.Figure(
    data=[
        go.Scatter(
            x=X_m['age'],
            y=X_m['proba'],
            name='males',
            line=dict(
                color='green'    
            )
        ),
        go.Scatter(
            x=X_f['age'],
            y=X_f['proba'],
            name='females',
            line=dict(
                color='orange'    
            )
        ),
    ],
    layout=go.Layout(
        xaxis=dict(
            title='age'
        ),
        yaxis=dict(
            title='probability'
        )
    )
)