from flask import send_from_directory
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, "../modules/")
from utils import cols_dummies
from app_components import inputs, buttons, fig_proba
from plotly import graph_objs as go
import pickle

model = joblib.load("../model/trained_model.pkl")

app = dash.Dash()

@app.server.route('/favicon.ico')
def favicon():
    return app.server.send_static_file(
        os.path.join(app.root_path, 'static'),
        '400x400SML-01.png'
    )

app.css.append_css({'external_url': './static/style.css'})

app.layout = html.Div(children=[
    html.H1("Heart disease predictions"),
    html.Div(
        id='params-container',
        children=inputs,
        style={'paddingBottom': '10px'}
    ),
    html.Div(
        id='buttons-container',
        children=buttons,
        style={'marginTop':'15px'}
    ),
    html.Div(
        id='prediction-container',
        children=[]
    ),
    html.Div(
        id='plot-container',
        children=[
            dcc.Graph(
                id='plot',
                figure=fig_proba
            )
        ],
        style={'width':'65%', 'margin':'auto'}
    )
],
    style={'text-align':"center"}
)

@app.callback(
    Output('age-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_age(n_clicks):
    if n_clicks:
        return 54
    else:
        pass

@app.callback(
    Output('sex-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_sex(n_clicks):
    if n_clicks:
        return 'male'
    else:
        pass

@app.callback(
    Output('cp-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_cp(n_clicks):
    if n_clicks:
        return 3.0
    else:
        pass
    
@app.callback(
    Output('trestbps-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_trestbps(n_clicks):
    if n_clicks:
        return 132.0
    else:
        pass

@app.callback(
    Output('chol-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_chol(n_clicks):
    if n_clicks:
        return 247.0
    else:
        pass

@app.callback(
    Output('fbs-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_fbs(n_clicks):
    if n_clicks:
        return 0
    else:
        pass

@app.callback(
    Output('restecg-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_restecg(n_clicks):
    if n_clicks:
        return 1
    else:
        pass

@app.callback(
    Output('thalach-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_thalach(n_clicks):
    if n_clicks:
        return 150
    else:
        pass

@app.callback(
    Output('exang-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_exang(n_clicks):
    if n_clicks:
        return 0
    else:
        pass

@app.callback(
    Output('oldpeak-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_oldpeak(n_clicks):
    if n_clicks:
        return 1.0
    else:
        pass

@app.callback(
    Output('slop-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_slop(n_clicks):
    if n_clicks:
        return 2
    else:
        pass

@app.callback(
    Output('ca-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_ca(n_clicks):
    if n_clicks:
        return 1
    else:
        pass

@app.callback(
    Output('thal-input', 'value'),
    [Input('fill-button', 'n_clicks')]
)
def fill_thal(n_clicks):
    if n_clicks:
        return 3
    else:
        pass

@app.callback(
    Output('plot', 'figure'),
    [Input('predict_button', 'n_clicks')],
    [
        State('age-input', 'value'),
        State('sex-input', 'value'),
        State('cp-input', 'value'),
        State('trestbps-input', 'value'),
        State('chol-input', 'value'),
        State('fbs-input', 'value'),
        State('restecg-input', 'value'),
        State('thalach-input', 'value'),
        State('exang-input', 'value'),
        State('oldpeak-input', 'value'),
        State('slop-input', 'value'),
        State('ca-input', 'value'),
        State('thal-input', 'value')
    ]
)
def predict(n_clicks, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slop, ca, thal):
    if sex=='male':
        sex_encoded = 1
    else:
        sex_encoded = 0

    data = {
        'age': age,
        'sex': sex_encoded,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slop': slop,
        'ca': ca,
        'thal': thal
    }
    
    data_df = pd.DataFrame(pd.Series(data)).T
    
    X = pd.DataFrame(columns=cols_dummies)
    
    X = X.append(pd.get_dummies(
        data_df,
        columns=['ca', 'thal']
    ).iloc[0]).fillna(0)
    
    with open("../data/X_m.pkl", "rb") as f:
        X_m = pickle.load(f)
    
    with open("../data/X_f.pkl", "rb") as f:
        X_f = pickle.load(f)
    
    trace_pred = go.Scatter(
        x=X['age'],
        y=[model.predict_proba(X)[0,2]],
        mode='markers',
        name='prediction',
        marker=dict(
            color='red',
            size=10
        )
    )
    
    trace_m = go.Scatter(
        x=X_m['age'],
        y=X_m['proba'],
        name='males',
        line=dict(
            color='green'    
        )
    )
    
    trace_f = go.Scatter(
        x=X_f['age'],
        y=X_f['proba'],
        name='females',
        line=dict(
            color='orange'    
        )
    )
    
    layout = go.Layout(
        xaxis=dict(
            title='age'
        ),
        yaxis=dict(
            title='probability'
        )
    )

    if n_clicks:
        return go.Figure(data=[trace_m, trace_f, trace_pred], layout=layout)
    else:
        pass


if __name__=='__main__':
    app.run_server(debug=True, port=8888)