from some_plots import *
from other_files import *
from ARIMAX_by_Alarm_or_Warning import *
from ARIMAX_by_code import *
from ARIMAX_with_specific_code import *
from ARMA_by_Alarm_or_Warning import *
from ARMA_by_code import *
from ARMA_with_specific_codes import *
from Lasso import *
from PyTorch import *
from LSTM import *
from Calculations import *
from Style import Style
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import pandas as pd  
#from MySQL import dataSQL

# get local database (mySQL)
#work = dataSQL()



colors = {
    'background': '#1A1F25',
    'text': '#aeadaf',
    'backgroundDiv': '#282D38',
}


# Get the normal graphs from the other files
def getGraphsStat():
    return html.Div(id='stat-graphs', className='container-fluid', children=[
        html.Div(className='row', children=[
            html.Div(html.P(TrendAlarms()), ),
            html.Div(html.P(TrendAlarms13()), ),
        ]),
        html.Div(className='row', children=[
            html.Div(firstplot(640, 'd'), ),
        ]),
        html.Div(className='row',   children=[
            html.Div(barplot(210, 220, 260, 270), ),
        ]),
        html.Div(className='row',children=[
            html.Div(Biggest_AW_Pie()),
            html.Div(Biggest_Codes_Pie()),
        ]),
        html.Div(className='row', children=[
            html.Div(AlarmOrWarningCode('A', 13)),
            html.Div(StartAndAlarm()),
        ]),
        html.Div(className='row', children=[
            html.Div(QuarterErrors()),
            # html.Div(QuarterErrorBar('A')),
        ]),
        html.Div(className='row', children=[
            html.Div(Biggest_AW_bar('A')),
            html.Div(Alarms_Warnings())
        ]),
    ])

# get the ARMA graphs and fill in their parameters
def getGraphsARMA ():
    return html.Div(id='arma-graphs', className='container-fluid', children=[
        html.Div(className='row', children=[
            html.Div(ARMA_AW('A', 8, 6, 6)),
        ]),
        html.Div(className='row', children=[
            html.Div(ARMA_Code(640, 3, 2, 2)),
        ]),
        html.Div(className='row', children=[
            html.Div(ARMA_SpecificCode('A', 13, 4, 1, 3)),
        ]),

    ])

# get the ARIMA graphs and fill in their parameters
def getGraphsARIMAX():
    return html.Div(id= 'arimax-graphs', className='container-fluid', children=[
        html.Div(className='row', children=[
            html.Div(ARIMAX_AW('A', 10, 1, 4)),
        ]),
        html.Div(className='row', children=[
            html.Div(ARIMAX_Code(640, 4, 1, 3)),
        ]),
        html.Div(className='row', children=[
            html.Div(ARIMAX_SpecificCode('A', 13, 5, 1, 2))
        ]),

    ])

# get the other prediction models and fill in their parameters
def getGraphsOther():
    return html.Div(id= 'other-graphs', className='container-fluid', children=[
        # html.Div(className='row', children=[
        #     html.Div(LassoFunction('A')),
        # ]),
        # html.Div(className='row', children=[
        #     html.Div(PyTorch('A')),
        # ]),
        html.Div(className='row', children=[
            html.Div(LSTM(13,640,340,330,work,'W'))  # 7day interval
        ]),

    ])


# run the app on the server
if __name__ == '__main__':
    app = dash.Dash(__name__)

    # append extern css files
    app.css.append_css({
        "external_url": "https://p.w3layouts.com/demos_new/template_demo/07-04-2018/bake-demo_Free/1027606894/web/css/bootstrap.min.css"})
    app.css.append_css({
        "external_url": "https://p.w3layouts.com/demos_new/template_demo/07-04-2018/bake-demo_Free/1027606894/web/css/fontawesome-all.min.css"})
    app.css.append_css({
        "external_url": "https://p.w3layouts.com/demos_new/template_demo/07-04-2018/bake-demo_Free/1027606894/web/css/owl.carousel.css"})
    app.css.append_css({
        "external_url": "https://p.w3layouts.com/demos_new/template_demo/07-04-2018/bake-demo_Free/1027606894/web/css/style.css"})

    # make a layout for the DASHboard
    app.layout = html.Div(style={'backgroundColor':colors['background'], 'color':'#ffffff'}, children=[
        # set header and css of dashboard
        html.Div([
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                Style.getWebsiteCss() + Style.getWebsiteHeader()),
        ]),

        # make a dropdown menu for selecting the right models
        html.Div(className='container-fluid', children=[
            dcc.Dropdown(
                id='dropdown-graphs',
                style={'backgroundColor': colors['backgroundDiv'], 'color': colors['text']},
                options=[
                    {'label': 'Statistical', 'value': 'Stat'},
                    {'label': 'ARMA', 'value': 'ARMA'},
                    {'label': 'ARIMAX', 'value': 'ARIMAX'},
                    {'label': 'Other', 'value': 'Other'},
                ],
                # set standard value
                value='Stat',
            ),
            html.Div(id='output-container')
        ])

    ])

    # to make an interactive DASHboard, callbacks are being used. This callback is for the dropdown menu
    @app.callback(
        dash.dependencies.Output('output-container', 'children'),
        [dash.dependencies.Input('dropdown-graphs', 'value')])
    def render_content(value):
        # making an if-statement for after selecting a value in the dropdown menu.Then he shows the graphs of that value
        if value == 'ARMA':
            return html.Div(id='output-ARMA', children=[
                html.Div(getGraphsARMA())
            ])
        elif value == 'ARIMAX':
            return html.Div(id='output-ARIMAX', children=[
                html.Div(getGraphsARIMAX())
            ])
        elif value == 'Stat':
            return html.Div(id='output-stat', children=[
                html.Div(getGraphsStat())
            ])
        elif value == 'Other':
            return html.Div(id='output-other', children=[
                html.Div(getGraphsOther())
            ])


    app.run_server(debug=True)
