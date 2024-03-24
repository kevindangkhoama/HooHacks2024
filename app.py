# %%
import base64
import datetime
import io
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.Div('Drag and Drop or', style={'display': 'inline-block', 'verticalAlign': 'middle'}),
            html.Div('t', style={'display': 'inline-block', 'color': 'white'}),  # Adding a white space
            html.A('Select Files', style={'display': 'inline-block', 'verticalAlign': 'middle'})
        ], style={'textAlign': 'center'}),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto'  # Center the upload box horizontally
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def machine_learning():
    # Generate synthetic data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)  # Generate 100 random values between 0 and 2
    y = 4 + 3 * X + np.random.randn(100, 1)  # Generate labels with noise

    # Train-test split
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot the training data, test data, and the regression line
    # plt.scatter(X_train, y_train, color='blue', label='Training Data')
    # plt.scatter(X_test, y_test, color='red', label='Test Data')
    # plt.plot(X_test, y_pred, color='green', label='Regression Line')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.show()


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Categorize purchases based on the 'Category' column
            purchase_categories = df.groupby('Category')['Amount'].sum().reset_index()
            # Filter out payments and credits from purchase categories
            purchase_categories = purchase_categories[purchase_categories['Category'] != 'Payments and Credits']
            # Identify repeat purchases by finding duplicates in the 'Description' column
            repeat_purchases = df[df.duplicated(subset='Description')]
            # Filter out payments and credits from repeat purchases
            repeat_purchases = repeat_purchases[repeat_purchases['Category'] != 'Payments and Credits']
            # Sort repeat purchases by description
            repeat_purchases = repeat_purchases.sort_values(by='Description')
            
            # Modify Description to first word
            repeat_purchases['Description'] = repeat_purchases['Description'].str.split().str[0]
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            purchase_categories = df.groupby('Category')['Amount'].sum().reset_index()
            purchase_categories = purchase_categories[purchase_categories['Category'] != 'Payments and Credits']
            repeat_purchases = df[df.duplicated(subset='Description')]
            repeat_purchases = repeat_purchases[repeat_purchases['Category'] != 'Payments and Credits']
            repeat_purchases = repeat_purchases.sort_values(by='Description')
            
            # Modify Description to first word
            repeat_purchases['Description'] = repeat_purchases['Description'].str.split().str[0]
            
        # Create stacked bar graph for repeat purchases
        repeat_purchases_fig = px.bar(repeat_purchases, x='Description', y='Amount', color='Category', barmode='stack')
        repeat_purchases_fig.update_layout(title='Repeat Purchases by Category', xaxis_title='Repeat Purchases')
        repeat_purchases_fig.update_xaxes(type='category')
        repeat_purchases_graph = dcc.Graph(id='repeat-purchases-graph', figure=repeat_purchases_fig)
        
        # Create pie chart for purchase categories
        pie_chart_fig = go.Figure(data=[go.Pie(
            labels=purchase_categories['Category'],
            values=purchase_categories['Amount'],
            hole=0.5,
            hoverinfo='label+percent',
            textinfo='value',
            textfont=dict(size=15),
            marker=dict(colors=px.colors.sequential.Viridis),
        )])
        pie_chart_fig.update_layout(title='Spending Category Proportion', margin=dict(t=50, b=50, r=50, l=50))
        pie_chart = dcc.Graph(id='pie-chart', figure=pie_chart_fig)
        
        # Create linear graph for purchases through the days
        df['Trans. Date'] = pd.to_datetime(df['Trans. Date'], format='%m/%d/%Y')
        purchases_through_days_fig = px.line(df, x='Trans. Date', y='Amount', title='Purchases Through Days')
        purchases_through_days_graph = dcc.Graph(id='purchases-through-days-graph', figure=purchases_through_days_fig)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        html.Div(className='row', children=[
            html.Div(className='six columns', children=[
                dash_table.DataTable(
                    data=purchase_categories.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in purchase_categories.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'whiteSpace': 'normal',
                        'textAlign': 'left',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                ),
            ]),
            html.Div(className='six columns', children=[
                pie_chart,
            ]),
        ]),

        html.Hr(),  # horizontal line
        
        html.Div([
            html.H5('Repeat Purchases:'),
        
            html.Details([
                html.Summary('Show/Hide Table'),
                dash_table.DataTable(
                    data=repeat_purchases.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in repeat_purchases.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'whiteSpace': 'normal',
                        'textAlign': 'left',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                ),
            ]),
        ], style={'display': 'inline-block', 'width': '48%'}),
        
        html.Div([
            html.H5('Repeat Purchases Graph:'),
            repeat_purchases_graph,
        ], style={'display': 'inline-block', 'width': '48%', 'float': 'right'}),
        
        html.Div([
            html.H5('Purchases Through Days:'),
            purchases_through_days_graph,
        ], style={'marginTop': '30px'}),

    ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
     app.run_server(debug=True)




