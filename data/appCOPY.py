import os
import pandas as pd
import psycopg2
import configparser
from flask import Flask, render_template
import plotly.express as px
import datetime
import numpy as np
import re
from datetime import datetime

# Specify the path to the 'dashboard' folder
dashboard_path = r'C:\Users\wjcmw\Documents\Data-Science-And-Society-main\Data-Science-And-Society-main\dashboard'
os.chdir(dashboard_path)

# Load database configurations from Config.txt
config = configparser.ConfigParser()
config.read('Config.txt')
db_config = config['DATABASE']

app = Flask(__name__)

def connect_to_db():
    try:
        cnx = psycopg2.connect(
            user=db_config['USER'],
            password=db_config['PASSWORD'],
            host=db_config['HOST'],
            port=db_config['PORT'],
            database=db_config['NAME']
        )
        print("Connected successfully!")
        return cnx
    except Exception as e:
        print(f"Error: {e}")
        return None

def fetch_incident_data():
    cnx = connect_to_db()
    if cnx:
        try:
            cur = cnx.cursor()
            cur.execute("SELECT date, incidents FROM incidentdata")
            columns = ['date', 'incidents']
            data = cur.fetchall()
            cnx.close()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"Error: {e}")
            return None

def fetch_total_incidents_data():
    cnx = connect_to_db()
    if cnx:
        try:
            cur = cnx.cursor()
            cur.execute("SELECT SUM(incidents) FROM incidentdata")
            total_incidents = cur.fetchone()[0]  # Fetch the total incidents
            cnx.close()
            return total_incidents
        except Exception as e:
            print(f"Error: {e}")
            return None


def fetch_music_data():
    cnx = connect_to_db()
    if cnx:
        try:
            cur = cnx.cursor()
            cur.execute("SELECT week, AVG(valence) FROM music_charts GROUP BY week ORDER BY week")
            columns = ['week', 'average_valence']
            data = cur.fetchall()
            cnx.close()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"Error: {e}")
            return None

def fetch_music_data2():
    cnx = connect_to_db()
    if cnx:
        try:
            cur = cnx.cursor()
            cur.execute("SELECT * FROM music_charts")
            data = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            cnx.close()
            return pd.DataFrame(data, columns=column_names)
        except Exception as e:
            print(f"Error: {e}")
            return None


def create_total_incidents_figure():
    total_incidents = fetch_total_incidents_data()
    if total_incidents is not None:
        return int(total_incidents)

def create_incidents_plot():
    incident_data = fetch_incident_data()
    if incident_data is not None:
        # Define a mapping dictionary for non-standard month abbreviations
        month_mapping = {
            'jan': 'Jan',
            'feb': 'Feb',
            'mrt': 'Mar',
            'apr': 'Apr',
            'mei': 'May',
            'jun': 'Jun',
            'jul': 'Jul',
            'aug': 'Aug',
            'sep': 'Sep',
            'okt': 'Oct',
            'nov': 'Nov',
            'dec': 'Dec',
        }

        # Replace non-standard month abbreviations with standard abbreviations
        incident_data['date'] = incident_data['date'].str[:3].map(month_mapping) + incident_data['date'].str[3:]

        # Parse the "date" column to datetime
        incident_data['date'] = pd.to_datetime(incident_data['date'], format='%b-%y')

        # Create a Plotly line chart
        fig = px.line(incident_data, x='date', y='incidents', title='Incidents Per Month')
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b '%y"
        )
        return fig
    
def create_music_scatter_plot():
    music_df = fetch_music_data2()
    if music_df is not None:
        fig = px.scatter(music_df, x='week', y='valence', title='Scatter Plot of Weekly Valence')

        # Define valence ranges for different moods and corresponding mood labels
        valence_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 1.0)]  # You can adjust these ranges
        mood_labels = ['Sad', 'Neutral', 'Happy']  # Mood labels corresponding to valence ranges

        # Define colors for different moods
        mood_colors = ['red', 'blue', 'green']

        # Assign colors based on valence ranges and specify mood labels in the legend
        for i, (start, end) in enumerate(valence_ranges):
            fig.add_shape(type='rect', x0=min(music_df['week']), x1=max(music_df['week']),
                          y0=start, y1=end,
                          fillcolor=mood_colors[i], opacity=0.2, layer='below', line_width=0, name=mood_labels[i])

        fig.update_traces(marker=dict(size=10, opacity=0.6), selector=dict(mode='markers'))
        fig.update_xaxes(title_text='Week of the Year')
        fig.update_yaxes(title_text='Distribution of Weekly Valence')
        fig.update_layout(showlegend=True)  # Show the legend

        return fig

# Create a function to create the second graph
def create_music_chart_figure():
    music_df = fetch_music_data()
    if music_df is not None:
        fig = px.line(music_df, x='week', y='average_valence', title='Average Valence Over Time')

# Define valence ranges for different moods and corresponding mood labels
        valence_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 1.0)]  # You can adjust these ranges
        mood_labels = ['Sad', 'Neutral', 'Happy']  # Mood labels corresponding to valence ranges

        # Define colors for different moods
        mood_colors = ['red', 'blue', 'green']

        # Assign colors based on valence ranges and specify mood labels in the legend
        for i, (start, end) in enumerate(valence_ranges):
            fig.add_shape(type='rect', x0=min(music_df['week']), x1=max(music_df['week']),
                          y0=start, y1=end,
                          fillcolor=mood_colors[i], opacity=0.2, layer='below', line_width=0, name=mood_labels[i])

        fig.update_traces(marker=dict(size=10, opacity=0.6), selector=dict(mode='markers'))
        fig.update_xaxes(title_text='Week of the Year')
        fig.update_yaxes(title_text='Average Weekly Valence')
        fig.update_layout(showlegend=True)  # Show the legend

        return fig

def calculate_current_valence_and_trend():
    music_df = fetch_music_data()
    if music_df is not None:
        current_week_valence = music_df['average_valence'].iloc[-1]  # Get the valence for the most recent week
        prev_week_valence = music_df['average_valence'].iloc[-2]  # Get the valence for the previous week

        # Determine the trend
        if current_week_valence > prev_week_valence:
            trend = "up"
        elif current_week_valence < prev_week_valence:
            trend = "down"
        else:
            trend = "stable"

        return current_week_valence, trend

# Define mood categories and corresponding valence thresholds
mood_categories = {
    "Sad": (0.0, 0.3),
    "Neutral": (0.3, 0.6),
    "Happy": (0.6, 1.0)
}

# Define custom colors for each mood
mood_categories = {
    "Sad": (0.0, 0.3),
    "Neutral": (0.3, 0.6),
    "Happy": (0.6, 1.0)
}

def categorize_mood(valence):
    for mood, (min_val, max_val) in mood_categories.items():
        if min_val <= valence <= max_val:
            return mood
    return "Unknown"  # If valence doesn't fall into any category

def create_mood_distribution_chart():
    music_df = fetch_music_data2()  # Fetch your music data

    # Apply the categorize_mood function to categorize songs
    music_df['mood'] = music_df['valence'].apply(categorize_mood)

    # Group the data by mood categories and count the number of songs in each category
    mood_distribution = music_df['mood'].value_counts().reset_index()
    mood_distribution.columns = ['Mood', 'Count']

    # Create the pie chart using Plotly Express
    fig = px.pie(mood_distribution, names='Mood', values='Count', title='Mood Distribution of Songs')

    # Specify custom colors for each mood
    colors = {'Sad': 'red', 'Neutral': 'blue', 'Happy': 'green'}

    # Update trace colors with custom colors
    fig.update_traces(marker=dict(colors=[colors[mood] for mood in mood_distribution['Mood']]))

    return fig

def calculate_monthly_incident_change():
    incident_data = fetch_incident_data()
    if incident_data is not None:
        # Define a mapping dictionary for non-standard month abbreviations
        month_mapping = {
            'jan': 'Jan',
            'feb': 'Feb',
            'mrt': 'Mar',
            'apr': 'Apr',
            'mei': 'May',
            'jun': 'Jun',
            'jul': 'Jul',
            'aug': 'Aug',
            'sep': 'Sep',
            'okt': 'Oct',
            'nov': 'Nov',
            'dec': 'Dec',
        }

        # Replace non-standard month abbreviations with standard abbreviations
        incident_data['date'] = incident_data['date'].str[:3].map(month_mapping) + incident_data['date'].str[3:]

        # Convert the "date" column to datetime
        incident_data['date'] = pd.to_datetime(incident_data['date'], format='%b-%y')

        # Sort the data by date
        incident_data = incident_data.sort_values(by='date')

        # Calculate the monthly change in incidents
        incident_data['monthly_change'] = incident_data['incidents'].diff()

        # Extract the last two rows (latest month and the month before)
        latest_month_data = incident_data.iloc[-1]
        previous_month_data = incident_data.iloc[-2]

        # Calculate the change from the last month
        change_from_last_month = latest_month_data['monthly_change']

        # Calculate the percentage change
        percentage_change = (change_from_last_month / previous_month_data['incidents']) * 100

        return {
            'change_from_last_month': change_from_last_month,
            'percentage_change': percentage_change,
        }
    else:
        return None

@app.route("/")
def dashboard():
    incidents_chart_fig = create_incidents_plot()
    incidents_chart_plot_html = incidents_chart_fig.to_html(full_html=False, include_plotlyjs='cdn') if incidents_chart_fig is not None else ''

    monthly_incident_change = calculate_monthly_incident_change()

    # Get the total incidents text
    total_incidents_text = create_total_incidents_figure()

    # Create the second graph
    music_chart_fig = create_music_chart_figure()
    music_chart_plot_html = music_chart_fig.to_html(full_html=False, include_plotlyjs='cdn') if music_chart_fig is not None else ''

    # Create the Mood Distribution Pie Chart
    mood_distribution_chart = create_mood_distribution_chart()
    mood_distribution_chart_html = mood_distribution_chart.to_html(full_html=False, include_plotlyjs='cdn') if mood_distribution_chart is not None else ''

    music_scatter_fig = create_music_scatter_plot()
    music_scatter_plot_html = music_scatter_fig.to_html(full_html=False, include_plotlyjs='cdn') if music_scatter_fig is not None else ''

    current_valence, valence_trend = calculate_current_valence_and_trend()

    last_data_update = datetime.now().strftime("%d-%m-%Y %H:%M")

    change_from_last_month = None
    percentage_change = None

    if monthly_incident_change is not None:
        # Extract the values from the dictionary
        change_from_last_month = monthly_incident_change.get('change_from_last_month')
        percentage_change = monthly_incident_change.get('percentage_change')

    return render_template("index.html", last_data_update=last_data_update, incidents_chart_plot=incidents_chart_plot_html, total_incidents_text=total_incidents_text,
                           music_chart_plot=music_chart_plot_html, music_scatter_plot=music_scatter_plot_html,
                           current_valence=current_valence, valence_trend=valence_trend,
                           mood_distribution_chart=mood_distribution_chart_html,
                           change_from_last_month=change_from_last_month, percentage_change=percentage_change)

if __name__ == "__main__":
    app.run(debug=True)
