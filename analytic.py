import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#
# Function to load default data
@st.cache_data
def load_default_data():
    return pd.read_csv(
       'analytics_data.csv'
    )

# Function to load uploaded files (supports Excel and CSV)
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

# Sidebar for file upload or default dataset
st.sidebar.title("Upload or Load Dataset")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Default Dataset", "Upload Your Own Dataset")
)

# Load dataset based on user input
if data_source == "Default Dataset":
    data = load_default_data()
    st.sidebar.success("Default dataset loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        data = load_uploaded_file(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a dataset to proceed.")
        st.stop()


# Define color palettes
default_colors = px.colors.qualitative.Plotly
time_series_colors = px.colors.qualitative.Set2

# Convert 'Timestamp' to datetime format and localize timezone if needed
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert('UTC')

# Sidebar options for filtering
st.sidebar.header("Filters")
view_option = st.sidebar.radio(
    "Select View",
    ["Overall", "Customers & Visitors Analysis", "Shelf Analysis"]
)

# Time filters
if 'Timestamp' in data.columns:
    start_time = st.sidebar.time_input("Start Time", data['Timestamp'].min().time())
    end_time = st.sidebar.time_input("End Time", data['Timestamp'].max().time())
    data.set_index('Timestamp', inplace=True)
    if start_time < end_time:
        filtered_data = data.between_time(start_time, end_time).reset_index()
    else:
        st.error("Please ensure that the start time is earlier than the end time.")
        filtered_data = pd.DataFrame()  # Empty DataFrame to prevent further errors
else:
    st.error("Timestamp column is missing from the dataset.")
    filtered_data = data

# Additional Filtering Options
st.sidebar.subheader("Additional Filters")
if st.sidebar.checkbox("Show Filters"):
    # Multi-select filters for categorical columns
    for col in filtered_data.select_dtypes(include=['object']).columns:
        unique_values = filtered_data[col].dropna().unique()
        selected_values = st.sidebar.multiselect(f"Filter by {col}", options=unique_values, default=unique_values)
        filtered_data = filtered_data[filtered_data[col].isin(selected_values)]

    # Range filters for numeric columns
    for col in filtered_data.select_dtypes(include=['number']).columns:
        min_val, max_val = filtered_data[col].min(), filtered_data[col].max()
        selected_range = st.sidebar.slider(f"Filter by {col} range", min_val, max_val, (min_val, max_val))
        filtered_data = filtered_data[(filtered_data[col] >= selected_range[0]) & (filtered_data[col] <= selected_range[1])]


#
# # Convert 'Timestamp' to datetime format and localize timezone if needed
# if 'Timestamp' in data.columns:
#     data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert('UTC')
#
#
# # Sidebar options for filtering
# st.sidebar.header("Filters")
# view_option = st.sidebar.radio(
#     "Select View",
#     ["Overall", "People & Customer Analysis", "Shelf Analysis"]
# )
#
# # Time filters
# start_time = st.sidebar.time_input("Start Time", data['Timestamp'].min().time() if 'Timestamp' in data.columns else None)
# end_time = st.sidebar.time_input("End Time", data['Timestamp'].max().time() if 'Timestamp' in data.columns else None)
#
# # Filter data based on selected time, checking start time < end time
# if 'Timestamp' in data.columns:
#     data.set_index('Timestamp', inplace=True)
#     if start_time < end_time:
#         filtered_data = data.between_time(start_time, end_time).reset_index()
#     else:
#         st.error("Please ensure that the start time is earlier than the end time.")
#         filtered_data = pd.DataFrame()  # Empty DataFrame to prevent further errors
# else:
#     st.error("Timestamp column is missing from the dataset.")
#     filtered_data = data
# Refresh Button
if st.button("Refresh Dashboard"):
    st.experimental_set_query_params()

# Tooltip Message
tooltip_message = (
    "The dataset is a working process. You cannot open the Excel file directly, "
    "and no modifications can be made. You can only add data to existing columns, "
    "and you cannot change the column names."
)
st.markdown(
    f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
    unsafe_allow_html=True
)
# Overall Summary Stats
if view_option == "Overall":

    st.subheader("Overall Summary")

    # Define metrics and dynamically calculate their values
    metrics = {
        "Total Customers": data['Total Customers'].sum() if 'Total Customers' in data.columns else 0,
        "Total Visitors": data['Total Visitors'].sum() if 'Total Visitors' in data.columns else 0,
        "Queue Count": data['Queue Count'].sum() if 'Queue Count' in data.columns else 0,
        "Current Customers": data['Current Customers'].sum() if 'Current Customers' in data.columns else 0,
        "Current Visitor": data['Current Visitor'].sum() if 'Current Visitor' in data.columns else 0
    }

    # Calculate total checks (row count)
    total_checks = len(data)

    # Display metrics with gauges
    st.subheader("Overall Summary Metrics")
    gauge_figures = []
    # Define a list of distinct colors for each gauge
    gauge_colors = ['#00BFFF', '#32CD32', '#FF4500', '#DAA520', '#9370DB', '#FF6347']  # Professional colors

    # Create a gauge for total checks
    gauge_fig_total = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_checks,
        title={'text': "Total Checks", 'font': {'size': 20}},  # Font color removed
        gauge={
            'axis': {'range': [0, max(metrics.values(), default=0) * 1.1]},  # Set range based on max metric value
            'bar': {'color': gauge_colors[0]}  # Color for the total checks gauge
        },
        number={'font': {'size': 18, 'family': 'Arial', 'weight': 'bold'}}  # Bold font, color removed
    ))
    gauge_figures.append(gauge_fig_total)  # Append total checks gauge

    for (label, value), color in zip(metrics.items(), gauge_colors[1:]):  # Start from the second color
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': label, 'font': {'size': 20}},  # Font color removed
            gauge={
                'axis': {'range': [0, max(metrics.values(), default=0) * 1.1]},
                'bar': {'color': color}  # Use distinct color for each gauge
            },
            number={'font': {'size': 18, 'family': 'Arial', 'weight': 'bold'}}  # Bold font, color removed
        ))
        gauge_figures.append(fig)

    # Create containers for each row of gauges
    # First row of gauges (including total checks)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(gauge_figures[0], use_container_width=True)  # Total Checks
    with col2:
        st.plotly_chart(gauge_figures[1], use_container_width=True)  # Total Customers
    with col3:
        st.plotly_chart(gauge_figures[2], use_container_width=True)  # Total Visitors

    # Second row of gauges (remaining metrics)
    col4, col5, col6 = st.columns(3)

    with col4:
        st.plotly_chart(gauge_figures[3], use_container_width=True)  # Queue Count
    with col5:
        st.plotly_chart(gauge_figures[4], use_container_width=True)  # Current Customers
    with col6:
        st.plotly_chart(gauge_figures[5], use_container_width=True)  # Current Visitor

    # Optional: Summary bar chart for all metrics
    st.subheader("Metric Summary - Bar Chart")
    bar_fig = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        text=list(metrics.values()),
        textposition='auto',
        marker_color=gauge_colors  # Use the distinct colors for the bar chart
    ))

    bar_fig.update_layout(
        title="Overall Metric Summary",
        xaxis_title="Metrics",
        yaxis_title="Values",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=24),  # Font color removed
        xaxis_title_font=dict(size=18),  # Font color removed
        yaxis_title_font=dict(size=18)  # Font color removed
    )
    st.plotly_chart(bar_fig)

    # Combined line chart of Customers and Visitors
    fig = go.Figure()
    if 'Timestamp' in filtered_data.columns:
        if 'Total Customers' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Total Customers'],
                                     mode='lines', name='Total Customers', line=dict(color='#32CD32')))
        if 'Total Visitors' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Total Visitors'],
                                     mode='lines', name='Total Visitors', line=dict(color='#FF4500')))
    fig.update_layout(
        title="Overall Customers and Visitors Over Time",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=24),  # Font color removed
        xaxis_title_font=dict(size=18),  # Font color removed
        yaxis_title_font=dict(size=18)  # Font color removed
    )
    st.plotly_chart(fig)
# People & Customer Analysis
elif view_option == "Customers & Visitors Analysis":
    # Minute-Level Trends for Current Customers and Current Visitors
    fig = go.Figure()
    if 'Timestamp' in filtered_data.columns:
        if 'Current Customers' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Current Customers'],
                                     mode='lines', name='Current Customers',
                                     line=dict(color='#1E90FF', width=3)))  # Dodger Blue for line chart
        if 'Current Visitor' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Current Visitor'],
                                     mode='lines', name='Current Visitors',
                                     line=dict(color='#FF4500', width=3)))  # Orange Red for line chart

    fig.update_layout(
        title="Minute-Level Trends in Current Customers and Visitors",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
        xaxis_title_font=dict(size=18),  # Font color removed
        yaxis_title_font=dict(size=18)  # Font color removed
    )
    st.plotly_chart(fig)

    # Extract minute from the Timestamp for grouping
    filtered_data['Minute'] = filtered_data['Timestamp'].dt.strftime('%H:%M')  # Format as "HH:MM" for each unique minute

    # Group data by minute, calculating the average for Current Customers and Visitors
    minute_data = filtered_data.groupby('Minute')[['Current Customers', 'Current Visitor']].mean().reset_index()

    # Grouped Bar Chart for minute-level data
    fig = go.Figure()

    # Define metrics and colors for better distinction
    metrics = ['Current Customers', 'Current Visitor']
    colors = ['#4682B4', '#FF8C00']  # Steel Blue and Dark Orange for the bar chart

    # Add a bar for each metric
    for metric, color in zip(metrics, colors):
        if metric in minute_data.columns:
            fig.add_trace(go.Bar(
                x=minute_data['Minute'], y=minute_data[metric],
                name=metric, marker=dict(color=color)
            ))

    # Update layout to ensure it’s readable even with minute-level data
    fig.update_layout(
        title="Minute-Level Grouped Bar Chart for Current Customers and Visitors",
        xaxis_title="Time (Minute-Level)",
        yaxis_title="Average Count",
        barmode='group',  # Grouped bars for easy comparison
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
        xaxis_title_font=dict(size=18),  # Font color removed
        yaxis_title_font=dict(size=18)  # Font color removed
    )

    # Customize x-axis for better readability with minute-level data
    fig.update_xaxes(tickangle=45, nticks=20, tickformat="%H:%M")

    st.plotly_chart(fig)
# Shelf Analysis with Minute-Level Data
elif view_option == "Shelf Analysis":
    st.subheader("Shelf Analysis (Minute-Level)")

    # Ensure shelf columns exist in filtered_data
    shelves = [col for col in ['Shelf 1', 'Shelf 2', 'Shelf 3', 'Shelf 4'] if col in filtered_data.columns]

    if shelves:
        # Summing each shelf's usage
        shelf_totals = filtered_data[shelves].sum().reset_index()
        shelf_totals.columns = ['Shelf', 'Count']

        # 1. Total Shelf Usage (Bar Chart)
        fig1 = px.bar(shelf_totals, x='Shelf', y='Count', text='Count', color='Shelf',
                      color_discrete_sequence=['#1E90FF', '#FF6347', '#32CD32', '#FFD700'],  # Different colors for each shelf
                      title="Total Shelf Usage", template="plotly_dark")
        fig1.update_traces(texttemplate='%{text}', textposition='outside')
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the chart area
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
            xaxis_title_font=dict(size=18),  # Font color removed
            yaxis_title_font=dict(size=18)  # Font color removed
        )
        st.plotly_chart(fig1)

        # 2. Minute-Level Stacked Bar Chart for Shelf Usage
        filtered_data['Minute'] = filtered_data['Timestamp'].dt.strftime('%H:%M')  # Group by minute (hour:minute)
        minute_shelf_usage = filtered_data.groupby('Minute')[shelves].sum().reset_index()

        fig2 = go.Figure()
        colors = ['#6A5ACD', '#FF4500', '#00FA9A', '#FFD700']  # Unique colors for each shelf in stacked bar chart
        for shelf, color in zip(shelves, colors):
            fig2.add_trace(go.Bar(
                x=minute_shelf_usage['Minute'], y=minute_shelf_usage[shelf], name=shelf,
                marker_color=color  # Different colors for each shelf
            ))
        fig2.update_layout(
            title="Minute-Level Shelf Usage (Stacked Bar Chart)",
            xaxis_title="Time (Minute)",
            yaxis_title="Usage Count",
            barmode='stack',  # Stacks bars for cumulative view
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the chart area
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
            xaxis_title_font=dict(size=18),  # Font color removed
            yaxis_title_font=dict(size=18),  # Font color removed
            xaxis=dict(tickformat="%H:%M")
        )
        st.plotly_chart(fig2)

        # 3. Line Chart for Minute-Level Shelf Usage Over Time
        fig3 = go.Figure()
        line_colors = ['#00FF7F', '#FF1493', '#1E90FF', '#FFD700']  # Professional colors for line chart
        for shelf, color in zip(shelves, line_colors):
            fig3.add_trace(go.Scatter(
                x=filtered_data['Timestamp'], y=filtered_data[shelf],
                mode='lines+markers',  # Added markers for better visibility
                name=shelf,
                line=dict(color=color, width=2)  # Thicker line width and distinct color
            ))
        fig3.update_layout(
            title="Shelf Usage Over Time (Minute-Level Line Chart)",
            xaxis_title="Time",
            yaxis_title="Usage Count",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the chart area
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
            xaxis_title_font=dict(size=18),  # Font color removed
            yaxis_title_font=dict(size=18)  # Font color removed
        )
        st.plotly_chart(fig3)
