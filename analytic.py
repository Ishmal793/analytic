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
    ["Overall", "People & Customer Analysis", "Shelf Analysis"]
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
        "Total People Entered": data['Total People Entered'].sum() if 'Total People Entered' in data.columns else 0,
        "Total Customers": data['Total Customers'].sum() if 'Total Customers' in data.columns else 0,
        "Total Visitors": data['Total Visitors'].sum() if 'Total Visitors' in data.columns else 0,
        "Queue Count": data['Queue Count'].sum() if 'Queue Count' in data.columns else 0,
        "Current Customers": data['Current Customers'].sum() if 'Current Customers' in data.columns else 0,
        "Current Visitor": data['Current Visitor'].sum() if 'Current Visitor' in data.columns else 0
    }

    # Calculate total checks
    total_checks = sum(metrics.values())

    # Display metrics with gauges
    st.subheader("Overall Summary Metrics")
    gauge_figures = []
    gauge_colors = ['#FF5733', '#33FF57', '#3357FF', '#F9FF33', '#FF33A8', '#33FFF9']  # Different colors for each gauge

    # Create a gauge for total checks
    gauge_fig_total = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_checks,
        title={'text': "Total Checks"},
        gauge={
            'axis': {'range': [0, max(metrics.values()) * 1.1]},  # Set range based on max metric value
            'bar': {'color': "#007bff"}
        },
        number={'font': {'color': "#007bff"}}
    ))
    gauge_figures.append(gauge_fig_total)  # Append total checks gauge

    for (label, value), color in zip(metrics.items(), gauge_colors):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': label},
            gauge={
                'axis': {'range': [0, max(metrics.values()) * 1.1]},
                'bar': {'color': color}
            },
            number={'font': {'color': color}}
        ))
        gauge_figures.append(fig)

    # Create containers for each row of gauges
    # First row of gauges (including total checks)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(gauge_figures[0], use_container_width=True)  # Total Checks
    with col2:
        st.plotly_chart(gauge_figures[1], use_container_width=True)  # Total People Entered
    with col3:
        st.plotly_chart(gauge_figures[2], use_container_width=True)  # Total Customers
    with col4:
        st.plotly_chart(gauge_figures[3], use_container_width=True)  # Total Visitors

    # Second row of gauges (remaining metrics)
    col5, col6, col7 = st.columns(3)

    with col5:
        st.plotly_chart(gauge_figures[4], use_container_width=True)  # Queue Count
    with col6:
        st.plotly_chart(gauge_figures[5], use_container_width=True)  # Current Customers
    with col7:
        st.plotly_chart(gauge_figures[6], use_container_width=True)  # Current Visitor

    # Optional: Summary bar chart for all metrics
    st.subheader("Metric Summary - Bar Chart")
    bar_fig = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        text=list(metrics.values()),
        textposition='auto',
        marker_color=['cyan', 'magenta', 'yellow', 'orange', 'blue', 'purple']
    ))

    bar_fig.update_layout(
        title="Overall Metric Summary",
        xaxis_title="Metrics",
        yaxis_title="Values",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(bar_fig)

    # Combined line chart of People Entered, Customers, and Visitors
    fig = go.Figure()
    if 'Timestamp' in filtered_data.columns:
        if 'Total People Entered' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Total People Entered'],
                                     mode='lines', name='Total People Entered', line=dict(color='cyan')))
        if 'Total Customers' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Total Customers'],
                                     mode='lines', name='Total Customers', line=dict(color='magenta')))
        if 'Total Visitors' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Total Visitors'],
                                     mode='lines', name='Total Visitors', line=dict(color='yellow')))
    fig.update_layout(
        title="Overall People, Customers, and Visitors Over Time",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig)


# People & Customer Analysis
elif view_option == "People & Customer Analysis":
    # Minute-Level Trends for Queue Count, Current Customers, and Current Visitors
    fig = go.Figure()
    if 'Timestamp' in filtered_data.columns:
        if 'Queue Count' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Queue Count'],
                                     mode='lines', name='Queue Count', line=dict(color='orange')))
        if 'Current Customers' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Current Customers'],
                                     mode='lines', name='Current Customers', line=dict(color='cyan')))
        if 'Current Visitor' in filtered_data.columns:
            fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Current Visitor'],
                                     mode='lines', name='Current Visitors', line=dict(color='magenta')))

    fig.update_layout(
        title="Minute-Level Trends in Queue Count, Current Customers, and Visitors",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig)
    # Minute-Level Queue Count vs Current Customers Scatter Plot
    if 'Queue Count' in filtered_data.columns and 'Current Customers' in filtered_data.columns:
        fig = px.scatter(
            filtered_data, x='Queue Count', y='Current Customers',
            title="Minute-Level Queue Count vs Current Customers",
            labels={"Queue Count": "Queue Count", "Current Customers": "Current Customers"},
            template="plotly_dark"
        )
        fig.update_traces(marker=dict(size=10, color='cyan', line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig)
    # Extract minute from the Timestamp for grouping
    filtered_data['Minute'] = filtered_data['Timestamp'].dt.strftime(
        '%H:%M')  # Format as "HH:MM" for each unique minute

    # Group data by minute, calculating the average for Queue Count, Current Customers, and Visitors
    minute_data = filtered_data.groupby('Minute')[
        ['Queue Count', 'Current Customers', 'Current Visitor']].mean().reset_index()

    # Grouped Bar Chart for minute-level data
    fig = go.Figure()

    # Define metrics and colors for better distinction
    metrics = ['Queue Count', 'Current Customers', 'Current Visitor']
    colors = ['orange', 'cyan', 'magenta']

    # Add a bar for each metric
    for metric, color in zip(metrics, colors):
        if metric in minute_data.columns:
            fig.add_trace(go.Bar(
                x=minute_data['Minute'], y=minute_data[metric],
                name=metric, marker=dict(color=color)
            ))

    # Update layout to ensure it’s readable even with minute-level data
    fig.update_layout(
        title="Minute-Level Grouped Bar Chart for Queue Count, Current Customers, and Visitors",
        xaxis_title="Time (Minute-Level)",
        yaxis_title="Average Count",
        barmode='group',  # Grouped bars for easy comparison
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
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
                      title="Total Shelf Usage", template="plotly_dark")
        fig1.update_traces(texttemplate='%{text}', textposition='outside')
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig1)

        # 2. Minute-Level Stacked Bar Chart for Shelf Usage
        filtered_data['Minute'] = filtered_data['Timestamp'].dt.strftime('%H:%M')  # Group by minute (hour:minute)
        minute_shelf_usage = filtered_data.groupby('Minute')[shelves].sum().reset_index()

        fig2 = go.Figure()
        for shelf in shelves:
            fig2.add_trace(go.Bar(
                x=minute_shelf_usage['Minute'], y=minute_shelf_usage[shelf], name=shelf
            ))
        fig2.update_layout(
            title="Minute-Level Shelf Usage (Stacked Bar Chart)",
            xaxis_title="Time (Minute)",
            yaxis_title="Usage Count",
            barmode='stack',  # Stacks bars for cumulative view
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickformat="%H:%M")
        )
        st.plotly_chart(fig2)

        # 3. Line Chart for Minute-Level Shelf Usage Over Time
        fig3 = go.Figure()
        for shelf in shelves:
            fig3.add_trace(go.Scatter(
                x=filtered_data['Timestamp'], y=filtered_data[shelf],
                mode='lines', name=shelf
            ))
        fig3.update_layout(
            title="Shelf Usage Over Time (Minute-Level Line Chart)",
            xaxis_title="Time",
            yaxis_title="Usage Count",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3)

        # 4. Heatmap for Minute-Level Shelf Usage
        heatmap_data = filtered_data.groupby('Minute')[shelves].sum().T  # Transpose for heatmap compatibility

        fig4 = px.imshow(heatmap_data, labels=dict(x="Minute", y="Shelf", color="Usage Count"),
                         x=minute_shelf_usage['Minute'], y=shelves,
                         title="Shelf Usage Heatmap by Minute", template="plotly_dark",
                         color_continuous_scale="Viridis")
        fig4.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig4)

    else:
        st.warning("No shelf data available in the uploaded dataset.")
