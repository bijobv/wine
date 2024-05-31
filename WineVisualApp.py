import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page layout

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Apply Arial font to the entire app */
    body, div, h1, h2, h3, h4, h5, h6, p, li, span, input, select, textarea {
        font-family: 'Arial', sans-serif;
    }
    /* Apply light theme */
    body {
        background-color: #ffffff;
        color: #000000;
    }
            
    /* Center the table */
    .center-table {
        width: 80%; /* Set the width to 80% */
        margin: 0 auto; /* Center the table container */
    }
    /* Ensure table inside the container takes full width */
    .center-table table {
        width: 100%; /* Table should take full width of its container */
    }
            
    /* Left align the values in the table */
    .center-table table td {
    text-align: left; /* Align text to the left */
    }
            
    .stButton>button {
        margin: 0 auto; /* Center the button */
        display: block; /* Ensure the button is a block element */
    }
    </style>
    """, unsafe_allow_html=True)


# Title of the app
st.title('Wine Analysis')

# Load the dataset
data1 = pd.read_csv('winequality-red.csv', delimiter=';')
data1['wine type'] = 'R'
data1['wine'] = 'Red'

data2 = pd.read_csv('winequality-white.csv', delimiter=';')
data2['wine type'] = 'W'
data2['wine'] = 'White'

df = pd.concat([data1, data2], ignore_index=True)


#### Reroder columns
dersired_order = ['alcohol', 'citric acid', 'fixed acidity', 'volatile acidity', 'pH', 'residual sugar', 'total sulfur dioxide', 'free sulfur dioxide', 'sulphates', 'chlorides', 'density', 'quality', 'wine type', 'wine']

data1 = data1[dersired_order]
data2 = data2[dersired_order]
df = df[dersired_order]


###########################
########### 1A ############
st.subheader('1A. Grouping of red and white wines')

# Paths to the icons
red_wine_icon_path = "red_wine.png"  
white_wine_icon_path = "white_wine.png"  


image_paths = {
    "Red": "som_red_wine.png",  
    "White": "som_white_wine.png",  
    "All": "som_wine.png"  
}



# Display the selected image
col1, col2 = st.columns([8, 10])  # Adjust the first number to control the width ratio
with col1:
    option = st.selectbox("Select view", ("All", "Red", "White"), index=0)
    if option:
        st.image(image_paths[option], use_column_width=True)
with col2:
    if option:
        st.write("<br>" * 4, unsafe_allow_html=True)
        # Create a horizontal layout for "Red Wine" image and text
        red_wine_col1, red_wine_col2 = st.columns([0.5, 20])
        with red_wine_col1:
            st.image(red_wine_icon_path, width=25)
        with red_wine_col2:
            st.write("Red Wine")

        # Create a horizontal layout for "White Wine" image and text
        white_wine_col1, white_wine_col2 = st.columns([0.5, 20])
        with white_wine_col1:
            st.image(white_wine_icon_path, width=25)
        with white_wine_col2:
            st.write("White Wine")




###########################
########### 1B ############
st.write("<br>" * 2, unsafe_allow_html=True)
st.subheader('1B. Association between Features and Quality')

df = df.drop(['wine type'], axis=1)

# Define a custom color map for quality
color_map = {
    3: '#0E078B',  
    4: '#6C00DA',
    5: '#8F2698',
    6: '#BE5377',  
    7: '#DE805B',
    8: '#F1B84E',
    9: '#FF0000'    
}

# Melt the DataFrame to long format
df_melted = df.melt(id_vars=['quality', 'wine'], var_name='attribute', value_name='value')
df_melted['attribute'] = df_melted['attribute'].apply(lambda x: x.title() if x != 'pH' else x)


# Mapping dictionary for attribute to unit
unit_mapping = {
    'Fixed Acidity': 'g(tartaric acid)/dm3',
    'Volatile Acidity': 'g(acetic acid)/dm3',
    'Citric Acid': 'g/dm3',
    'Residual Sugar': 'g/dm3',
    'Chlorides': 'g(sodium chloride)/dm3',
    'Free Sulfur Dioxide': 'mg/dm3',
    'Total Sulfur Dioxide': 'mg/dm3',
    'Density': 'g/cm3',
    'pH': '',
    'Sulphates': 'g(potassium sulphate)/dm3',
    'Alcohol': 'vol.%3',
}

# Create the 'Unit' column based on the 'Attribute' column
df_melted['unit'] = df_melted['attribute'].map(unit_mapping)


# Streamlit filters
all_attributes = df_melted['attribute'].unique()  # Get all unique attribute values
selected_attributes = st.multiselect('Select features:', all_attributes, default=all_attributes)

if not selected_attributes:
    st.warning("Please select at least one attribute.")
    st.stop() 

selected_wine_type = st.selectbox('Select wine:', df['wine'].unique(), index=0)

# Filter the DataFrame based on selected attributes and wine type
filtered_df = df_melted[df_melted['attribute'].isin(selected_attributes) & 
                        (df_melted['wine'] == selected_wine_type)]

# Create subplots
num_attributes = filtered_df['attribute'].nunique()
rows = (num_attributes // 4) + (num_attributes % 4 > 0)
fig = make_subplots(rows=rows, cols=4, subplot_titles=filtered_df['attribute'].unique(), vertical_spacing=0.12, horizontal_spacing=0.1)

# Add individual box plots to the subplots
for i, attribute in enumerate(filtered_df['attribute'].unique()):
    row = i // 4 + 1
    col = i % 4 + 1
    
    # Create a box plot for the current attribute
    fig_box = px.box(filtered_df[filtered_df['attribute'] == attribute], 
                     x='quality', y='value', 
                     color='quality', 
                     color_discrete_map=color_map, 
                     template='plotly_dark')
    
    # Add traces from the box plot to the subplot
    for trace in fig_box['data']:
        fig.add_trace(trace, row=row, col=col)

    unit = df_melted[df_melted['attribute'] == attribute]['unit'].iloc[0]  # Get unit for the attribute
    fig.update_yaxes(title=f'{unit}', row=i//4 + 1, col=i%4 + 1,
                     title_font=dict(family='Arial', size=12, color='black'))

# Define the tick values and tick labels for the x-axis
quality_values = df['quality'].unique()
quality_values.sort()
tickvals = list(quality_values)
ticktext = [str(val) for val in tickvals]

# Update layout to ensure all tick labels show up on x-axis
for i in range(1, rows + 1):
    for j in range(1, 5):
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=i, col=j)
        fig.update_xaxes(title_font=dict(family='Arial', size=12, color='black'), row=i, col=j)
        fig.update_yaxes(title_font=dict(family='Arial', size=12, color='black'), row=i, col=j)

# Update subplot titles to use Arial font
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(family='Arial', size=12, color='black')

fig.update_layout(
    showlegend=False,
    height=800,
    width=1500
)

# Show the plot
st.plotly_chart(fig)




###########################
########### 1C ############
st.write("<br>" * 2, unsafe_allow_html=True)
st.subheader('1C. Predicting Quality Values')

df = pd.concat([data1, data2], ignore_index=True)

Wine_filter = st.selectbox('Select Wine', options=list(df['wine'].unique()), index=0)
filtered_df = df[df['wine'] == Wine_filter]

#st.write(desc_df)

# Red Wine
df1 = data1.drop(['wine type'], axis = 1)
color_sequence = [color_map[quality] for quality in sorted(df['quality'].unique())]

fig1 = px.parallel_coordinates(df1, color="quality", labels={
        'fixed acidity': 'Fixed Acidity', 'volatile acidity': 'Volatile Acidity', 'citric acid': 'Citric Acid',
        'residual sugar': 'Residual Sugar', 'chlorides': 'Chlorides', 'free sulfur dioxide': 'Free Sulfur Dioxide',
        'total sulfur dioxide': 'Total Sulfur Dioxide', 'density': 'Density', 'pH': 'pH', 'sulphates': 'Sulphates',
        'alcohol': 'Alcohol', 'quality': 'Quality'},
        color_continuous_scale=color_sequence, template = 'plotly_white', height = 700)

fig1.update_traces(unselected=dict(line=dict(color='white')))

fig1.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="black"
    ),
    margin=dict(l=100)
)

for dim in fig1.data[0]['dimensions']:
    if dim['label'] == 'Quality':
        dim['tickvals'] = list(range(int(df1['quality'].min()), int(df1['quality'].max()) + 1))
        dim['ticktext'] = [str(x) for x in dim['tickvals']]

num_axes = len(df1.columns) - 2  # Exclude the color dimension 'quality'
positions = [(i / num_axes) for i in range(num_axes + 1)]

annotations = []
labels = ['Alcohol', 'Citric Acid', 'Fixed Acidity', 'Volatile acidity', 'pH', 'Residual Sugar', 'Total Sulfur Dioxide', 'Free Sulfur Dioxide', 'Sulphates', 'Chlorides', 'Density', 'Quality']

for idx, label in enumerate(labels):
    # Bottom annotations
    annotations.append(dict(
        x=positions[idx],
        y=-0.075,
        xref='paper',
        yref='paper',
        text=label,
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor='center'  # Center align the text
    ))

fig1.update_layout(annotations=annotations)

# White Wine

df2 = data2.drop(['wine type'], axis = 1)
fig2 = px.parallel_coordinates(df2, color="quality", labels={
        'fixed acidity': 'Fixed Acidity', 'volatile acidity': 'Volatile Acidity', 'citric acid': 'Citric Acid',
        'residual sugar': 'Residual Sugar', 'chlorides': 'Chlorides', 'free sulfur dioxide': 'Free Sulfur Dioxide',
        'total sulfur dioxide': 'Total Sulfur Dioxide', 'density': 'Density', 'pH': 'pH', 'sulphates': 'Sulphates',
        'alcohol': 'Alcohol', 'quality': 'Quality'},
        color_continuous_scale=color_sequence, template = 'plotly_dark', height = 700)

fig2.update_traces(unselected=dict(line=dict(color='white')))

fig2.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="black"
    ),
    margin=dict(l=100)
)

for dim in fig2.data[0]['dimensions']:
    if dim['label'] == 'Quality':
        dim['tickvals'] = list(range(int(df2['quality'].min()), int(df2['quality'].max()) + 1))
        dim['ticktext'] = [str(x) for x in dim['tickvals']]


num_axes = len(df2.columns) - 2  # Exclude the color dimension 'quality'
positions = [(i / num_axes) for i in range(num_axes + 1)]

annotations = []
labels = ['Alcohol', 'Citric Acid', 'Fixed Acidity', 'Volatile acidity', 'pH', 'Residual Sugar', 'Total Sulfur Dioxide', 'Free Sulfur Dioxide', 'Sulphates', 'Chlorides', 'Density', 'Quality']

for idx, label in enumerate(labels):
    # Bottom annotations
    annotations.append(dict(
        x=positions[idx],
        y=-0.075,
        xref='paper',
        yref='paper',
        text=label,
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor='center'  # Center align the text
    ))

fig2.update_layout(annotations=annotations)
fig2.update_layout(showlegend=False)


df1 = df1.drop(['wine', 'quality'], axis = 1)
df2 = df2.drop(['wine', 'quality'], axis = 1)

def title_case_except_ph(column_name):
    if column_name.lower() == 'ph':
        return column_name  # Leave 'pH' unchanged
    else:
        return column_name.title()  # Apply title case to other column names

# Apply the function to column titles
df1.columns = df1.columns.map(title_case_except_ph)
df2.columns = df2.columns.map(title_case_except_ph)


if Wine_filter == 'Red':
    st.plotly_chart(fig1)

    def display_random_rows():
        random_rows = df1.sample(1)
        html_table = random_rows.to_html(index=False, classes="center-table")
        st.markdown(f'<div class="center-table">{html_table}</div>', unsafe_allow_html=True)

    if st.button('Generate Random Wine Data'):
        display_random_rows()

elif Wine_filter == 'White':
    st.plotly_chart(fig2)
    
    def display_random_rows():
        random_rows = df2.sample(1)
        html_table = random_rows.to_html(index=False, classes="center-table")
        st.markdown(f'<div class="center-table">{html_table}</div>', unsafe_allow_html=True)
        
    if st.button('Generate Random Wine Data'):
        display_random_rows()

