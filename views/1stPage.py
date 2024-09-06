import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from contact import contact_form

np.random.seed(123)

@st.dialog("Contact Me")
def show_contact_form():
    contact_form()

st.set_page_config(
    page_title="My app", # The page title, shown in the browser tab.
    page_icon=":shark:", # The page favicon.
    layout="wide", # How the page content should be laid out.
    initial_sidebar_state="auto", # How the sidebar should start out.
    menu_items={ # Configure the menu that appears on the top-right side of this app.
        "Get help": "https://github.com/LMAPcoder" # The URL this menu item should point to.
    }
)

# ---- SIDEBAR ----

with st.sidebar:
    st.header("Header: Options")

    # Slider widget
    n_points = st.slider(
        "Select a number of data points:",
        min_value=5,  # The minimum permitted value.
        max_value=100,  # The maximum permitted value.
        value=10  # The value of the slider when it first renders.
    )

    # Display slider value
    st.write(f"N of points: {n_points}")


    # Slider widget
    mean = st.slider(
        "Select a mean:",
        min_value=1, # The minimum permitted value.
        max_value=10, # The maximum permitted value.
        value=10 # The value of the slider when it first renders.
    )

    # Display slider value
    st.write(f"Mean: {mean}")

    serie = st.multiselect(
        "Filter out the series:",
        options=['A', 'B', 'C']
    )

    st.write(f"Multiselectot Value: {serie}")

    #question = st.sidebar.text_area("Input question")
    #button1 = st.sidebar.button("Clear text")

    # if button1:
    #    st.session_state.question.value = ""


# ---- MAINPAGE ----

# Set the title of the web app
st.title("Title: Interactive Streamlit App with Data :bar_chart:")

# Generate random data
data = np.random.normal(loc=mean, scale=mean*0.2, size=(n_points, 3))
data = pd.DataFrame(data, columns=['A', 'B', 'C'])
data.drop(serie, axis=1, inplace=True)
# Columns
col1, col2 = st.columns([0.7, 0.3], gap="medium")

with col1:

    st.header("Header: Line chart")

    st.markdown("Markdown: Line chart")

    # Line chart
    st.line_chart(data)


# Checkbox widget
if col2.checkbox("Show Data"):
    col2.subheader("Subheader: Table")
    col2.write(data[:5])

st.divider()

col1, col2 = st.columns(2, gap="medium")

group = data.sum()

with col1:
    st.subheader("Subheader: Pie chart on Matplotlib")

    st.pyplot(
        fig=group.plot.pie(figsize=(3, 3), title="Proportions").get_figure(),
        clear_figure=True,
        use_container_width=True
    )

with col2:
    st.subheader("Subheader: Pie chart on Plotly")

    group = group.reset_index()
    group.columns = ['Category', 'Sum']

    fig = px.pie(group, values='Sum', names='Category', title='Pie Chart Example')

    st.plotly_chart(fig)


with st.expander("Summary cards"):

    # Columns
    col1, col2, col3 = st.columns(3, gap="medium")

    totalA = data['A'].sum()
    totalB = data['B'].sum()
    totalC = data['C'].sum()

    col1.metric(
        "Sum A",
        value=f'{totalA:.1f}$',
        delta="-8"
    )

    col2.metric(
        "Sum B",
        value=f'{totalB:.1f}$',
        delta="-8"
    )

    col3.metric(
        "Sum C",
        value=f'{totalC:.1f}$',
        delta="-8"
    )

button2 = st.button("✉️ Contact Me")
if button2:
    show_contact_form()