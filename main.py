import streamlit as st

# --- PAGE SETUP ---

project_1_page = st.Page(
    "views/1stPage.py",
    title="1stPage",
    icon=":material/bar_chart:",
    default=True,
)
project_2_page = st.Page(
    "views/2ndPage.py",
    title="2ndPage",
    icon=":material/smart_toy:",
)

pg = st.navigation(pages=[project_1_page, project_2_page])

# --- SHARED ON ALL PAGES ---
st.logo("logo.png")

# --- RUN NAVIGATION ---
pg.run()