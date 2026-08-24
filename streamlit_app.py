import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/final_infection.py", title="Final Infection", icon="📊"),
])

pg.run()
