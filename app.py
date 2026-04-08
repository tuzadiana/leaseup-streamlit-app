import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diana Streamlit App", layout="wide")

st.title("Diana’s Streamlit App")
st.write("This is running in the cloud.")

col1, col2, col3 = st.columns(3)
col1.metric("Sites Scored", "15,480")
col2.metric("Avg Probability", "31.1%")
col3.metric("Priority Sites", "3,455")

st.dataframe(pd.DataFrame({
    "Site": ["A", "B", "C"],
    "Score": [91.2, 88.4, 93.1]
}))
