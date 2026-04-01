"""Dashboard"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import RealEstateModel

st.set_page_config(page_title="Real Estate", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('data/properties.csv')

df = load_data()

st.title("🏠 Real Estate Price Analytics")
col1, col2, col3 = st.columns(3)
with col1: st.metric("Total Properties", len(df))
with col2: st.metric("Avg Price", f"${df['price'].mean():,.0f}")
with col3: st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(df, x='sqft', y='price', color='neighborhood', title='Price vs Size')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(df, x='neighborhood', y='price', title='Price by Neighborhood',color='neighborhood')
    st.plotly_chart(fig, use_container_width=True)

fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='price', size='lot_size', 
                     hover_data=['bedrooms', 'price', 'sqft'], title='Property Map by Location',
                     size_max=30)
st.plotly_chart(fig, use_container_width=True)
