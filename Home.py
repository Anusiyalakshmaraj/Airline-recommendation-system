# app.py
import streamlit as st

st.set_page_config(page_title="Airline - Home", layout="wide")
st.title("✈️ Airline Recommendation System")
st.markdown("""
            Welcome to FlyRight! 

The Airline Recommendation System (ARS) simplifies booking by clearly showing you the best flight options. We go beyond just price, using a smart Utility Score that balances two key factors price and sentimental anlysis.
            
Tell us where you want to go, and let us recommend the flight that offers you the best value for your money and your experience. Safe travels!
""")
