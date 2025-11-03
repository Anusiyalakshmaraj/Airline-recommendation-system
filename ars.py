import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading (Cached) ---

DF_SENTIMENT_DATA = {
    'airline': ['vistara', 'air_india', 'spicejet', 'indigo', 'airasia', 'go_first', 'air_india_express'],
    'airline_quality_score': [0.8177, 0.7988, 0.8377, 0.8103, 0.7907, 0.7419, 0.7988] 
}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Clean_Dataset.csv")
        df.drop(columns=['Unnamed: 0', 'flight'], inplace=True, errors='ignore')
    except FileNotFoundError:
        st.error("Error: Clean_Dataset.csv not found. Please ensure it is in the same directory.")
        return None

    df['airline'] = df['airline'].str.lower().str.replace(' ', '_').str.replace('-', '_')

    df_sentiment = pd.DataFrame(DF_SENTIMENT_DATA)
    df_sentiment['airline'] = df_sentiment['airline'].str.lower().str.replace(' ', '_').str.replace('-', '_')

    df = pd.merge(df, df_sentiment, on='airline', how='left')
    df['airline_quality_score'].fillna(df.groupby('class')['airline_quality_score'].transform('mean'), inplace=True) 
    
    return df


# --- 2. Recommendation Logic ---

def recommend_flights_by_hybrid_score(df_raw, source, destination, flight_class, days_left, top_n, 
                                      price_weight, quality_weight, duration_weight):
    df = df_raw.copy()

    filtered_df = df[
        (df['source_city'] == source) &
        (df['destination_city'] == destination) &
        (df['class'] == flight_class) &
        (df['days_left'] == days_left)
    ].reset_index(drop=True)

    if filtered_df.empty:
        return pd.DataFrame()

    price_factor = (filtered_df['price'] + 1) ** price_weight
    duration_factor = (filtered_df['duration'] + 0.01) ** duration_weight
    
    filtered_df['utility_score'] = (
        filtered_df['airline_quality_score'] ** quality_weight
    ) / (
        price_factor * duration_factor
    )
    
    final_recommendations = filtered_df.sort_values(by='utility_score', ascending=False)
    
    return final_recommendations.head(top_n)


# --- 3. Streamlit App Interface ---

def main():
    st.set_page_config(page_title="Airline Recommender", layout="wide")
    st.title("✈️ Airline Recommendation System")
    st.markdown("---")

    df_full = load_data()
    
    if df_full is None:
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Search Criteria")
        cities = sorted(df_full['source_city'].unique().tolist())
        classes = sorted(df_full['class'].unique().tolist())
        days_left_range = sorted(df_full['days_left'].unique().tolist())
        
        source = st.selectbox("Source City", options=cities, index=cities.index('Delhi') if 'Delhi' in cities else 0)
        dest_options = sorted(df_full[df_full['source_city'] != source]['destination_city'].unique().tolist())
        destination = st.selectbox("Destination City", options=dest_options, index=dest_options.index('Mumbai') if 'Mumbai' in dest_options else 0)
        flight_class = st.selectbox("Flight Class", options=classes, index=0)
        days_left = st.select_slider("Days Left Until Flight", options=days_left_range, value=15)
        top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    with col2:
        st.header("Weight Preferences (Value: Higher = More Important)")
        st.markdown("Tune these sliders to reflect what matters most to the passenger.")
        
        quality_weight = st.slider("Airline Quality Score (Service & Reliability)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        price_weight = st.slider("Price (Cost Sensitivity)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        duration_weight = st.slider("Duration (Time Sensitivity)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        
        st.info(f"""
        **Current Formula:** $Utility \\propto \\frac{{Quality^{{{quality_weight}}}}}{{(Price)^{{{price_weight}}} \\times (Duration)^{{{duration_weight}}}}}$
        """)

    st.markdown("---")

    if st.button("Find Recommended Flights 🔍", use_container_width=True):
        with st.spinner("Calculating hybrid utility scores..."):
            recommendations = recommend_flights_by_hybrid_score(
                df_full, source, destination, flight_class, days_left, top_n,
                price_weight, quality_weight, duration_weight
            )

        if recommendations.empty:
            st.warning(f"No flights found for {source} to {destination} on day {days_left}.")
        else:
            st.success("Recommendations Found!")
            
            display_cols = [
                'airline', 'departure_time', 'arrival_time', 'stops', 'duration', 
                'price', 'airline_quality_score', 'utility_score'
            ]
            
            st.dataframe(
                recommendations[display_cols].style.format({
                    'price': '₹{:,.2f}', 
                    'duration': '{:.2f} hrs',
                    'airline_quality_score': '{:.4f}',
                    'utility_score': '{:.8f}'
                }),
                hide_index=True,
                use_container_width=True
            )

            # --- 💾 Download Option ---
            csv_data = recommendations[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Recommendations as CSV",
                data=csv_data,
                file_name=f"Flight_Recommendations_{source}_to_{destination}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("#### 💡 Interpretation of Results")
            st.markdown(f"- **Utility Score:** Higher = better overall value for your preferences.")
            st.markdown(f"- **Airline Quality Score:** Sentiment-derived reliability score (max 1.0).")
            st.markdown(f"- **Search:** {source} → {destination}, {flight_class}, {days_left} days left.")
            st.markdown(f"- **Weights:** Quality={quality_weight}, Price={price_weight}, Duration={duration_weight}")

            


if __name__ == '__main__':
    main()
