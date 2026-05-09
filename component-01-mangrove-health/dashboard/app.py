import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Mangrove Trend Prediction Dashboard",
    layout="wide"
)

HISTORICAL_FILE = "outputs/predictions.csv"
FUTURE_FILE = "outputs/future_predictions.csv"
FEATURE_IMPORTANCE_FILE = "outputs/feature_importance.csv"
CONFUSION_MATRIX_FILE = "outputs/confusion_matrix.png"

historical_df = pd.read_csv(HISTORICAL_FILE)
future_df = pd.read_csv(FUTURE_FILE)
feature_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

st.title("AI-Based Mangrove Trend Prediction Dashboard")
st.caption(
    "Polygon-level mangrove trend prediction using remote sensing, GIS, and machine learning."
)

def get_color(trend):
    if trend == "increase":
        return "green"
    elif trend == "decrease":
        return "red"
    else:
        return "orange"

with st.sidebar:
    st.header("Navigation")

    view_type = st.radio(
        "Select View",
        ["Future Trend Prediction", "Historical Data View"]
    )

    if view_type == "Future Trend Prediction":
        working_df = pd.concat([
            historical_df[historical_df["year"].isin([2024, 2025])],
            future_df
        ], ignore_index=True)

        available_years = sorted(working_df["year"].unique())

        selected_year = st.selectbox(
            "Select Prediction Year",
            available_years,
            index=0
        )

    else:
        working_df = historical_df[historical_df["year"] <= 2023].copy()

        available_years = sorted(working_df["year"].unique())

        selected_year = st.selectbox(
            "Select Historical Year",
            available_years,
            index=len(available_years) - 1
        )

    year_df = working_df[working_df["year"] == selected_year].copy()

    polygon_list = sorted(year_df["polygon_id"].unique())

    selected_polygon = st.selectbox(
        "Select Polygon ID",
        polygon_list
    )

    trend_filter = st.multiselect(
        "Filter Trend",
        ["increase", "decrease", "stable"],
        default=["increase", "decrease", "stable"]
    )

    st.markdown("---")
    st.subheader("Model Information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Prediction Type:** Polygon-level trend classification")
    st.write("**Output Classes:** Increase / Decrease / Stable")
    st.write("**Explainability:** Feature importance")

filtered_df = year_df[year_df["predicted_trend"].isin(trend_filter)].copy()

selected_row = year_df[
    year_df["polygon_id"] == selected_polygon
].iloc[0]

m1, m2, m3, m4 = st.columns(4)

m1.metric("Current View", view_type)
m2.metric("Selected Year", selected_year)
m3.metric("Total Polygons", year_df["polygon_id"].nunique())
m4.metric("Model Accuracy", "84.54%")

increase_count = len(year_df[year_df["predicted_trend"] == "increase"])
decrease_count = len(year_df[year_df["predicted_trend"] == "decrease"])
stable_count = len(year_df[year_df["predicted_trend"] == "stable"])

c1, c2, c3 = st.columns(3)

c1.success(f"Increase Predictions: {increase_count}")
c2.error(f"Decrease Predictions: {decrease_count}")
c3.warning(f"Stable Predictions: {stable_count}")

st.markdown("---")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Selected Polygon Result")

    trend = selected_row["predicted_trend"]
    confidence = selected_row["prediction_confidence"] * 100

    st.metric("Polygon ID", selected_polygon)
    st.metric("Predicted Trend", trend.upper())
    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.markdown("### Environmental Factor Values")

    if selected_row["factor_source"] == "Observed environmental data":
        st.write("**Factor Source:** Observed environmental data")
        factor_label = ""
    else:
        st.write("**Factor Source:** Forecasted environmental factors")
        st.info(
            "These values are forecasted using historical temporal trend analysis. "
            "They are not directly observed future values."
        )
        factor_label = "Forecasted "

    st.write(f"**{factor_label}NDVI:** {selected_row['ndvi']:.4f}")
    st.write(f"**Previous Year NDVI:** {selected_row['ndvi_previous_year']:.4f}")
    st.write(f"**{factor_label}NDWI:** {selected_row['ndwi_mean']:.4f}")
    st.write(f"**Previous Year NDWI:** {selected_row['ndwi_previous_year']:.4f}")
    st.write(f"**{factor_label}Rainfall:** {selected_row['rainfall_mm']:.2f} mm")
    st.write(f"**{factor_label}Temperature:** {selected_row['temperature_celsius']:.2f} °C")
    st.write(f"**Elevation:** {selected_row['elevation_mean']:.2f} m")
    st.write(f"**Coastal Distance:** {selected_row['coastal_distance_km']:.2f} km")

    st.markdown("### Simple Explainability")

    top_features = feature_df.head(5)

    st.write("The prediction is mainly influenced by:")

    for _, row in top_features.iterrows():
        st.write(f"- **{row['Feature']}**")


with right_col:
    st.subheader(f"{view_type} Map - {selected_year}")

    if len(filtered_df) == 0:
        st.warning("No records available for selected filters.")
    else:
        map_center = [
            filtered_df["latitude"].mean(),
            filtered_df["longitude"].mean()
        ]

        m = folium.Map(
            location=map_center,
            zoom_start=10,
            tiles="OpenStreetMap"
        )

        for _, row in filtered_df.iterrows():

            if row["factor_source"] == "Observed environmental data":
                factor_text = "Observed environmental data"
                ndvi_label = "NDVI"
                ndwi_label = "NDWI"
                rainfall_label = "Rainfall"
                temp_label = "Temperature"
            else:
                factor_text = "Forecasted environmental factors using temporal trend analysis"
                ndvi_label = "Forecasted NDVI"
                ndwi_label = "Forecasted NDWI"
                rainfall_label = "Forecasted Rainfall"
                temp_label = "Forecasted Temperature"

            popup_text = f"""
            <b>Polygon ID:</b> {row['polygon_id']}<br>
            <b>Year:</b> {row['year']}<br>
            <b>Predicted Trend:</b> {row['predicted_trend']}<br>
            <b>Confidence:</b> {round(row['prediction_confidence'] * 100, 2)}%<br><br>
            <b>Factor Source:</b> {factor_text}<br><br>
            <b>{ndvi_label}:</b> {round(row['ndvi'], 4)}<br>
            <b>{ndwi_label}:</b> {round(row['ndwi_mean'], 4)}<br>
            <b>{rainfall_label}:</b> {round(row['rainfall_mm'], 2)} mm<br>
            <b>{temp_label}:</b> {round(row['temperature_celsius'], 2)} °C<br>
            <b>Elevation:</b> {round(row['elevation_mean'], 2)} m<br>
            <b>Coastal Distance:</b> {round(row['coastal_distance_km'], 2)} km
            """

            is_selected = row["polygon_id"] == selected_polygon

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=12 if is_selected else 5,
                popup=folium.Popup(popup_text, max_width=350),
                color="blue" if is_selected else get_color(row["predicted_trend"]),
                fill=True,
                fill_color=get_color(row["predicted_trend"]),
                fill_opacity=0.9 if is_selected else 0.7,
                weight=4 if is_selected else 1
            ).add_to(m)

        legend_html = """
        <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 190px;
        height: 130px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
        color: black;
        ">
        <b>Prediction Legend</b><br>
        <span style="color:green;">●</span> Increase<br>
        <span style="color:red;">●</span> Decrease<br>
        <span style="color:orange;">●</span> Stable<br>
        <span style="color:blue;">●</span> Selected Polygon
        </div>
        """

        m.get_root().html.add_child(folium.Element(legend_html))

        html(m._repr_html_(), height=600)

st.markdown("---")

st.subheader(f"Selected Polygon Timeline - {selected_polygon}")

timeline_df = pd.concat([historical_df, future_df], ignore_index=True)
timeline_df = timeline_df[timeline_df["polygon_id"] == selected_polygon]
timeline_df = timeline_df.sort_values("year")

timeline_columns = [
    "year",
    "predicted_trend",
    "prediction_confidence",
    "factor_source",
    "ndvi",
    "ndwi_mean",
    "rainfall_mm",
    "temperature_celsius",
    "elevation_mean",
    "coastal_distance_km"
]

st.dataframe(
    timeline_df[timeline_columns],
    use_container_width=True
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Importance")
    st.bar_chart(feature_df.set_index("Feature"))

with col2:
    st.subheader("Confusion Matrix")
    st.image(CONFUSION_MATRIX_FILE, use_container_width=True)

st.subheader("Displayed Prediction Records")

display_columns = [
    "record_id",
    "polygon_id",
    "year",
    "longitude",
    "latitude",
    "predicted_trend",
    "prediction_confidence",
    "factor_source",
    "ndvi",
    "ndwi_mean",
    "rainfall_mm",
    "temperature_celsius",
    "elevation_mean",
    "coastal_distance_km"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)