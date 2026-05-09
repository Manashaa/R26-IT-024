import os
import pandas as pd
import folium

PREDICTION_FILE = "outputs/future_predictions.csv"
OUTPUT_MAP = "outputs/prediction_map.html"

df = pd.read_csv(PREDICTION_FILE)

selected_year = df["year"].min()
df = df[df["year"] == selected_year].copy()

map_center = [
    df["latitude"].mean(),
    df["longitude"].mean()
]

m = folium.Map(
    location=map_center,
    zoom_start=10,
    tiles="OpenStreetMap"
)

def get_color(trend):
    if trend == "increase":
        return "green"
    elif trend == "decrease":
        return "red"
    else:
        return "orange"

for _, row in df.iterrows():

    popup_text = f"""
    <b>Polygon ID:</b> {row['polygon_id']}<br>
    <b>Year:</b> {row['year']}<br>
    <b>Predicted Trend:</b> {row['predicted_trend']}<br>
    <b>Confidence:</b> {round(row['prediction_confidence'] * 100, 2)}%<br><br>
    <b>Factor Source:</b> {row['factor_source']}<br><br>
    <b>Forecasted NDVI:</b> {round(row['ndvi'], 4)}<br>
    <b>Forecasted NDWI:</b> {round(row['ndwi_mean'], 4)}<br>
    <b>Forecasted Rainfall:</b> {round(row['rainfall_mm'], 2)} mm<br>
    <b>Forecasted Temperature:</b> {round(row['temperature_celsius'], 2)} °C<br>
    <b>Elevation:</b> {round(row['elevation_mean'], 2)} m<br>
    <b>Coastal Distance:</b> {round(row['coastal_distance_km'], 2)} km
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        popup=folium.Popup(popup_text, max_width=350),
        color=get_color(row["predicted_trend"]),
        fill=True,
        fill_color=get_color(row["predicted_trend"]),
        fill_opacity=0.8
    ).add_to(m)

legend_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 180px;
height: 120px;
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
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

os.makedirs("outputs", exist_ok=True)

m.save(OUTPUT_MAP)

print("Prediction map created successfully.")
print("Year displayed:", selected_year)
print("Saved:", OUTPUT_MAP)