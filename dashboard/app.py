import gradio as gr
import plotly.express as px
import pandas as pd
import numpy as np
import h3

# --- Real Hexagon Data Generation (Nuremberg) ---
lat_center = 49.4521
lon_center = 11.0767
resolution = 10  # Resolution 8 is perfect for neighborhood-level city data

# H3 Version 4.x
center_hex = h3.latlng_to_cell(lat_center, lon_center, resolution)
hexes = list(h3.grid_disk(center_hex, 40))  # Radius of 7 hexes


# We manually flip the (lat, lon) tuples to (lon, lat) for GeoJSON
def get_boundary(h):
    return [coord[::-1] for coord in h3.cell_to_boundary(h)]


# Build the GeoJSON dictionary required by Plotly to draw polygons
geojson = {"type": "FeatureCollection", "features": []}
for h in hexes:
    coords = list(get_boundary(h))
    coords.append(coords[0])  # GeoJSON requires the polygon loop to be closed

    feature = {
        "type": "Feature",
        "id": h,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"hex_id": h},
    }
    geojson["features"].append(feature)

base_df = pd.DataFrame({"Hexagon_ID": hexes})


def update_dashboard(start_year, end_year):
    """Generates dummy ML predictions and returns a map + metrics."""
    np.random.seed(int(start_year))

    # Mock the Machine Learning Output
    df = base_df.copy()
    df["Predicted Change (%)"] = np.random.normal(loc=0, scale=15, size=len(df))
    df["Dominant Class"] = np.random.choice(
        ["Built-up", "Tree Cover", "Grassland", "Cropland"], size=len(df)
    )
    df["Confidence"] = np.random.choice(
        ["High", "Medium", "Low"], size=len(df), p=[0.6, 0.3, 0.1]
    )

    # Build the Plotly Map
    fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="Hexagon_ID",
        color="Predicted Change (%)",
        color_continuous_scale="RdYlGn_r",
        hover_name="Hexagon_ID",
        hover_data={"Hexagon_ID": False, "Dominant Class": True, "Confidence": True},
        zoom=11,
        center={"lat": lat_center, "lon": lon_center},
        map_style="carto-voyager",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        # Attempt to replicate the Leaflet Toolbar
        modebar=dict(
            orientation="v",  # Make it vertical
            bgcolor="rgba(255, 255, 255, 0.9)",  # Give it a solid white background
            color="black",  # Dark icons
            activecolor="#0078A8",  # Leaflet's classic active blue color
        ),
        # Add drawing tools to the modebar
        # TODO: Remove if not used
        # modebar_add=[
        #     "drawline",
        #     "drawclosedpath",
        #     "drawcircle",
        #     "drawrect",
        #     "eraseshape",
        # ],
    )

    # Only show hexes that are selected.
    fig.update_traces(
        # Make everything nearly invisible by default
        marker=dict(opacity=0.1),
        # Make hexes visible when lassoed/boxed
        selected=dict(marker=dict(opacity=0.8)),
        # Ensure unselected hexes stay invisible
        unselected=dict(marker=dict(opacity=0.0)),
    )

    # 3. Mock the Evaluation Metrics
    acc = f"{np.random.uniform(85, 93):.1f}%"
    fcr = f"{np.random.uniform(4, 9):.1f}%"
    stability = f"{np.random.uniform(92, 98):.1f}%"

    metrics_display = f"""
    * **Baseline Accuracy:** {acc}
    * **False Change Rate:** {fcr}
    * **Stability (Unchanged):** {stability}
    """

    return fig, metrics_display


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🏙️ Nuremberg Urban Dynamics Dashboard")
    gr.Markdown("### Predicting and Analyzing Land-Cover Changes (Prototype)")

    with gr.Row():
        with gr.Column(scale=1):
            start_year_dropdown = gr.Dropdown(
                choices=["2016", "2017", "2018", "2019"],
                value="2019",
                label="Start Year Selection",
            )
            end_year_dropdown = gr.Dropdown(
                choices=[str(i) for i in range(2016, 2026)],
                value="2020",
                label="End Year Selection",
            )
            clear_selection_button = gr.Button("Reset Map")

            gr.Markdown("### 📊 Performance Metrics")
            metrics_box = gr.Markdown(value="*Loading metrics...*")

        with gr.Column(scale=3):
            map_output = gr.Plot(show_label=False)

    with gr.Row():
        gr.Markdown("""
        ---
        ### ⚠️ Limitations & Disclosures
        * **DO NOT** use for zoning or building permits.
        * Prediction model is aggregated at the grid level.
        * Labels contain inherent noise and historical errors.
        * Model relies on historical trends and cannot predict external shocks or abrupt policy shifts.
        """)

    start_year_dropdown.change(
        fn=update_dashboard,
        inputs=[start_year_dropdown, end_year_dropdown],
        outputs=[map_output, metrics_box],
    )
    end_year_dropdown.change(
        fn=update_dashboard,
        inputs=[start_year_dropdown, end_year_dropdown],
        outputs=[map_output, metrics_box],
    )
    clear_selection_button.click(
        fn=update_dashboard,
        inputs=[start_year_dropdown, end_year_dropdown],
        outputs=[map_output, metrics_box],
    )
    app.load(
        fn=update_dashboard,
        inputs=[start_year_dropdown, end_year_dropdown],
        outputs=[map_output, metrics_box],
    )

# TODO: remove if unused
# css = """
# /* Force the Plotly modebar to the top-left corner */
# .modebar-container {
#     left: 10px !important;
#     right: auto !important;
#     top: 10px !important;
# }
# """

if __name__ == "__main__":
    app.launch(theme=gr.themes.Monochrome())
