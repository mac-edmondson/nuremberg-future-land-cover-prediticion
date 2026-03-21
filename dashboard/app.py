import json

import geopandas as gpd
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from shapely.geometry import box, shape

lat_center = 49.4521
lon_center = 11.0767

class_cols = [
    "tree_cover",
    "built_up",
    "grassland",
    "cropland",
    "bare_sparse_vegetation",
    "water",
]

display_cell_size_m = 200

# 1. Load your CSV data
# Replace 'your_data.csv' and 'polygon_column' with your actual file and column names
df = pd.read_csv("data.csv")

# 2. Parse the JSON strings into Shapely geometry objects
# This reads that long string you pasted and turns it into a mathematical polygon
df["geometry"] = df[".geo"].apply(lambda x: shape(json.loads(x)))

# 3. Create a GeoDataFrame and explicitly tell it the current format is EPSG:3857 (meters)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32632")

# Aggregate 20x20m cells into 200x200m display cells.
# x and y are cell-center coordinates in EPSG:32632.
gdf["grid_x"] = (np.floor(gdf["x"] / display_cell_size_m) * display_cell_size_m).astype(
    int
)
gdf["grid_y"] = (np.floor(gdf["y"] / display_cell_size_m) * display_cell_size_m).astype(
    int
)

agg_df = (
    gdf.groupby(["grid_x", "grid_y"], as_index=False)
    .agg({**{column: "mean" for column in class_cols}, "cell_id": "count"})
    .rename(columns={"cell_id": "cell_count"})
)

agg_df["geometry"] = agg_df.apply(
    lambda row: box(
        row["grid_x"],
        row["grid_y"],
        row["grid_x"] + display_cell_size_m,
        row["grid_y"] + display_cell_size_m,
    ),
    axis=1,
)

gdf = gpd.GeoDataFrame(agg_df, geometry="geometry", crs="EPSG:32632")

# 4. CRITICAL STEP: Reproject the coordinates to EPSG:4326 (Latitude/Longitude)
gdf = gdf.to_crs("EPSG:4326")

# 5. Create a unique ID for Plotly to link the map to your data
# (If your CSV already has an ID column, you can use that instead)
gdf["Hexagon_ID"] = gdf.index

# 6. Convert the fixed geometries into the GeoJSON dictionary format Plotly expects
plotly_geojson = json.loads(gdf.geometry.to_json())


def map_class_to_string(cls: int):
    try:
        match int(cls):
            case 0:
                return "Tree Cover"
            case 1:
                return "Built-up"
            case 2:
                return "Grassland"
            case 3:
                return "Cropland"
            case 4:
                return "Bare / Sparse veg."
            case 5:
                return "Permanent Water"
            case _:
                return "unclassified"
    except ValueError:
        return "unclassified"


color_map = {
    "Tree Cover": "#006400",
    "Shrubland": "#ffbb22",
    "Grassland": "#ffff4c",
    "Cropland": "#f096ff",
    "Built-up": "#fa0000",
    "Bare / Sparse veg.": "#b4b4b4",
    "Snow and Ice": "#f0f0f0",
    "Permanent Water": "#0064ff",
    "Herbaceous wetland": "#0096a0",
    "Mangroves": "#00cf75",
    "Moss and Lichen": "#fae6a0",
    "unclassified": "#2c3e50",
}


def update_dashboard(start_year, end_year, map_type):
    """Generates dummy ML predictions and returns a map + metrics."""
    np.random.seed(int(start_year))

    # Mock the Machine Learning Output
    df = gdf.copy()
    df["Dominant Class"] = np.argmax(df[class_cols].to_numpy(), axis=1)
    df["Dominant Class"] = df["Dominant Class"].apply(map_class_to_string)
    df["Confidence"] = np.random.choice(
        ["High", "Medium", "Low"], size=len(df), p=[0.6, 0.3, 0.1]
    )

    # Build the Plotly Map
    fig = px.choropleth_map(
        df,
        geojson=plotly_geojson,
        locations="Hexagon_ID",
        color="Dominant Class",
        color_discrete_map=color_map,
        hover_name="Hexagon_ID",
        hover_data={
            "Hexagon_ID": False,
            "Dominant Class": True,
            "cell_count": True,
            "tree_cover": True,
            "built_up": True,
            "grassland": True,
            "cropland": True,
            "bare_sparse_vegetation": True,
            "water": True,
        },
        zoom=11,
        center={"lat": lat_center, "lon": lon_center},
        map_style=map_type,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        legend=dict(
            orientation="v",  # Vertical orientation
            yanchor="top",  # Anchor the top of the legend...
            y=0.98,  # ...at 98% height (near the top)
            xanchor="left",  # Anchor the left side of the legend...
            x=0.02,  # ...at 2% width (near the left edge)
            bgcolor="rgba(255, 255, 255, 0.7)",  # Add a background for readability
            bordercolor="Black",
            borderwidth=1,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        # Attempt to replicate the Leaflet Toolbar
        modebar=dict(
            orientation="v",  # Make it vertical
            bgcolor="rgba(255, 255, 255, 0.9)",  # Give it a solid white background
            color="black",  # Dark icons
            activecolor="#0078A8",  # Leaflet's classic active blue color
        ),
        height=700,  # TODO: Get this as an arg to the function if possible?
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
        marker=dict(opacity=0.2),
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
with gr.Blocks(fill_height=True) as app:
    gr.Markdown("# 🏙️ Nuremberg Urban Dynamics Dashboard")
    gr.Markdown("### Predicting and Analyzing Land-Cover Changes (Prototype)")

    with gr.Row():
        with gr.Column(scale=1):
            map_type_radio = gr.Radio(
                choices=[
                    ("Street", "carto-voyager"),
                    ("Satellite", "satellite-streets"),
                ],
                value="carto-voyager",
                label="Map View",
            )
            clear_selection_button = gr.Button("Reset Map")
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

    update_args = {
        "fn": update_dashboard,
        "inputs": [start_year_dropdown, end_year_dropdown, map_type_radio],
        "outputs": [map_output, metrics_box],
    }
    map_type_radio.change(**update_args)
    start_year_dropdown.change(**update_args)
    end_year_dropdown.change(**update_args)
    clear_selection_button.click(**update_args)
    app.load(**update_args)

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
