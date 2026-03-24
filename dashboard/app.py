import json
import pickle
from pathlib import Path

import geopandas as gpd
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from shapely.geometry import box, shape

lat_center = 49.4330
lon_center = 11.0767

class_cols = [
    "built_up",
    "vegetation",
    "water",
]


def load_data_from_csv(csv_path="data.csv"):
    # 1. Load CSV data at full resolution (no aggregation)
    df = pd.read_csv(csv_path)

    # 2. Parse geometry from GeoJSON strings
    df["geometry"] = pd.Series(
        [shape(json.loads(x)) for x in df[".geo"]],
        index=df.index,
        dtype="object",
    )

    # 3. Build GeoDataFrame in projected CRS (meters)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32632")

    # 4. Create unique IDs and centroids for rendering (before CRS conversion for accuracy)
    gdf["grid_id"] = gdf.index
    centroids = gdf.geometry.centroid
    gdf = gdf.to_crs("EPSG:4326")
    centroids = centroids.to_crs("EPSG:4326")
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

    # Add vegetation target label
    gdf["vegetation"] = gdf[["tree_cover", "cropland", "grassland"]].sum(axis=1)

    # Add engineered features
    gdf["NDVI"] = (gdf["B8"] - gdf["B4"]) / (gdf["B8"] + gdf["B4"] + 1e-8)
    gdf["EVI2"] = 2.5 * (
        (gdf["B8"] - gdf["B4"]) / (gdf["B8"] + 2.4 * gdf["B4"] + 1 + 1e-8)
    )
    gdf["SAVI"] = ((gdf["B8"] - gdf["B4"]) * 1.5) / (gdf["B8"] + gdf["B4"] + 0.5 + 1e-8)
    gdf["NDBI"] = (gdf["B11"] - gdf["B8"]) / (gdf["B11"] + gdf["B8"] + 1e-8)
    gdf["NDWI"] = (gdf["B3"] - gdf["B8"]) / (gdf["B3"] + gdf["B8"] + 1e-8)
    gdf["MNDWI"] = (gdf["B3"] - gdf["B11"]) / (gdf["B3"] + gdf["B11"] + 1e-8)

    return gdf


def map_class_to_string(cls: int):
    try:
        return str.join(" ", class_cols[cls].lower().split("_")).title()
    except IndexError:
        return "unclassified"


def assign_group_dominant_class(
    df: pd.DataFrame,
    class_columns: list[str],
    group_columns: tuple[str, str] = ("grid_x", "grid_y"),
) -> pd.DataFrame:
    """Assign one dominant class per grid group based on summed class scores."""
    missing_cols = [
        col for col in [*group_columns, *class_columns] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for dominant class assignment: {missing_cols}"
        )

    grouped_scores = df.groupby(list(group_columns), dropna=False)[class_columns].sum()
    print(f"{len(grouped_scores)=} {len(df)=}")
    dominant_idx = np.argmax(grouped_scores.to_numpy(), axis=1)

    dominant_per_group = grouped_scores.reset_index()[list(group_columns)]
    dominant_per_group["Dominant Class"] = [
        map_class_to_string(idx) for idx in dominant_idx
    ]

    df_out = df.drop(columns=["Dominant Class"], errors="ignore").merge(
        dominant_per_group, on=list(group_columns), how="left"
    )
    return df_out


def load_prediction_model(model_path="artifacts/XGBoost_delta.pkl"):
    model_file = Path(model_path)
    if not model_file.exists():
        return None

    with model_file.open("rb") as f:
        return pickle.load(f)


prediction_model = load_prediction_model()

try:
    gdf = load_data_from_csv("data_3x3/delta_table_2021_3x3.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback

    traceback.print_exc()
    gdf = None


color_map = {
    "Tree Cover": "#006400",
    "Vegetation": "#006400",
    "Shrubland": "#ffbb22",
    "Grassland": "#ffff4c",
    "Cropland": "#f096ff",
    "Built Up": "#fa0000",
    "Bare / Sparse veg.": "#b4b4b4",
    "Snow and Ice": "#f0f0f0",
    "Water": "#0064ff",
    "Permanent Water": "#0064ff",
    "Herbaceous wetland": "#0096a0",
    "Mangroves": "#00cf75",
    "Moss and Lichen": "#fae6a0",
    "unclassified": "#2c3e50",
}


def update_dashboard(start_year, time_delta, map_type, render_mode, grid_cell_size):
    """Build two maps: selected year (delta=0) and future (delta=time_delta)."""
    if gdf is None:
        return None, None

    start_year = int(start_year)
    time_delta = int(time_delta)
    grid_cell_size = int(grid_cell_size)
    np.random.seed(start_year)

    # Select the rows from the selected data
    # This is a bit weird since all the data is coming from a CSV oriented
    # around 2021, so delta of 0 is 2021, 1 is 2020, etc.
    delta_selection = 2021 - start_year

    base_df = gdf[gdf["delta_years"].astype(int) == delta_selection].copy()

    # Calculate grid coordinates once for all aggregations
    base_df["grid_x"] = (
        np.floor(base_df["x"] / grid_cell_size) * grid_cell_size
    ).astype(int)
    base_df["grid_y"] = (
        np.floor(base_df["y"] / grid_cell_size) * grid_cell_size
    ).astype(int)

    # Build stable grid metadata once; geometry/GeoJSON is created lazily for polygons.
    grid_metadata = (
        base_df[["delta_years", "grid_x", "grid_y"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid_metadata["grid_id"] = np.arange(len(grid_metadata), dtype=int)
    plotly_geojson = None

    def get_plotly_geojson():
        nonlocal plotly_geojson
        if plotly_geojson is None:
            grid_with_geometry = grid_metadata.copy()
            grid_with_geometry["geometry"] = grid_with_geometry.apply(
                lambda row: box(
                    row["grid_x"],
                    row["grid_y"],
                    row["grid_x"] + grid_cell_size,
                    row["grid_y"] + grid_cell_size,
                ),
                axis=1,
            )
            display_grid_gdf = gpd.GeoDataFrame(
                grid_with_geometry, geometry="geometry", crs="EPSG:32632"
            )
            display_grid_gdf = display_grid_gdf.to_crs("EPSG:4326")
            plotly_geojson = json.loads(display_grid_gdf.to_json())
        return plotly_geojson

    feature_names = None
    if prediction_model is not None:
        feature_names = None
        if hasattr(prediction_model, "feature_names_in_"):
            feature_names = prediction_model.feature_names_in_.tolist()
        elif (
            hasattr(prediction_model, "estimators_")
            and len(prediction_model.estimators_) > 0
            and hasattr(prediction_model.estimators_[0], "feature_names_in_")
        ):
            feature_names = prediction_model.estimators_[0].feature_names_in_.tolist()

    def build_predicted_df(prediction_delta: int) -> pd.DataFrame:
        df_out = base_df.copy()
        df_out["delta_years"] = prediction_delta

        if prediction_model is not None and feature_names is not None:
            missing_features = [
                name for name in feature_names if name not in df_out.columns
            ]
            print(f"delta: {prediction_delta} ; feat: {missing_features}")
            if not missing_features:
                predicted_targets = prediction_model.predict(df_out[feature_names])
                predicted_targets_df = pd.DataFrame(
                    predicted_targets, index=df_out.index, columns=class_cols
                )
                df_out[class_cols] = predicted_targets_df[class_cols]
            else:
                print(f"Skipping Prediction: {prediction_model=}; {feature_names=}")
        else:
            print(f"Skipping Prediction: {prediction_model=}; {feature_names=}")

        df_out = assign_group_dominant_class(df_out, class_cols)
        df_out["Confidence"] = np.random.choice(
            ["High", "Medium", "Low"], size=len(df_out), p=[0.6, 0.3, 0.1]
        )
        return df_out

    def build_map(df_in: pd.DataFrame):
        # Aggregate per grid cell using pre-computed grid coordinates
        agg_dict = {
            "Dominant Class": "first",
            "built_up": "mean",
            "vegetation": "mean",
            "water": "mean",
            "lat": "mean",  # Average lat/lon for representative point
            "lon": "mean",
        }
        grid_df = df_in.groupby(["grid_x", "grid_y"], as_index=False).agg(agg_dict)
        grid_df = grid_df.merge(
            grid_metadata[["grid_x", "grid_y", "grid_id"]],
            on=["grid_x", "grid_y"],
            how="left",
            validate="one_to_one",
        )

        if render_mode == "points":
            # Scatter map with aggregated points at grid cell resolution
            fig = px.scatter_map(
                grid_df,
                lat="lat",
                lon="lon",
                color="Dominant Class",
                color_discrete_map=color_map,
                hover_name="grid_id",
                hover_data={
                    "grid_id": False,
                    "Dominant Class": True,
                    "built_up": True,
                    "vegetation": True,
                    "water": True,
                    "lat": False,
                    "lon": False,
                },
                zoom=11,
                center={"lat": lat_center, "lon": lon_center},
                map_style=map_type,
            )
        else:
            # Choropleth map with polygon grid cells
            fig = px.choropleth_map(
                grid_df,
                geojson=get_plotly_geojson(),
                locations="grid_id",
                featureidkey="properties.grid_id",
                color="Dominant Class",
                color_discrete_map=color_map,
                hover_name="grid_id",
                hover_data={
                    "grid_id": False,
                    "Dominant Class": True,
                    "built_up": True,
                    "vegetation": True,
                    "water": True,
                },
                zoom=11,
                center={"lat": lat_center, "lon": lon_center},
                map_style=map_type,
            )
        fig.update_layout(
            coloraxis_showscale=False,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="Black",
                borderwidth=1,
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            modebar=dict(
                orientation="v",
                bgcolor="rgba(255, 255, 255, 0.9)",
                color="black",
                activecolor="#0078A8",
            ),
            height=700,
        )
        if render_mode == "points":
            fig.update_traces(
                marker=dict(size=7, opacity=0.4),
                selected=dict(marker=dict(opacity=0.8)),
                unselected=dict(marker=dict(opacity=0.1)),
            )
        else:
            fig.update_traces(
                marker=dict(opacity=0.3),
                selected=dict(marker=dict(opacity=0.8)),
                unselected=dict(marker=dict(opacity=0.0)),
            )
        return fig

    if delta_selection == 0:
        predicted_dfs = [
            assign_group_dominant_class(base_df.copy(), class_cols),
            build_predicted_df(prediction_delta=time_delta),
        ]
    else:
        predicted_dfs = [
            build_predicted_df(prediction_delta=d) for d in [0, time_delta]
        ]
    selected_fig, future_fig = [build_map(df_item) for df_item in predicted_dfs]

    # # 3. Mock the Evaluation Metrics
    # acc = f"{np.random.uniform(85, 93):.1f}%"
    # fcr = f"{np.random.uniform(4, 9):.1f}%"
    # stability = f"{np.random.uniform(92, 98):.1f}%"

    # metrics_display = f"""
    # * **Baseline Accuracy:** {acc}
    # * **False Change Rate:** {fcr}
    # * **Stability (Unchanged):** {stability}
    # """

    # return fig, metrics_display
    selected_pred_df, future_pred_df = predicted_dfs
    numeric_diff_mask = (
        ~np.isclose(
            selected_pred_df[class_cols].to_numpy(),
            future_pred_df[class_cols].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )
    ).any(axis=1)
    dominant_diff_mask = (
        selected_pred_df["Dominant Class"].to_numpy()
        != future_pred_df["Dominant Class"].to_numpy()
    )
    diff_mask = numeric_diff_mask | dominant_diff_mask

    if diff_mask.any():
        differing_rows = pd.concat(
            [
                selected_pred_df.loc[
                    diff_mask, ["grid_id", "Dominant Class", *class_cols]
                ].add_suffix("_selected"),
                future_pred_df.loc[
                    diff_mask, ["grid_id", "Dominant Class", *class_cols]
                ].add_suffix("_future"),
            ],
            axis=1,
        )
        print(f"Differing prediction rows: {len(differing_rows)}")
        print(differing_rows.head(20).to_string(index=False))
    else:
        print("No differing rows between selected and future predictions.")

    return selected_fig, future_fig


# Gradio UI
with gr.Blocks(fill_height=True) as app:
    gr.Markdown("# 🏙️ Nuremberg Future Land-Cover Prediction")
    # gr.Markdown("# 🏙️ Nuremberg Urban Dynamics Dashboard")
    # gr.Markdown("## Future Land-Cover Prediction")

    with gr.Row():
        map_type_radio = gr.Radio(
            choices=[
                ("Street", "carto-voyager"),
                ("Satellite", "satellite-streets"),
            ],
            value="carto-voyager",
            label="Map View",
        )
        render_mode_radio = gr.Radio(
            choices=[
                ("Fast (Points)", "points"),
                ("Detailed (Polygons)", "polygons"),
            ],
            value="points",
            label="Render Mode",
        )
        grid_cell_size_dropdown = gr.Dropdown(
            choices=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
            value=100,
            label="Grid Cell Size (m)",
        )
        start_year_dropdown = gr.Dropdown(
            choices=[2016, 2017, 2018, 2019, 2020, 2021],
            value=2021,
            label="Start Year Selection",
        )
        time_delta_drop_down = gr.Dropdown(
            choices=[i for i in range(0, 5)],
            value=0,
            label="Future Time (Years)",
        )
        submit_button = gr.Button("Submit", variant="primary")

    # TODO: Remove if unused
    # with gr.Row():
    #     clear_selection_button = gr.Button("Reset Map")

    # TODO: Move to a new row?
    # gr.Markdown("### 📊 Performance Metrics")
    # metrics_box = gr.Markdown(value="*Loading metrics...*")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Selected Year Land-Cover")
            map_output_left = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("### Future Prediction")
            map_output_right = gr.Plot(show_label=False)

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
        "inputs": [
            start_year_dropdown,
            time_delta_drop_down,
            map_type_radio,
            render_mode_radio,
            grid_cell_size_dropdown,
        ],
        "outputs": [map_output_left, map_output_right],
    }
    submit_button.click(**update_args)
    # clear_selection_button.click(**update_args)
    # app.load(**update_args)

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
    app.launch()
