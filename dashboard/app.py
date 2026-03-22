import json
import pickle
from pathlib import Path

import geopandas as gpd
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from shapely.geometry import box, shape

lat_center = 49.4521
lon_center = 11.0767

class_cols = [
    "built_up",
    "tree_cover",
    "grassland",
    "cropland",
    "water",
]

display_cell_size_m = 200


def load_data_from_csv(csv_path="data.csv", cell_size_m=display_cell_size_m):
    # 1. Load CSV data
    df = pd.read_csv(csv_path)

    # 2. Parse geometry from GeoJSON strings
    df["geometry"] = df[".geo"].apply(lambda x: shape(json.loads(x)))

    # 3. Build GeoDataFrame in projected CRS (meters)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32632")

    # Aggregate 20x20m cells into larger display cells.
    gdf["grid_x"] = (np.floor(gdf["x"] / cell_size_m) * cell_size_m).astype(int)
    gdf["grid_y"] = (np.floor(gdf["y"] / cell_size_m) * cell_size_m).astype(int)

    # agg_df = (
    #     gdf.groupby(["grid_x", "grid_y"], as_index=False)
    #     .agg({**{column: "mean" for column in class_cols}, "cell_id": "count"})
    #     .rename(columns={"cell_id": "cell_count"})
    # )

    numeric_columns = gdf.select_dtypes(include=[np.number]).columns.tolist()
    agg_columns = {
        column: "mean"
        for column in numeric_columns
        if column not in {"grid_x", "grid_y"}
    }
    agg_df = gdf.groupby(["delta_years", "grid_x", "grid_y"], as_index=False).agg(
        agg_columns
    )

    agg_df["geometry"] = agg_df.apply(
        lambda row: box(
            row["grid_x"],
            row["grid_y"],
            row["grid_x"] + cell_size_m,
            row["grid_y"] + cell_size_m,
        ),
        axis=1,
    )

    gdf = gpd.GeoDataFrame(agg_df, geometry="geometry", crs="EPSG:32632")

    # 4. Reproject to latitude/longitude
    gdf = gdf.to_crs("EPSG:4326")

    # 5. Create unique IDs for Plotly linking
    gdf["Hexagon_ID"] = gdf.index

    # Add engineered features
    gdf["NDVI"] = (gdf["B8"] - gdf["B4"]) / (gdf["B8"] + gdf["B4"] + 1e-8)
    gdf["EVI2"] = 2.5 * (
        (gdf["B8"] - gdf["B4"]) / (gdf["B8"] + 2.4 * gdf["B4"] + 1 + 1e-8)
    )
    gdf["SAVI"] = ((gdf["B8"] - gdf["B4"]) * 1.5) / (gdf["B8"] + gdf["B4"] + 0.5 + 1e-8)
    gdf["NDBI"] = (gdf["B11"] - gdf["B8"]) / (gdf["B11"] + gdf["B8"] + 1e-8)
    gdf["NDWI"] = (gdf["B3"] - gdf["B8"]) / (gdf["B3"] + gdf["B8"] + 1e-8)
    gdf["MNDWI"] = (gdf["B3"] - gdf["B11"]) / (gdf["B3"] + gdf["B11"] + 1e-8)

    # 6. Convert to GeoJSON format expected by Plotly
    plotly_geojson = json.loads(gdf.geometry.to_json())
    return gdf, plotly_geojson


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


def load_prediction_model(model_path="artifacts/xgboost_multioutput.pkl"):
    model_file = Path(model_path)
    if not model_file.exists():
        return None

    with model_file.open("rb") as f:
        return pickle.load(f)


gdf, plotly_geojson = load_data_from_csv("data_3x3/delta_table_2021_3x3.csv")
prediction_model = load_prediction_model()


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


def update_dashboard(start_year, time_delta, map_type):
    """Build two maps: selected year (delta=0) and future (delta=time_delta)."""
    start_year = int(start_year)
    time_delta = int(time_delta)
    np.random.seed(start_year)

    # Select the rows from the selected data
    # This is a bit weird since all the data is coming from a CSV oriented
    # around 2021, so delta of 0 is 2021, 1 is 2020, etc.
    delta_selection = 2021 - start_year

    base_df = gdf[gdf["delta_years"].astype(int) == delta_selection].copy()

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

        df_out["Dominant Class"] = np.argmax(df_out[class_cols].to_numpy(), axis=1)
        df_out["Dominant Class"] = df_out["Dominant Class"].apply(map_class_to_string)
        df_out["Confidence"] = np.random.choice(
            ["High", "Medium", "Low"], size=len(df_out), p=[0.6, 0.3, 0.1]
        )
        return df_out

    def build_map(df_in: pd.DataFrame):
        fig = px.choropleth_map(
            df_in,
            geojson=plotly_geojson,
            locations="Hexagon_ID",
            color="Dominant Class",
            color_discrete_map=color_map,
            hover_name="Hexagon_ID",
            hover_data={
                "Hexagon_ID": False,
                "Dominant Class": True,
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
        fig.update_traces(
            marker=dict(opacity=0.2),
            selected=dict(marker=dict(opacity=0.8)),
            unselected=dict(marker=dict(opacity=0.0)),
        )
        return fig

    predicted_dfs = [build_predicted_df(prediction_delta=d) for d in [0, time_delta]]
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
                    diff_mask, ["Hexagon_ID", "Dominant Class", *class_cols]
                ].add_suffix("_selected"),
                future_pred_df.loc[
                    diff_mask, ["Hexagon_ID", "Dominant Class", *class_cols]
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
        start_year_dropdown = gr.Dropdown(
            choices=[2016, 2017, 2018, 2019, 2020, 2021],
            value=2019,
            label="Start Year Selection",
        )
        time_delta_drop_down = gr.Dropdown(
            choices=[i for i in range(0, 5)],
            value=1,
            label="Future Time (Years)",
        )

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
        "inputs": [start_year_dropdown, time_delta_drop_down, map_type_radio],
        "outputs": [map_output_left, map_output_right],
    }
    map_type_radio.change(**update_args)
    start_year_dropdown.change(**update_args)
    time_delta_drop_down.change(**update_args)
    # clear_selection_button.click(**update_args)
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
