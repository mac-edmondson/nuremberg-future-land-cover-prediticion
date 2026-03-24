import json
import pickle
from pathlib import Path

import geopandas as gpd
import gradio as gr
from gradio.themes import Ocean
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import box, shape

lat_center = 49.4330
lon_center = 11.0767

class_cols = [
    "built_up",
    "vegetation",
    "water",
]

SELECTION_COLUMNS = [
    "grid_id",
    "Dominant Class",
    "built_up",
    "vegetation",
    "water",
    "lat",
    "lon",
]

SUBGRID_COLUMNS = [
    "parent_grid_id",
    "subgrid_id",
    "grid_x",
    "grid_y",
    "Parent Dominant Class",
    "Subgrid Dominant Class",
    "built_up",
    "vegetation",
    "water",
    "lat",
    "lon",
]

LAST_GRID_TABLES: dict[str, pd.DataFrame] = {
    "selected_year": pd.DataFrame(columns=SELECTION_COLUMNS),
    "future_prediction": pd.DataFrame(columns=SELECTION_COLUMNS),
}

LAST_SUBGRID_TABLES: dict[str, pd.DataFrame] = {
    "selected_year": pd.DataFrame(columns=SUBGRID_COLUMNS),
    "future_prediction": pd.DataFrame(columns=SUBGRID_COLUMNS),
}

LAND_COVER_LABELS = {
    "built_up": "Built Up",
    "vegetation": "Vegetation",
    "water": "Water",
}

LAND_COVER_COLORS = {
    "Built Up": "#fa0000",
    "Vegetation": "#006400",
    "Water": "#0064ff",
}

DEFAULT_SELECTION_MESSAGE = "Select a region to compare land-cover composition between selected year and future prediction."


class PlotSelectionBridge(gr.HTML):
    """Custom Gradio HTML component that emits Plotly selection payloads."""

    def __init__(
        self,
        left_plot_id: str,
        right_plot_id: str,
        *,
        value: dict | None = None,
        label: str | None = None,
        **kwargs,
    ):
        html_template = """
        <div class="selection-bridge" style="padding: 8px 12px; border: 1px solid #d8dee4; border-radius: 8px; margin-top: 8px;">
            <div style="font-size: 0.9rem; color: #24292f;">
                <strong>Selection Bridge:</strong> Use box/lasso select or click on either map to capture grid IDs.
            </div>
            <div id="selection-bridge-status" style="font-size: 0.85rem; color: #57606a; margin-top: 4px;">
                Waiting for selection...
            </div>
        </div>
        """

        js_on_load = """
        const leftPlotId = props.left_plot_id;
        const rightPlotId = props.right_plot_id;
        const statusEl = element.querySelector('#selection-bridge-status');

        const SYNC_KEY = '__plotSyncState';

        function ensureSyncState(plotlyDiv) {
            if (!plotlyDiv[SYNC_KEY]) {
                plotlyDiv[SYNC_KEY] = {
                    applyingRelayout: false,
                    applyingSelection: false,
                };
            }
            return plotlyDiv[SYNC_KEY];
        }

        function asInt(value) {
            if (value === null || value === undefined) return null;
            const parsed = parseInt(value, 10);
            return Number.isNaN(parsed) ? null : parsed;
        }

        function extractGridIds(eventData) {
            if (!eventData || !Array.isArray(eventData.points)) return [];
            const ids = [];
            for (const point of eventData.points) {
                const custom = point?.customdata;
                if (Array.isArray(custom) && custom.length > 0) {
                    const id = asInt(custom[0]);
                    if (id !== null) ids.push(id);
                    continue;
                }
                const location = asInt(point?.location);
                if (location !== null) {
                    ids.push(location);
                    continue;
                }
                const pointNumber = asInt(point?.pointNumber);
                if (pointNumber !== null) ids.push(pointNumber);
            }
            return Array.from(new Set(ids));
        }

        function buildMapRelayoutPatch(relayoutData) {
            if (!relayoutData || typeof relayoutData !== 'object') {
                return null;
            }

            const patch = {};

            // Maplibre-style keys (Plotly 6+)
            if (relayoutData['map.center']) patch['map.center'] = relayoutData['map.center'];
            if (relayoutData['map.zoom'] !== undefined) patch['map.zoom'] = relayoutData['map.zoom'];
            if (relayoutData['map.bearing'] !== undefined) patch['map.bearing'] = relayoutData['map.bearing'];
            if (relayoutData['map.pitch'] !== undefined) patch['map.pitch'] = relayoutData['map.pitch'];
            if (relayoutData['map.center.lat'] !== undefined && relayoutData['map.center.lon'] !== undefined) {
                patch['map.center'] = {
                    lat: relayoutData['map.center.lat'],
                    lon: relayoutData['map.center.lon'],
                };
            }

            // Mapbox-style keys (older Plotly)
            if (relayoutData['mapbox.center']) patch['mapbox.center'] = relayoutData['mapbox.center'];
            if (relayoutData['mapbox.zoom'] !== undefined) patch['mapbox.zoom'] = relayoutData['mapbox.zoom'];
            if (relayoutData['mapbox.bearing'] !== undefined) patch['mapbox.bearing'] = relayoutData['mapbox.bearing'];
            if (relayoutData['mapbox.pitch'] !== undefined) patch['mapbox.pitch'] = relayoutData['mapbox.pitch'];
            if (relayoutData['mapbox.center.lat'] !== undefined && relayoutData['mapbox.center.lon'] !== undefined) {
                patch['mapbox.center'] = {
                    lat: relayoutData['mapbox.center.lat'],
                    lon: relayoutData['mapbox.center.lon'],
                };
            }

            return Object.keys(patch).length ? patch : null;
        }

        function selectedPointsByTrace(targetDiv, selectedIds) {
            const selectedSet = new Set(selectedIds);
            const selectionPerTrace = [];

            const traces = Array.isArray(targetDiv.data) ? targetDiv.data : [];
            for (const trace of traces) {
                const customData = trace?.customdata;
                if (!Array.isArray(customData) || customData.length === 0) {
                    selectionPerTrace.push(null);
                    continue;
                }

                const indices = [];
                for (let i = 0; i < customData.length; i += 1) {
                    const row = customData[i];
                    const rawId = Array.isArray(row) ? row[0] : row;
                    const id = asInt(rawId);
                    if (id !== null && selectedSet.has(id)) {
                        indices.push(i);
                    }
                }
                selectionPerTrace.push(indices.length ? indices : []);
            }
            return selectionPerTrace;
        }

        function mirrorSelection(sourceDiv, targetDiv, eventKind, eventData) {
            if (!targetDiv) return;
            const targetState = ensureSyncState(targetDiv);
            const sourceState = ensureSyncState(sourceDiv);
            if (sourceState.applyingSelection || targetState.applyingSelection) {
                return;
            }

            const selectedIds = extractGridIds(eventData);
            targetState.applyingSelection = true;
            const traces = Array.isArray(targetDiv.data) ? targetDiv.data : [];
            const traceIndices = traces.map((_, i) => i);
            if (!traceIndices.length) {
                targetState.applyingSelection = false;
                return;
            }

            // Important: use null (not []) to clear selection, otherwise polygons can remain hidden.
            if (eventKind === 'deselect' || selectedIds.length === 0) {
                const clearSelection = traceIndices.map(() => null);
                Plotly.restyle(targetDiv, { selectedpoints: clearSelection }, traceIndices)
                    .finally(() => {
                        targetState.applyingSelection = false;
                    });
                return;
            }

            const targetSelection = selectedPointsByTrace(targetDiv, selectedIds);
            Plotly.restyle(targetDiv, { selectedpoints: targetSelection }, traceIndices)
                .finally(() => {
                    targetState.applyingSelection = false;
                });
        }

        function mirrorRelayout(sourceDiv, targetDiv, relayoutData) {
            if (!targetDiv) return;
            const patch = buildMapRelayoutPatch(relayoutData);
            if (!patch) return;

            const sourceState = ensureSyncState(sourceDiv);
            const targetState = ensureSyncState(targetDiv);
            if (sourceState.applyingRelayout || targetState.applyingRelayout) {
                return;
            }

            targetState.applyingRelayout = true;
            Plotly.relayout(targetDiv, patch)
                .finally(() => {
                    targetState.applyingRelayout = false;
                });
        }

        function pushPayload(source, eventKind, eventData) {
            const gridIds = extractGridIds(eventData);
            const payload = {
                source: source,
                event_kind: eventKind,
                grid_ids: gridIds,
                ts: Date.now(),
            };
            props.value = payload;
            if (statusEl) {
                if (gridIds.length > 0) {
                    statusEl.textContent = `${source}: ${gridIds.length} selected (${gridIds.slice(0, 10).join(', ')}${gridIds.length > 10 ? ', ...' : ''})`;
                } else {
                    statusEl.textContent = `${source}: selection cleared`;
                }
            }
        }

        function attachHandlers(plotContainer, sourceName, getPeerDiv) {
            if (!plotContainer) {
                return;
            }
            const plotlyDiv = plotContainer.querySelector('.js-plotly-plot');
            if (!plotlyDiv || typeof plotlyDiv.on !== 'function') {
                return;
            }

            if (plotContainer.__selectionBridgeBoundTo === plotlyDiv) {
                return;
            }

            plotlyDiv.on('plotly_selected', (eventData) => {
                const peerDiv = getPeerDiv();
                mirrorSelection(plotlyDiv, peerDiv, 'selected', eventData);
                pushPayload(sourceName, 'selected', eventData);
            });

            plotlyDiv.on('plotly_click', (eventData) => {
                const peerDiv = getPeerDiv();
                mirrorSelection(plotlyDiv, peerDiv, 'click', eventData);
                pushPayload(sourceName, 'click', eventData);
            });

            plotlyDiv.on('plotly_deselect', () => {
                const emptyEvent = { points: [] };
                const peerDiv = getPeerDiv();
                mirrorSelection(plotlyDiv, peerDiv, 'deselect', emptyEvent);
                pushPayload(sourceName, 'deselect', emptyEvent);
            });

            plotlyDiv.on('plotly_relayout', (relayoutData) => {
                const peerDiv = getPeerDiv();
                mirrorRelayout(plotlyDiv, peerDiv, relayoutData);
            });

            plotContainer.__selectionBridgeBoundTo = plotlyDiv;
        }

        function getPlotlyDiv(containerId) {
            const container = document.getElementById(containerId);
            if (!container) return null;
            return container.querySelector('.js-plotly-plot');
        }

        function tryAttachAll() {
            const leftContainer = document.getElementById(leftPlotId);
            const rightContainer = document.getElementById(rightPlotId);
            attachHandlers(leftContainer, 'selected_year', () => getPlotlyDiv(rightPlotId));
            attachHandlers(rightContainer, 'future_prediction', () => getPlotlyDiv(leftPlotId));
        }

        const observer = new MutationObserver(() => tryAttachAll());
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(() => tryAttachAll(), 0);
        setTimeout(() => tryAttachAll(), 400);
        setTimeout(() => tryAttachAll(), 1000);
        """

        super().__init__(
            value=value or {},
            html_template=html_template,
            js_on_load=js_on_load,
            left_plot_id=left_plot_id,
            right_plot_id=right_plot_id,
            label=label,
            **kwargs,
        )

    def api_info(self):
        return {
            "type": "object",
            "description": "Selection payload with source, event_kind, and grid_ids.",
        }


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


def assign_row_dominant_class(
    df: pd.DataFrame,
    class_columns: list[str],
    output_column: str = "Subgrid Dominant Class",
) -> pd.DataFrame:
    """Assign dominant class for each individual row using per-row class scores."""
    missing_cols = [col for col in class_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for row dominant class assignment: {missing_cols}"
        )

    row_scores = df[class_columns].to_numpy()
    dominant_idx = np.argmax(row_scores, axis=1)
    out = df.copy()
    out[output_column] = [map_class_to_string(idx) for idx in dominant_idx]
    return out


def normalize_class_scores(
    df: pd.DataFrame,
    class_columns: list[str],
) -> pd.DataFrame:
    """Normalize class score columns row-wise so they are non-negative and sum to 1."""
    missing_cols = [col for col in class_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for score normalization: {missing_cols}"
        )

    out = df.copy()
    scores = out[class_columns].astype(float).to_numpy()
    scores = np.clip(scores, a_min=0.0, a_max=None)
    row_sums = scores.sum(axis=1, keepdims=True)

    # Keep rows stable even if model outputs all non-positive values.
    zero_sum_mask = row_sums.squeeze(axis=1) <= 1e-12
    if np.any(zero_sum_mask):
        scores[zero_sum_mask] = 1.0 / len(class_columns)
        row_sums = scores.sum(axis=1, keepdims=True)

    normalized = scores / row_sums
    out[class_columns] = normalized
    return out


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
                predicted_targets_df = normalize_class_scores(
                    predicted_targets_df,
                    class_cols,
                )
                df_out[class_cols] = predicted_targets_df[class_cols]
            else:
                print(f"Skipping Prediction: {prediction_model=}; {feature_names=}")
        else:
            print(f"Skipping Prediction: {prediction_model=}; {feature_names=}")

        df_out = assign_row_dominant_class(df_out, class_cols)
        df_out = assign_group_dominant_class(df_out, class_cols)
        df_out["Parent Dominant Class"] = df_out["Dominant Class"]
        return df_out

    def build_subgrid_table(df_in: pd.DataFrame) -> pd.DataFrame:
        subgrid_df = df_in.merge(
            grid_metadata[["grid_x", "grid_y", "grid_id"]],
            on=["grid_x", "grid_y"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"grid_id_x": "subgrid_id", "grid_id_y": "parent_grid_id"})

        # Backfill for compatibility if a prior path did not create both columns.
        if (
            "Parent Dominant Class" not in subgrid_df.columns
            and "Dominant Class" in subgrid_df.columns
        ):
            subgrid_df["Parent Dominant Class"] = subgrid_df["Dominant Class"]
        if "Subgrid Dominant Class" not in subgrid_df.columns:
            subgrid_df = assign_row_dominant_class(
                subgrid_df,
                class_cols,
                output_column="Subgrid Dominant Class",
            )

        for col in SUBGRID_COLUMNS:
            if col not in subgrid_df.columns:
                subgrid_df[col] = np.nan

        return subgrid_df[SUBGRID_COLUMNS].copy()

    def build_map(df_in: pd.DataFrame):
        # Aggregate sub-grid predictions into parent-grid scores, then compute dominant class/confidence.
        grouped_scores = df_in.groupby(["grid_x", "grid_y"], as_index=False)[
            class_cols
        ].sum()
        grid_df = grouped_scores.copy()

        dominant_idx = np.argmax(grid_df[class_cols].to_numpy(), axis=1)
        grid_df["Dominant Class"] = [map_class_to_string(idx) for idx in dominant_idx]

        score_matrix = grid_df[class_cols].to_numpy()
        dominant_values = score_matrix[np.arange(len(grid_df)), dominant_idx]
        total_values = score_matrix.sum(axis=1)
        dominant_pct = np.where(
            total_values > 1e-12,
            (dominant_values / total_values) * 100.0,
            50.0,
        )
        clipped_pct = np.clip(dominant_pct, 50.0, 100.0)
        grid_df["Confidence"] = np.where(
            clipped_pct < (50.0 + (50.0 / 3.0)),
            "Low",
            np.where(
                clipped_pct < (50.0 + (2.0 * 50.0 / 3.0)),
                "Medium",
                "High",
            ),
        )

        # Use average lat/lon for representative point rendering.
        lat_lon_df = (
            df_in.groupby(["grid_x", "grid_y"], as_index=False)[["lat", "lon"]]
            .mean()
            .reset_index(drop=True)
        )
        grid_df = grid_df.merge(
            lat_lon_df,
            on=["grid_x", "grid_y"],
            how="left",
            validate="one_to_one",
        )

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
                custom_data=["grid_id", "Dominant Class", "Confidence"],
                color_discrete_map=color_map,
                hover_name=None,
                hover_data={"grid_id": False, "lat": False, "lon": False},
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
                custom_data=["grid_id", "Dominant Class", "Confidence"],
                color_discrete_map=color_map,
                hover_name=None,
                hover_data={"grid_id": False},
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
            clickmode="event+select",
            height=700,
        )
        if render_mode == "points":
            fig.update_traces(
                hovertemplate="Dominant Class: %{customdata[1]}<br>Confidence: %{customdata[2]}<extra></extra>",
                marker=dict(size=7, opacity=0.4),
                selected=dict(marker=dict(opacity=0.8)),
                unselected=dict(marker=dict(opacity=0.1)),
            )
        else:
            fig.update_traces(
                hovertemplate="Dominant Class: %{customdata[1]}<br>Confidence: %{customdata[2]}<extra></extra>",
                marker=dict(opacity=0.3),
                selected=dict(marker=dict(opacity=0.8)),
                unselected=dict(marker=dict(opacity=0.0)),
            )
        return fig, grid_df

    if delta_selection == 0:
        predicted_dfs = [
            assign_group_dominant_class(base_df.copy(), class_cols),
            build_predicted_df(prediction_delta=time_delta),
        ]
    else:
        predicted_dfs = [
            build_predicted_df(prediction_delta=d) for d in [0, time_delta]
        ]
    (selected_fig, selected_grid_df), (future_fig, future_grid_df) = [
        build_map(df_item) for df_item in predicted_dfs
    ]

    global LAST_GRID_TABLES, LAST_SUBGRID_TABLES
    LAST_GRID_TABLES = {
        "selected_year": selected_grid_df[SELECTION_COLUMNS].copy(),
        "future_prediction": future_grid_df[SELECTION_COLUMNS].copy(),
    }
    LAST_SUBGRID_TABLES = {
        "selected_year": build_subgrid_table(predicted_dfs[0]),
        "future_prediction": build_subgrid_table(predicted_dfs[1]),
    }

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


def selection_payload_to_outputs(payload: dict | None):
    def hidden_plot_update():
        return gr.update(value=None, visible=False)

    def visible_plot_update(fig: go.Figure):
        return gr.update(value=fig, visible=True)

    def build_empty_pie(title: str, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=340,
            margin={"r": 10, "t": 50, "l": 10, "b": 10},
            annotations=[
                {
                    "text": message,
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return fig

    def build_empty_delta(title: str, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=290,
            margin={"r": 30, "t": 50, "l": 30, "b": 30},
            xaxis_title="Delta (percentage points)",
            yaxis_title="Land Cover",
            annotations=[
                {
                    "text": message,
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return fig

    def build_land_cover_composition(subgrid_df: pd.DataFrame) -> pd.DataFrame:
        if subgrid_df.empty:
            return pd.DataFrame(columns=["class", "value", "percent"])

        values = subgrid_df[class_cols].sum(axis=0).clip(lower=0)
        total = float(values.sum())
        if total <= 0:
            return pd.DataFrame(columns=["class", "value", "percent"])

        comp_df = pd.DataFrame(
            {
                "class": [LAND_COVER_LABELS[col] for col in class_cols],
                "value": [float(values[col]) for col in class_cols],
            }
        )
        comp_df["percent"] = (comp_df["value"] / total) * 100.0
        return comp_df

    def build_land_cover_pie(title: str, comp_df: pd.DataFrame) -> go.Figure:
        if comp_df.empty:
            return build_empty_pie(title, "No selected sub-grid data")

        labels = comp_df["class"].tolist()
        values = comp_df["value"].tolist()
        percents = comp_df["percent"].tolist()
        colors = [LAND_COVER_COLORS[label] for label in labels]

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            column_widths=[0.58, 0.42],
            horizontal_spacing=0.08,
        )

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.35,
                marker=dict(colors=colors),
                textposition="inside",
                texttemplate="%{label}<br>%{percent:.1%}",
                hovertemplate="%{label}: %{percent:.2%}<extra></extra>",
                sort=False,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=percents,
                marker=dict(color=colors),
                text=[f"{v:.2f}%" for v in percents],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_yaxes(
            title_text="Percent",
            ticksuffix="%",
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text="Class",
            showgrid=False,
            row=1,
            col=2,
        )
        fig.update_layout(
            title=title,
            height=360,
            margin={"r": 10, "t": 50, "l": 10, "b": 20},
            legend=dict(orientation="h", y=-0.18),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def build_delta_plot(delta_df: pd.DataFrame) -> go.Figure:
        if delta_df.empty:
            return build_empty_delta(
                "Land-Cover Composition Delta",
                "No selected sub-grid data",
            )

        fig = px.bar(
            delta_df,
            x="delta_pp",
            y="class",
            color="direction",
            orientation="h",
            color_discrete_map={
                "Increase": "#2e7d32",
                "Decrease": "#c62828",
                "No Change": "#6c757d",
            },
        )
        fig.update_traces(
            texttemplate="%{x:.2f}",
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} pp<extra></extra>",
        )
        fig.update_layout(
            title="Land-Cover Composition Delta (Future - Selected Year)",
            height=290,
            margin={"r": 30, "t": 50, "l": 30, "b": 30},
            xaxis_title="Predicted Change (percentage points)",
            yaxis_title="Land Cover",
            legend_title_text="Change",
            showlegend=True,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#444")
        return fig

    if not payload or not isinstance(payload, dict):
        return (
            DEFAULT_SELECTION_MESSAGE,
            hidden_plot_update(),
            hidden_plot_update(),
            hidden_plot_update(),
        )

    raw_ids = payload.get("grid_ids") or []

    grid_ids = []
    seen = set()
    for item in raw_ids:
        try:
            as_int = int(item)
        except Exception:
            continue
        if as_int not in seen:
            seen.add(as_int)
            grid_ids.append(as_int)

    if not grid_ids:
        return (
            DEFAULT_SELECTION_MESSAGE,
            hidden_plot_update(),
            hidden_plot_update(),
            hidden_plot_update(),
        )

    selected_subgrid_table = LAST_SUBGRID_TABLES.get("selected_year")
    future_subgrid_table = LAST_SUBGRID_TABLES.get("future_prediction")
    if selected_subgrid_table is None or selected_subgrid_table.empty:
        return (
            "Selected-year sub-grid data is not available yet. Click Submit and select a region.",
            hidden_plot_update(),
            hidden_plot_update(),
            hidden_plot_update(),
        )
    if future_subgrid_table is None or future_subgrid_table.empty:
        return (
            "Future-year sub-grid data is not available yet. Click Submit and select a region.",
            hidden_plot_update(),
            hidden_plot_update(),
            hidden_plot_update(),
        )

    selected_subgrid_rows = selected_subgrid_table[
        selected_subgrid_table["parent_grid_id"].isin(grid_ids)
    ].copy()
    future_subgrid_rows = future_subgrid_table[
        future_subgrid_table["parent_grid_id"].isin(grid_ids)
    ].copy()

    selected_comp = build_land_cover_composition(selected_subgrid_rows)
    future_comp = build_land_cover_composition(future_subgrid_rows)

    selected_pie = build_land_cover_pie("Selected Year Composition", selected_comp)
    future_pie = build_land_cover_pie("Future Prediction Composition", future_comp)

    selected_pct = {
        row["class"]: float(row["percent"]) for _, row in selected_comp.iterrows()
    }
    future_pct = {
        row["class"]: float(row["percent"]) for _, row in future_comp.iterrows()
    }

    delta_rows = []
    for label in ["Built Up", "Vegetation", "Water"]:
        delta = future_pct.get(label, 0.0) - selected_pct.get(label, 0.0)
        if delta > 1e-9:
            direction = "Increase"
        elif delta < -1e-9:
            direction = "Decrease"
        else:
            direction = "No Change"
        delta_rows.append({"class": label, "delta_pp": delta, "direction": direction})

    delta_df = pd.DataFrame(delta_rows)
    delta_plot = build_delta_plot(delta_df)

    summary = (
        f"Selection: {len(grid_ids)} parent grids | "
        f"{len(selected_subgrid_rows)} selected-year sub-grids | "
        f"{len(future_subgrid_rows)} future sub-grids."
    )
    return (
        summary,
        visible_plot_update(selected_pie),
        visible_plot_update(future_pie),
        visible_plot_update(delta_plot),
    )


def reset_selection_insights():
    return selection_payload_to_outputs(None)


def build_map_titles(start_year, time_delta):
    start_year = int(start_year)
    time_delta = int(time_delta)
    selected_year_label_type = (
        "True Labels" if start_year == 2021 else "Predicted Labels"
    )
    selected_title = f"### {start_year} Land-Cover ({selected_year_label_type})"
    future_title = f"### {start_year + time_delta} Land-Cover (Predicted)"
    return selected_title, future_title


def update_dashboard_with_titles(
    start_year,
    time_delta,
    map_type,
    render_mode,
    grid_cell_size,
):
    selected_fig, future_fig = update_dashboard(
        start_year=start_year,
        time_delta=time_delta,
        map_type=map_type,
        render_mode=render_mode,
        grid_cell_size=grid_cell_size,
    )
    left_title, right_title = build_map_titles(start_year, time_delta)
    return selected_fig, future_fig, left_title, right_title


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
            left_map_title = gr.Markdown("### 2021 Land-Cover (True Labels)")
            map_output_left = gr.Plot(show_label=False, elem_id="map_output_left")
        with gr.Column():
            right_map_title = gr.Markdown("### 2021 Land-Cover (Predicted)")
            map_output_right = gr.Plot(show_label=False, elem_id="map_output_right")

    selection_bridge = PlotSelectionBridge(
        left_plot_id="map_output_left",
        right_plot_id="map_output_right",
        label="Selection Bridge",
        visible="hidden",
    )

    with gr.Row():
        with gr.Column():
            selected_year_pie = gr.Plot(
                show_label=False,
                value=None,
                visible=False,
            )
        with gr.Column():
            future_year_pie = gr.Plot(
                show_label=False,
                value=None,
                visible=False,
            )

    with gr.Row():
        delta_plot = gr.Plot(
            show_label=False,
            value=None,
            visible=False,
        )

    with gr.Row():
        selection_summary = gr.Markdown(DEFAULT_SELECTION_MESSAGE)

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
        "fn": update_dashboard_with_titles,
        "inputs": [
            start_year_dropdown,
            time_delta_drop_down,
            map_type_radio,
            render_mode_radio,
            grid_cell_size_dropdown,
        ],
        "outputs": [map_output_left, map_output_right, left_map_title, right_map_title],
    }
    submit_button.click(**update_args)
    submit_button.click(
        fn=reset_selection_insights,
        inputs=None,
        outputs=[selection_summary, selected_year_pie, future_year_pie, delta_plot],
    )
    selection_bridge.change(
        fn=selection_payload_to_outputs,
        inputs=[selection_bridge],
        outputs=[selection_summary, selected_year_pie, future_year_pie, delta_plot],
    )
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

theme = Ocean()
if __name__ == "__main__":
    app.launch(theme=theme)
