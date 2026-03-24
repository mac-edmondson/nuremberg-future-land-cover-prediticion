"""
Nuremberg Future Land-Cover Prediction Dashboard

This Gradio-based web application provides interactive visualization and analysis of land-cover
changes in Nuremberg, Germany. It leverages Sentinel-2 satellite imagery, XGBoost predictions,
and GIS data to forecast future land-cover compositions across georeferenced grid cells.

Key Features:
  - Compare actual vs. predicted land cover for user-selected years
  - Interactive borough-level composition analysis
  - Detailed sub-grid breakdowns with confidence estimates
  - Dual-map synchronization for seamless selection and exploration
  - Zoom and pan to specific boroughs for focused analysis

Data Sources:
  - ESA WorldCover 2020/2021 satellite imagery (Sentinel-2)
  - Nuremberg borough boundaries from OpenStreetMap Nominatim
  - XGBoost delta-year prediction model trained on historical imagery
"""

import json
import pickle
import time
import traceback
import warnings
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

# ============================================================================
# MAP CENTER AND COORDINATE CONSTANTS
# ============================================================================

# Geographic center of Nuremberg for map initialization
LAT_CENTER: float = 49.4330
LON_CENTER: float = 11.0767


# ============================================================================
# LAND COVER CLASS DEFINITIONS
# ============================================================================

# Primary land cover classes predicted by the model
CLASS_COLS: list[str] = [
    "built_up",
    "vegetation",
    "water",
]

# Human-readable labels for land cover classes
LAND_COVER_LABELS: dict[str, str] = {
    "built_up": "Built Up",
    "vegetation": "Vegetation",
    "water": "Water",
}

# Hex color codes for consistent visualization across all plots
LAND_COVER_COLORS: dict[str, str] = {
    "Built Up": "#fa0000",  # Red for built-up areas
    "Vegetation": "#006400",  # Dark green for vegetation
    "Water": "#0064ff",  # Blue for water
}


# ============================================================================
# DATAFRAME COLUMN SPECIFICATIONS
# ============================================================================

# Columns included in grid-level aggregated output tables
SELECTION_COLUMNS: list[str] = [
    "grid_id",
    "Dominant Class",
    "built_up",
    "vegetation",
    "water",
    "lat",
    "lon",
]

# Columns included in sub-grid level detailed output tables
SUBGRID_COLUMNS: list[str] = [
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


# ============================================================================
# NUREMBERG BOROUGH DEFINITIONS
# ============================================================================

# Official 10 Nuremberg borough names for spatial joins and visualization
NURNBERG_BOROUGH_NAMES: list[str] = [
    "Altstadt und engere Innenstadt",
    "Weiterer Innenstadtgürtel Süd",
    "Östliche Außenstadt",
    "Nordöstliche Außenstadt",
    "Nordwestliche Außenstadt",
    "Westliche Außenstadt",
    "Südwestliche Außenstadt",
    "Südliche Außenstadt",
    "Südöstliche Außenstadt",
    "Weiterer Innenstadtgürtel West/Nord/Ost",
]

# Precomputed borough boundaries are stored as a static GeoJSON asset.
BOROUGH_BOUNDARIES_PATH: Path = (
    Path(__file__).resolve().parent / "assets" / "nuremberg_borough_boundaries.geojson"
)


# ============================================================================
# GLOBAL STATE FOR CACHING AND EVENT SYNCHRONIZATION
# ============================================================================

# Cache for grid-level aggregated data (parent grids with dominant classes)
LAST_GRID_TABLES: dict[str, pd.DataFrame] = {
    "selected_year": pd.DataFrame(columns=SELECTION_COLUMNS),
    "future_prediction": pd.DataFrame(columns=SELECTION_COLUMNS),
}

# Cache for sub-grid level data (detailed pixel-level composition per parent grid)
LAST_SUBGRID_TABLES: dict[str, pd.DataFrame] = {
    "selected_year": pd.DataFrame(columns=SUBGRID_COLUMNS),
    "future_prediction": pd.DataFrame(columns=SUBGRID_COLUMNS),
}

# Cache for the borough-level composition change chart (updated on Submit)
LAST_BOROUGH_CHANGE_FIG: go.Figure | None = None

# Cache for the two interactive maps (selected year and future prediction)
LAST_MAP_FIGURES: dict[str, go.Figure | None] = {
    "selected_year": None,
    "future_prediction": None,
}

# Timestamp (milliseconds) of the last Submit button click for event guard
LAST_SUBMIT_TS_MS: int = 0

# Time window (milliseconds) for filtering out stale re-render events from Plotly
SELECTION_EVENT_GUARD_MS: int = 900

# User-facing message displayed when no region is selected
DEFAULT_SELECTION_MESSAGE: str = "Select a region to compare land-cover composition between selected year and future prediction."


# ============================================================================
# PLACEHOLDER FIGURE BUILDERS
# ============================================================================


def build_empty_borough_change_plot(message: str) -> go.Figure:
    """
    Create an empty borough change chart with a centered message.

    Used as a placeholder when no data is available yet or while data is loading.
    Displays a centered text annotation on a blank figure with appropriate styling.

    Args:
        message: The user-facing message to display in the center of the figure.

    Returns:
        A Plotly Figure with empty state styling and centered message.
    """
    fig = go.Figure()
    fig.update_layout(
        title="Borough Change Overview",
        height=340,
        margin={"r": 20, "t": 60, "l": 20, "b": 20},
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


def build_empty_pie(title: str, message: str) -> go.Figure:
    """
    Create an empty pie chart placeholder with a centered message.

    Used when no grid selection exists or data hasn't been computed yet.
    Maintains consistent styling with other empty state plots.

    Args:
        title: The title to display at the top of the figure.
        message: The user-facing message to display in the center of the figure.

    Returns:
        A Plotly Figure with pie chart styling and centered message.
    """
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
    """
    Create an empty delta/bar chart placeholder with a centered message.

    Used when no grid selection exists or delta data hasn't been computed yet.
    Maintains consistent ax is labels and styling with data-filled delta plots.

    Args:
        title: The title to display at the top of the figure.
        message: The user-facing message to display in the center of the figure.

    Returns:
        A Plotly Figure with delta chart styling and centered message.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=290,
        margin={"r": 30, "t": 50, "l": 30, "b": 30},
        xaxis_title="Delta (%)",
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


# ============================================================================
# GIS AND BOROUGH BOUNDARY FUNCTIONS
# ============================================================================


def load_nuremberg_borough_boundaries() -> gpd.GeoDataFrame | None:
    """
    Load precomputed Nuremberg borough polygons from a static GeoJSON asset.

    The boundaries file is generated once and committed under assets so dashboard startup
    does not rely on live network calls.

    Args:
        None

    Returns:
        GeoDataFrame with columns (borough, geometry) in EPSG:4326 (WGS84), or None if
        the file is missing, invalid, or empty.

    Raises:
        File and parsing errors are caught internally; the function logs warnings rather
        than raising exceptions.
    """
    if not BOROUGH_BOUNDARIES_PATH.exists():
        print(f"Borough boundaries file not found: {BOROUGH_BOUNDARIES_PATH}")
        return None

    try:
        boundaries = gpd.read_file(BOROUGH_BOUNDARIES_PATH)
    except Exception as err:
        print(
            f"Failed to load borough boundaries from {BOROUGH_BOUNDARIES_PATH}: {err}"
        )
        return None

    if boundaries.empty:
        print(f"Borough boundaries file is empty: {BOROUGH_BOUNDARIES_PATH}")
        return None

    if "borough" not in boundaries.columns:
        print(
            f"Borough boundaries file is missing required 'borough' column: {BOROUGH_BOUNDARIES_PATH}"
        )
        return None

    # Ensure boundaries are in WGS84 for map projection compatibility.
    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:4326", allow_override=True)
    else:
        boundaries = boundaries.to_crs("EPSG:4326")

    # Keep output schema stable for downstream spatial join logic.
    boundaries = boundaries[["borough", "geometry"]].copy()

    if len(boundaries) < len(NURNBERG_BOROUGH_NAMES):
        print(
            f"Warning: loaded {len(boundaries)} borough boundaries out of {len(NURNBERG_BOROUGH_NAMES)}."
        )

    return boundaries


# Load borough boundaries at module initialization; used globally by multiple functions
BOROUGH_BOUNDARIES_GDF = load_nuremberg_borough_boundaries()


def calculate_borough_bounds() -> dict[str, dict]:
    """
    Calculate lat/lon bounds and zoom level for each borough from BOROUGH_BOUNDARIES_GDF.

    Computes bounding boxes (min/max lat/lon), center coordinates, and an appropriate
    zoom level based on borough spatial extent. Used for map zoom-to-borough functionality.

    Args:
        None

    Returns:
        Dictionary mapping borough names to dicts with keys:
        - min_lat, max_lat, min_lon, max_lon: Bounding box in decimal degrees
        - center_lat, center_lon: Centroid coordinates
        - zoom: Integer zoom level (11-14) suitable for Plotly map view

        Returns empty dict if BOROUGH_BOUNDARIES_GDF is None or empty.
    """
    bounds = {}

    # Return empty dict if borough data unavailable
    if BOROUGH_BOUNDARIES_GDF is None or BOROUGH_BOUNDARIES_GDF.empty:
        return bounds

    # Iterate through each borough and compute its spatial bounds
    for _, row in BOROUGH_BOUNDARIES_GDF.iterrows():
        borough_name = row["borough"]
        geom = row["geometry"]

        # Use Shapely bounds method: (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max)
        minx, miny, maxx, maxy = geom.bounds

        # Calculate center coordinates as average of bounds
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

        # Compute borough spatial extent in degrees
        lon_span = maxx - minx
        lat_span = maxy - miny
        max_span = max(lon_span, lat_span)

        # Empirical formula for Plotly zoom levels based on spatial extent.
        # Smaller boroughs (~0.02 degrees) → higher zoom (14)
        # Larger boroughs (~0.10 degrees) → lower zoom (11)
        zoom = max(11, min(14, int(14 - (max_span * 100))))

        bounds[borough_name] = {
            "min_lat": float(miny),
            "max_lat": float(maxy),
            "min_lon": float(minx),
            "max_lon": float(maxx),
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "zoom": zoom,
        }

    return bounds


def build_top_changed_boroughs_chart(
    selected_subgrid_df: pd.DataFrame,
    future_subgrid_df: pd.DataFrame,
) -> go.Figure:
    """
    Build a stacked bar chart showing positive land-cover composition changes by borough.

    Compares current (selected year) composition to future (predicted) composition for each
    borough, and displays only positive changes (classes that increased). This provides insight
    into which boroughs are gaining specific land-cover types.

    Processing steps:
    1. Spatially join sub-grid points to borough polygons
    2. Aggregate land-cover class scores by borough
    3. Convert to percentages to ensure comparable metrics
    4. Compute per-class positive deltas (future_pct - selected_pct, clipped at 0)
    5. Ensure all 10 official boroughs appear (fill missing with 0)
    6. Create stacked bar chart with text value labels

    Args:
        selected_subgrid_df: DataFrame with selected year sub-grid data (columns: lat, lon, built_up, vegetation, water)
        future_subgrid_df: DataFrame with future prediction sub-grid data (same schema)

    Returns:
        Plotly Figure showing a stacked bar chart with one bar per borough, grouped by
        land-cover class (Built Up, Vegetation, Water) with value labels. Returns empty
        state figure if input data is insufficient or borough boundaries unavailable.
    """
    # Check for empty input dataframes
    if selected_subgrid_df.empty or future_subgrid_df.empty:
        return build_empty_borough_change_plot("No grid data available yet.")

    boundaries = BOROUGH_BOUNDARIES_GDF
    if boundaries is None or boundaries.empty:
        return build_empty_borough_change_plot("Borough boundaries are not available.")

    def attach_boroughs_to_subgrid(grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Spatially join sub-grid points to borough polygons.

        Creates point geometries from lat/lon and performs spatial join to find which
        borough contains each point. Points outside all boroughs are labeled "Unassigned".

        Args:
            grid_df: DataFrame with lat/lon columns

        Returns:
            DataFrame with added "borough" column containing borough name or "Unassigned"
        """
        # Create point geometries from latitude/longitude coordinates
        points_gdf = gpd.GeoDataFrame(
            grid_df[["lat", "lon", *CLASS_COLS]].copy(),
            geometry=gpd.points_from_xy(grid_df["lon"], grid_df["lat"]),
            crs="EPSG:4326",
            index=grid_df.index,
        )

        # Spatial join: find borough polygon containing each point
        joined = gpd.sjoin(
            points_gdf,
            boundaries[["borough", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # Remove geometry columns and fill NA borough names
        out = joined.drop(columns=["geometry", "index_right"], errors="ignore")
        out["borough"] = out["borough"].fillna("Unassigned")
        return out

    # Perform spatial joins for both selected and future data
    selected_with_borough = attach_boroughs_to_subgrid(selected_subgrid_df)
    future_with_borough = attach_boroughs_to_subgrid(future_subgrid_df)

    # Aggregate class scores by borough (sum all sub-grid values within each borough)
    selected_scores = selected_with_borough.groupby("borough", as_index=False)[
        CLASS_COLS
    ].sum()
    future_scores = future_with_borough.groupby("borough", as_index=False)[
        CLASS_COLS
    ].sum()

    def to_percent(df_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Convert class score sums to percentages (0-100) within each borough.

        Ensures scores are comparable across boroughs despite different grid densities.
        Handles edge case of zero-sum rows by returning 0% for all classes.

        Args:
            df_scores: DataFrame with columns (borough, built_up, vegetation, water)

        Returns:
            Same DataFrame with CLASS_COLS values converted to percentages
        """
        out = df_scores.copy()
        # Compute total score per borough
        sums = out[CLASS_COLS].sum(axis=1)
        # Convert each class to percentage (avoid division by zero)
        for col in CLASS_COLS:
            out[col] = np.where(sums > 1e-12, (out[col] / sums) * 100.0, 0.0)
        return out

    # Convert both selected and future scores to percentages
    selected_pct = to_percent(selected_scores).rename(
        columns={c: f"{c}_selected" for c in CLASS_COLS}
    )
    future_pct = to_percent(future_scores).rename(
        columns={c: f"{c}_future" for c in CLASS_COLS}
    )

    # Merge selected and future data by borough (outer join to capture all boroughs)
    merged = selected_pct.merge(future_pct, on="borough", how="outer")
    if merged.empty:
        return build_empty_borough_change_plot("No borough stats available.")

    # Fill missing percentages with 0 (important for partial data)
    for col in CLASS_COLS:
        merged[f"{col}_selected"] = merged[f"{col}_selected"].fillna(0.0)
        merged[f"{col}_future"] = merged[f"{col}_future"].fillna(0.0)

    # Compute per-class change (positive delta = future_pct - selected_pct, clipped at 0)
    for col in CLASS_COLS:
        delta_col = f"{col}_delta_pp"
        merged[delta_col] = merged[f"{col}_future"] - merged[f"{col}_selected"]
        # Only keep positive deltas; negative deltas become 0
        merged[f"{col}_positive_delta_pp"] = merged[delta_col].clip(lower=0.0)

    # Prepare data for all 10 official boroughs (fill missing with 0)
    positive_cols = [f"{col}_positive_delta_pp" for col in CLASS_COLS]
    full = pd.DataFrame({"borough": NURNBERG_BOROUGH_NAMES}).merge(
        merged[["borough", *positive_cols]],
        on="borough",
        how="left",
    )
    # Fill missing rows (boroughs with no data) with 0
    for col in positive_cols:
        full[col] = full[col].fillna(0.0)

    if full.empty:
        return build_empty_borough_change_plot("No borough change values available.")

    # Create class label and color mappings for visualization
    class_label_map = {
        "built_up": "Built Up",
        "vegetation": "Vegetation",
        "water": "Water",
    }
    class_color_map = {
        "Built Up": LAND_COVER_COLORS.get("Built Up", "#fa0000"),
        "Vegetation": LAND_COVER_COLORS.get("Vegetation", "#006400"),
        "Water": LAND_COVER_COLORS.get("Water", "#0064ff"),
    }

    # Reshape data from wide to long format for Plotly bar chart
    positive_long = full.melt(
        id_vars="borough",
        value_vars=positive_cols,
        var_name="class_metric",
        value_name="positive_delta_pp",
    )
    # Extract class name from metric column (remove "_positive_delta_pp" suffix)
    positive_long["class"] = positive_long["class_metric"].str.replace(
        "_positive_delta_pp", "", regex=False
    )
    # Apply human-readable labels to class names
    positive_long["class"] = positive_long["class"].map(class_label_map)

    # Compute total positive change per borough for text annotation
    borough_totals = full.assign(
        total_positive_change_pp=full[positive_cols].sum(axis=1)
    )

    # Create stacked bar chart
    fig = px.bar(
        positive_long,
        x="borough",
        y="positive_delta_pp",
        color="class",
        color_discrete_map=class_color_map,
        barmode="stack",
    )

    # Enhance hover text with formatted percentage
    fig.update_traces(
        hovertemplate="%{x}<br>%{fullData.name} increase: %{y:.2f}%<extra></extra>",
    )

    # Add text annotation labels above each bar showing total change
    fig.add_scatter(
        x=borough_totals["borough"],
        y=borough_totals["total_positive_change_pp"],
        mode="text",
        text=borough_totals["total_positive_change_pp"].map(lambda v: f"{v:.2f}%"),
        textposition="top center",
        textfont={"size": 12, "color": "#1f2937"},
        hoverinfo="skip",
        showlegend=False,
    )

    # Update layout for professional appearance
    fig.update_layout(
        title="Positive Composition Change Across All 10 Boroughs",
        height=620,
        margin={"r": 20, "t": 60, "l": 20, "b": 20},
        xaxis_title="Borough",
        yaxis_title="Positive composition change (%)",
        legend_title_text="Class",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    # Rotate x-axis labels for readability given long German borough names
    fig.update_xaxes(tickangle=-30)
    # Disable gridlines for cleaner appearance
    fig.update_yaxes(showgrid=False)

    return fig


class PlotSelectionBridge(gr.HTML):
    """Custom Gradio HTML component that emits Plotly selection payloads."""

    def __init__(
        self,
        left_plot_id: str,
        right_plot_id: str,
        borough_plot_id: str | None = None,
        borough_bounds: dict | None = None,
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
        const boroughPlotId = props.borough_plot_id || 'borough_change_plot';
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
            pushPayloadWithGridIds(source, eventKind, gridIds, null);
        }

        function pushPayloadWithGridIds(source, eventKind, gridIds, boroughName) {
            const payload = {
                source: source,
                event_kind: eventKind,
                grid_ids: gridIds,
                borough: boroughName,
                ts: Date.now(),
            };
            props.value = payload;
            if (statusEl) {
                if (gridIds.length > 0) {
                    const boroughPrefix = boroughName ? `${boroughName}: ` : '';
                    statusEl.textContent = `${boroughPrefix}${source}: ${gridIds.length} selected (${gridIds.slice(0, 10).join(', ')}${gridIds.length > 10 ? ', ...' : ''})`;
                } else {
                    statusEl.textContent = `${source}: selection cleared`;
                }
            }
        }

        function setSelectionOnMap(targetDiv, eventKind, selectedIds) {
            if (!targetDiv) return;
            const targetState = ensureSyncState(targetDiv);
            if (targetState.applyingSelection) {
                return;
            }

            targetState.applyingSelection = true;
            const traces = Array.isArray(targetDiv.data) ? targetDiv.data : [];
            const traceIndices = traces.map((_, i) => i);
            if (!traceIndices.length) {
                targetState.applyingSelection = false;
                return;
            }

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

        function collectGridIdsForBorough(plotlyDiv, boroughName) {
            if (!plotlyDiv || !boroughName) return [];
            const ids = [];
            const traces = Array.isArray(plotlyDiv.data) ? plotlyDiv.data : [];
            for (const trace of traces) {
                const customData = trace?.customdata;
                if (!Array.isArray(customData)) continue;
                for (const row of customData) {
                    if (!Array.isArray(row) || row.length < 4) continue;
                    const rowBorough = row[3];
                    if (rowBorough !== boroughName) continue;
                    const id = asInt(row[0]);
                    if (id !== null) ids.push(id);
                }
            }
            return Array.from(new Set(ids));
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

        function attachBoroughHandler(plotContainer) {
            if (!plotContainer) {
                return;
            }
            const plotlyDiv = plotContainer.querySelector('.js-plotly-plot');
            if (!plotlyDiv || typeof plotlyDiv.on !== 'function') {
                return;
            }

            if (plotContainer.__boroughBridgeBoundTo === plotlyDiv) {
                return;
            }

            plotlyDiv.on('plotly_click', (eventData) => {
                const point = eventData?.points?.[0];
                const boroughName = (point?.x ?? point?.y ?? '').toString();
                if (!boroughName) return;

                const leftDiv = getPlotlyDiv(leftPlotId);
                const rightDiv = getPlotlyDiv(rightPlotId);
                const selectedIds = Array.from(
                    new Set([
                        ...collectGridIdsForBorough(leftDiv, boroughName),
                        ...collectGridIdsForBorough(rightDiv, boroughName),
                    ])
                );

                setSelectionOnMap(leftDiv, 'selected', selectedIds);
                setSelectionOnMap(rightDiv, 'selected', selectedIds);

                // Zoom to borough on both maps
                zoomToBorough(leftDiv, boroughName);
                zoomToBorough(rightDiv, boroughName);

                pushPayloadWithGridIds('borough_change_plot', 'borough_click', selectedIds, boroughName);
            });

            plotContainer.__boroughBridgeBoundTo = plotlyDiv;
        }

        function zoomToBorough(plotlyDiv, boroughName) {
            if (!plotlyDiv || !boroughName) return;
            const boroughBounds = props.borough_bounds && props.borough_bounds[boroughName];
            if (!boroughBounds) return;

            const relayout = {
                'map.center': {
                    lat: boroughBounds.center_lat,
                    lon: boroughBounds.center_lon,
                },
                'map.zoom': boroughBounds.zoom,
                'mapbox.center': {
                    lat: boroughBounds.center_lat,
                    lon: boroughBounds.center_lon,
                },
                'mapbox.zoom': boroughBounds.zoom,
            };

            Plotly.relayout(plotlyDiv, relayout).catch(() => {
                // Fail silently if relayout doesn't work
            });
        }

        function tryAttachAll() {
            const leftContainer = document.getElementById(leftPlotId);
            const rightContainer = document.getElementById(rightPlotId);
            const boroughContainer = document.getElementById(boroughPlotId);
            attachHandlers(leftContainer, 'selected_year', () => getPlotlyDiv(rightPlotId));
            attachHandlers(rightContainer, 'future_prediction', () => getPlotlyDiv(leftPlotId));
            attachBoroughHandler(boroughContainer);
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
            borough_plot_id=borough_plot_id,
            borough_bounds=borough_bounds or {},
            label=label,
            **kwargs,
        )

    def api_info(self):
        return {
            "type": "object",
            "description": "Selection payload with source, event_kind, and grid_ids.",
        }


# ============================================================================
# DATA LOADING AND UTILITY FUNCTIONS
# ============================================================================


def load_data_from_csv(data_path: str) -> gpd.GeoDataFrame:
    """
    Load satellite imagery data from parquet file and prepare for analysis.

    Reads the parquet file, parses geometry from GeoJSON, creates a GeoDataFrame,
    and computes derived vegetation aggregate and spectral indices (NDVI, EVI2, etc.)
    for use in the prediction model.

    Processing steps:
    1. Read parquet file into DataFrame
    2. Parse .geo column (GeoJSON strings) into Shapely geometries
    3. Create GeoDataFrame in projected CRS (UTM Zone 32N)
    4. Create unique grid IDs and compute centroids
    5. Convert to WGS84 for map rendering
    6. Aggregate vegetation sub-classes into single vegetation column
    7. Compute spectral indices from Sentinel-2 bands (B3, B4, B8, B11)

    Args:
        data_path: Path to parquet file (e.g., "data_3x3/delta_table_2021_3x3.parquet")

    Returns:
        GeoDataFrame with columns: grid_id, lat, lon, vegetation, NDVI, EVI2, SAVI, NDBI, NDWI, MNDWI, etc.
        Coordinates in WGS84 (EPSG:4326) for Plotly map projection compatibility.

    Raises:
        FileNotFoundError: If parquet file does not exist
        KeyError: If required columns (.geo, B3, B4, B8, B11, tree_cover, etc.) are missing
    """
    # 1. Load entire dataset at full resolution (no pre-aggregation)
    df = pd.read_parquet(data_path)

    # 2. Parse GeoJSON geometry strings into Shapely geometry objects
    df["geometry"] = pd.Series(
        [shape(json.loads(x)) for x in df[".geo"]],
        index=df.index,
        dtype="object",
    )

    # 3. Create GeoDataFrame using projected CRS (UTM Zone 32N for Nuremberg, Germany)
    # UTM provides meter-based coordinates for accurate area calculations and grid operations
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32632")

    # 4. Assign unique IDs and compute centroids for rendering
    gdf["grid_id"] = gdf.index
    centroids = gdf.geometry.centroid

    # 5. Convert to WGS84 (EPSG:4326) for Plotly map projection compatibility
    gdf = gdf.to_crs("EPSG:4326")
    centroids = centroids.to_crs("EPSG:4326")

    # Extract latitude/longitude from centroids for point-based visualization
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

    # 6. Aggregate vegetation sub-classes into single vegetation column
    # ESA WorldCover classifies vegetation into: tree_cover, cropland, grassland
    gdf["vegetation"] = gdf[["tree_cover", "cropland", "grassland"]].sum(axis=1)

    # 7. Compute spectral indices from Sentinel-2 bands for use in XGBoost model

    # Normalized Difference Vegetation Index: measures vegetation greenness
    gdf["NDVI"] = (gdf["B8"] - gdf["B4"]) / (gdf["B8"] + gdf["B4"] + 1e-8)

    # Enhanced Vegetation Index (2-band): more sensitive to vegetation than NDVI
    gdf["EVI2"] = 2.5 * (
        (gdf["B8"] - gdf["B4"]) / (gdf["B8"] + 2.4 * gdf["B4"] + 1 + 1e-8)
    )

    # Soil-Adjusted Vegetation Index: reduces soil background effects
    gdf["SAVI"] = ((gdf["B8"] - gdf["B4"]) * 1.5) / (gdf["B8"] + gdf["B4"] + 0.5 + 1e-8)

    # Normalized Difference Built-up Index: highlights built-up/urban areas
    gdf["NDBI"] = (gdf["B11"] - gdf["B8"]) / (gdf["B11"] + gdf["B8"] + 1e-8)

    # Normalized Difference Water Index: identifies water bodies
    gdf["NDWI"] = (gdf["B3"] - gdf["B8"]) / (gdf["B3"] + gdf["B8"] + 1e-8)

    # Modified Normalized Difference Water Index: alternative water detection
    gdf["MNDWI"] = (gdf["B3"] - gdf["B11"]) / (gdf["B3"] + gdf["B11"] + 1e-8)

    return gdf


def map_class_to_string(cls: int) -> str:
    """
    Convert class index to human-readable label.

    Args:
        cls: Integer index (0, 1, or 2) corresponding to class_cols order

    Returns:
        Human-readable class name (e.g., "Built Up"), or "unclassified" if index invalid
    """
    try:
        # Map class index to column name, then format as title case with spaces
        return str.join(" ", CLASS_COLS[cls].lower().split("_")).title()
    except IndexError:
        return "unclassified"


def assign_group_dominant_class(
    df: pd.DataFrame,
    class_columns: list[str],
    group_columns: tuple[str, str] = ("grid_x", "grid_y"),
) -> pd.DataFrame:
    """
    Assign one dominant class per grid group based on summed class scores.

    Groups rows by grid coordinates and determines the class with highest sum within each group.
    Adds "Dominant Class" column to output DataFrame.

    Args:
        df: Input DataFrame with class_columns and group_columns
        class_columns: List of column names containing class scores (e.g., ["built_up", "vegetation", "water"])
        group_columns: Column names to group by (default: grid coordinates)

    Returns:
        DataFrame with added "Dominant Class" column

    Raises:
        ValueError: If required columns are missing from input DataFrame
    """
    # Validate that all required columns exist
    missing_cols = [
        col for col in [*group_columns, *class_columns] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for dominant class assignment: {missing_cols}"
        )

    # Aggregate class scores by group (sum scores within each grid cell)
    grouped_scores = df.groupby(list(group_columns), dropna=False)[class_columns].sum()

    # Find index of maximum score for each group
    dominant_idx = np.argmax(grouped_scores.to_numpy(), axis=1)

    # Create result dataframe with group keys and dominant class label
    dominant_per_group = grouped_scores.reset_index()[list(group_columns)]
    dominant_per_group["Dominant Class"] = [
        map_class_to_string(idx) for idx in dominant_idx
    ]

    # Merge dominant class back to original dataframe
    df_out = df.drop(columns=["Dominant Class"], errors="ignore").merge(
        dominant_per_group, on=list(group_columns), how="left"
    )
    return df_out


def assign_row_dominant_class(
    df: pd.DataFrame,
    class_columns: list[str],
    output_column: str = "Subgrid Dominant Class",
) -> pd.DataFrame:
    """
    Assign dominant class for each individual row using per-row class scores.

    Determines the class with highest score in each row (used for sub-grid level labels).
    Adds specified output_column to DataFrame.

    Args:
        df: Input DataFrame with class_columns
        class_columns: List of column names containing class scores
        output_column: Name of column to add with dominant class labels (default: "Subgrid Dominant Class")

    Returns:
        DataFrame with added output_column containing dominant class labels

    Raises:
        ValueError: If required columns are missing from input DataFrame
    """
    # Validate required columns
    missing_cols = [col for col in class_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for row dominant class assignment: {missing_cols}"
        )

    # Get class scores and find maximum for each row
    row_scores = df[class_columns].to_numpy()
    dominant_idx = np.argmax(row_scores, axis=1)

    # Add dominant class column
    out = df.copy()
    out[output_column] = [map_class_to_string(idx) for idx in dominant_idx]
    return out


def normalize_class_scores(
    df: pd.DataFrame,
    class_columns: list[str],
) -> pd.DataFrame:
    """
    Normalize class score columns row-wise so they are non-negative and sum to 1.

    Clips negative values to 0 and scales each row so class scores sum to 1.0 (100%).
    Handles edge case where all scores are near-zero by assigning equal (1/n) weight.

    Args:
        df: Input DataFrame with class score columns
        class_columns: Column names to normalize

    Returns:
        DataFrame with normalized class_columns (non-negative, row-wise sum = 1.0)

    Raises:
        ValueError: If required columns are missing from input DataFrame
    """
    # Validate required columns
    missing_cols = [col for col in class_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for score normalization: {missing_cols}"
        )

    out = df.copy()

    # Extract scores as numpy array for efficient computation
    scores = out[class_columns].astype(float).to_numpy()

    # Clip negative values to 0
    scores = np.clip(scores, a_min=0.0, a_max=None)

    # Compute row sums and normalize
    row_sums = scores.sum(axis=1, keepdims=True)

    # Handle rows where sum is near-zero (assign equal weight to all classes)
    zero_sum_mask = row_sums.squeeze(axis=1) <= 1e-12
    if np.any(zero_sum_mask):
        scores[zero_sum_mask] = 1.0 / len(class_columns)
        row_sums = scores.sum(axis=1, keepdims=True)

    # Normalize: divide each score by row sum
    normalized = scores / row_sums
    out[class_columns] = normalized
    return out


# ============================================================================
# MODEL LOADING AND GLOBAL DATA INITIALIZATION
# ============================================================================


def load_prediction_model(
    model_path: str = "artifacts/XGBoost_delta.pkl",
) -> object | None:
    """
    Load XGBoost prediction model from pickle file.

    Attempts to load a pre-trained XGBoost model (presumably trained to predict
    land-cover deltas between years). Used for forecasting future land-cover
    compositions.

    Args:
        model_path: Path to pickled model file (default: "artifacts/XGBoost_delta.pkl")

    Returns:
        Loaded model object, or None if file not found or loading fails.
        Silently returns None on errors (model is optional; app can still run without it).
    """
    model_file = Path(model_path)

    # Return None if model file doesn't exist
    if not model_file.exists():
        return None

    try:
        with model_file.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load prediction model from {model_path}: {e}")
        return None


# Load model at module initialization
prediction_model = load_prediction_model()

# Load satellite data at module initialization
try:
    gdf = load_data_from_csv("data_3x3/delta_table_2021_3x3.parquet")
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
    gdf = None


# Comprehensive color mapping for all potential land-cover classes
# Used for consistent color assignment across all visualizations
COLOR_MAP: dict[str, str] = {
    "Tree Cover": "#006400",  # Dark green
    "Vegetation": "#006400",  # Dark green
    "Shrubland": "#ffbb22",  # Golden yellow
    "Grassland": "#ffff4c",  # Bright yellow
    "Cropland": "#f096ff",  # Magenta
    "Built Up": "#fa0000",  # Red
    "Bare / Sparse veg.": "#b4b4b4",  # Gray
    "Snow and Ice": "#f0f0f0",  # Light gray
    "Water": "#0064ff",  # Blue
    "Permanent Water": "#0064ff",  # Blue
    "Herbaceous wetland": "#0096a0",  # Teal
    "Mangroves": "#00cf75",  # Bright green
    "Moss and Lichen": "#fae6a0",  # Pale yellow
    "unclassified": "#2c3e50",  # Dark blue-gray
}


# ============================================================================
# MAIN DASHBOARD COMPUTATION PIPELINE
# ============================================================================


def update_dashboard(
    start_year: int,
    time_delta: int,
    map_type: str,
    render_mode: str,
    grid_cell_size: int,
) -> tuple[go.Figure | None, go.Figure | None, go.Figure]:
    """
    Build two synchronized maps showing selected year and future predictions.

    Main entry point for dashboard computation. Orchestrates the entire pipeline:
    1. Filter base data by selected year (using inverted delta logic)
    2. Create grid cells at specified resolution
    3. Build or predict land-cover classifications
    4. Generate map visualizations with confidence metrics
    5. Compute borough-level composition changes
    6. Cache results for downstream selection interactions

    The delta calculation uses inverted logic: delta_selection = 2021 - start_year
    This is necessary because all data is sourced from a 2021 parquet file:
    - start_year=2021 → delta_selection=0 (use actual 2021 data)
    - start_year=2020 → delta_selection=1 (predict back to 2020)
    - start_year=2016 → delta_selection=5 (predict back to 2016)

    Args:
        start_year: Base year for visualization (2016-2021)
        time_delta: Years ahead to predict (0-4)
        map_type: Map style for Plotly (e.g., "carto-voyager" or "satellite-streets")
        render_mode: Either "points" (scatter) or "polygons" (choropleth) for grid rendering
        grid_cell_size: Grid cell size in meters (20-200m)

    Returns:
        Tuple of (selected_map_fig, future_map_fig, borough_change_fig) where:
        - selected_map_fig: Plotly map showing selected year land-cover
        - future_map_fig: Plotly map showing predicted future land-cover
        - borough_change_fig: Stacked bar chart of borough-level changes

        Returns (None, None, empty_figure) if base data unavailable.

    Side Effects:
        Updates global caches: LAST_GRID_TABLES, LAST_SUBGRID_TABLES,
        LAST_BOROUGH_CHANGE_FIG, LAST_MAP_FIGURES
    """
    # Validate base data availability
    if gdf is None:
        return (
            None,
            None,
            build_empty_borough_change_plot("No grid data available yet."),
        )

    # Convert parameters to integers for safety
    start_year = int(start_year)
    time_delta = int(time_delta)
    grid_cell_size = int(grid_cell_size)

    # Set random seed for reproducibility of any stochastic operations
    np.random.seed(start_year)

    # Calculate delta for data filtering using inverted delta logic
    # (All data sourced from 2021, so delta=0 means 2021, delta=1 means 2020, etc.)
    delta_selection = 2021 - start_year

    # Filter base data to rows matching the selected year's delta
    base_df = gdf[gdf["delta_years"].astype(int) == delta_selection].copy()

    # Compute grid cell coordinates for all data points
    # Each point is assigned to its containing grid cell based on grid_cell_size
    base_df["grid_x"] = (
        np.floor(base_df["x"] / grid_cell_size) * grid_cell_size
    ).astype(int)
    base_df["grid_y"] = (
        np.floor(base_df["y"] / grid_cell_size) * grid_cell_size
    ).astype(int)

    # Create stable grid metadata (unique grid cells and assigned IDs)
    # Grid geometry is created lazily only when needed for choropleth rendering
    grid_metadata = (
        base_df[["delta_years", "grid_x", "grid_y"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid_metadata["grid_id"] = np.arange(len(grid_metadata), dtype=int)

    # Lazy-loaded GeoJSON for choropleth mode (created only when render_mode="polygons")
    plotly_geojson = None

    def get_plotly_geojson() -> dict:
        """
        Generate GeoJSON representation of grid cells for choropleth mapping.

        Creates rectangular polygons in projected CRS, then converts to WGS84 for
        Plotly compatibility. Cached on first call to avoid expensive recomputation.

        Returns:
            GeoJSON dict mapping grid_ids to polygon geometries
        """
        nonlocal plotly_geojson
        if plotly_geojson is None:
            grid_with_geometry = grid_metadata.copy()
            # Create bounding box polygons for each grid cell
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
            # Convert to WGS84 for Plotly projection compatibility
            display_grid_gdf = display_grid_gdf.to_crs("EPSG:4326")
            plotly_geojson = json.loads(display_grid_gdf.to_json())
        return plotly_geojson

    # Extract feature names from prediction model
    # Attempts to find feature names from either the model directly or from its sub-estimators
    feature_names = None
    if prediction_model is not None:
        feature_names = None
        # Try to get feature names from primary model attribute (sklearn models)
        if hasattr(prediction_model, "feature_names_in_"):
            feature_names = prediction_model.feature_names_in_.tolist()
        # Fallback: try to get from first estimator if model is an ensemble
        elif (
            hasattr(prediction_model, "estimators_")
            and len(prediction_model.estimators_) > 0
            and hasattr(prediction_model.estimators_[0], "feature_names_in_")
        ):
            feature_names = prediction_model.estimators_[0].feature_names_in_.tolist()

    def build_predicted_df(prediction_delta: int) -> pd.DataFrame:
        """
        Generate predictions and dominant class labels for a future time point.

        Creates a copy of base data with updated delta_years, runs XGBoost predictions
        if available, normalizes outputs, and assigns dominant classes at both row
        and group levels.

        Args:
            prediction_delta: Delta value for future year (0-4)

        Returns:
            DataFrame with predicted CLASS_COLS, dominant class labels, and metadata
        """
        df_out = base_df.copy()
        df_out["delta_years"] = prediction_delta

        # Run model predictions if model and features are available
        if prediction_model is not None and feature_names is not None:
            # Check for missing features before prediction
            missing_features = [
                name for name in feature_names if name not in df_out.columns
            ]
            # Only proceed if all required features are present
            if not missing_features:
                # Suppress XGBoost warning about GPU/CUDA device fallback
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*Falling back to prediction using DMatrix.*",
                    )
                    predicted_targets = prediction_model.predict(df_out[feature_names])

                # Convert predictions to DataFrame with class column names
                predicted_targets_df = pd.DataFrame(
                    predicted_targets, index=df_out.index, columns=CLASS_COLS
                )
                # Normalize predictions to sum to 1.0 per row
                predicted_targets_df = normalize_class_scores(
                    predicted_targets_df,
                    CLASS_COLS,
                )
                # Replace original class scores with normalized predictions
                df_out[CLASS_COLS] = predicted_targets_df[CLASS_COLS]

        # Assign dominant classes at row and group levels
        df_out = assign_row_dominant_class(df_out, CLASS_COLS)
        df_out = assign_group_dominant_class(df_out, CLASS_COLS)
        # Parent dominant class (grid-level) matches group dominant class
        df_out["Parent Dominant Class"] = df_out["Dominant Class"]
        return df_out

    def build_subgrid_table(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Build sub-grid level output table with all required columns and metadata.

        Joins sub-grid level data with parent grid metadata, ensures all columns
        are present, and formats for downstream consumption.

        Args:
            df_in: Input DataFrame with grid/subgrid data and class scores

        Returns:
            DataFrame with columns from SUBGRID_COLUMNS
        """
        # Merge with grid metadata to assign parent_grid_id
        subgrid_df = df_in.merge(
            grid_metadata[["grid_x", "grid_y", "grid_id"]],
            on=["grid_x", "grid_y"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"grid_id_x": "subgrid_id", "grid_id_y": "parent_grid_id"})

        # Backfill missing dominant class columns for compatibility
        if (
            "Parent Dominant Class" not in subgrid_df.columns
            and "Dominant Class" in subgrid_df.columns
        ):
            subgrid_df["Parent Dominant Class"] = subgrid_df["Dominant Class"]
        if "Subgrid Dominant Class" not in subgrid_df.columns:
            subgrid_df = assign_row_dominant_class(
                subgrid_df,
                CLASS_COLS,
                output_column="Subgrid Dominant Class",
            )

        # Ensure all expected columns are present (fill with NaN if missing)
        for col in SUBGRID_COLUMNS:
            if col not in subgrid_df.columns:
                subgrid_df[col] = np.nan

        # Return only required columns in proper order
        return subgrid_df[SUBGRID_COLUMNS].copy()

    def build_map(df_in: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
        """
        Build interactive Plotly map with grid cells and metadata.

        Aggregates sub-grid predictions to parent grid level, computes confidence
        estimates, performs spatial join to boroughs, and creates Plotly visualization.

        Args:
            df_in: Sub-grid level DataFrame with CLASS_COLS scores

        Returns:
            Tuple of (Plotly figure, grid-level output DataFrame)
        """
        # Aggregate class scores by grid cell (parent grid level)
        grouped_scores = df_in.groupby(["grid_x", "grid_y"], as_index=False)[
            CLASS_COLS
        ].sum()
        grid_df = grouped_scores.copy()

        # Assign dominant class at grid level
        dominant_idx = np.argmax(grid_df[CLASS_COLS].to_numpy(), axis=1)
        grid_df["Dominant Class"] = [map_class_to_string(idx) for idx in dominant_idx]

        # Compute confidence as percentage of dominant class
        # Confidence ranges from 50% (all classes equal) to 100% (single class dominates)
        score_matrix = grid_df[CLASS_COLS].to_numpy()
        dominant_values = score_matrix[np.arange(len(grid_df)), dominant_idx]
        total_values = score_matrix.sum(axis=1)
        dominant_pct = np.where(
            total_values > 1e-12,
            (dominant_values / total_values) * 100.0,
            50.0,
        )
        # Confidence categories (Low: 50-66%, Medium: 66-83%, High: 83-100%)
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

        # Compute average lat/lon for each grid cell (used for point rendering)
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

        # Merge with grid metadata to get grid_id
        grid_df = grid_df.merge(
            grid_metadata[["grid_x", "grid_y", "grid_id"]],
            on=["grid_x", "grid_y"],
            how="left",
            validate="one_to_one",
        )

        # Perform spatial join to borough polygons
        boundaries = BOROUGH_BOUNDARIES_GDF
        if boundaries is not None and not boundaries.empty:
            points_gdf = gpd.GeoDataFrame(
                grid_df[["lat", "lon"]].copy(),
                geometry=gpd.points_from_xy(grid_df["lon"], grid_df["lat"]),
                crs="EPSG:4326",
                index=grid_df.index,
            )
            joined = gpd.sjoin(
                points_gdf,
                boundaries[["borough", "geometry"]],
                how="left",
                predicate="intersects",
            )
            grid_df["Borough"] = joined["borough"].fillna("Unassigned").to_numpy()
        else:
            grid_df["Borough"] = "Unassigned"

        # Create Plotly visualization based on render mode
        if render_mode == "points":
            # Scatter map mode: fast rendering with point markers
            fig = px.scatter_map(
                grid_df,
                lat="lat",
                lon="lon",
                color="Dominant Class",
                custom_data=["grid_id", "Dominant Class", "Confidence", "Borough"],
                color_discrete_map=COLOR_MAP,
                hover_name=None,
                hover_data={"grid_id": False, "lat": False, "lon": False},
                zoom=11,
                center={"lat": LAT_CENTER, "lon": LON_CENTER},
                map_style=map_type,
            )
        else:
            # Choropleth map mode: detailed polygon rendering at grid resolution
            fig = px.choropleth_map(
                grid_df,
                geojson=get_plotly_geojson(),
                locations="grid_id",
                featureidkey="properties.grid_id",
                color="Dominant Class",
                custom_data=["grid_id", "Dominant Class", "Confidence", "Borough"],
                color_discrete_map=COLOR_MAP,
                hover_name=None,
                hover_data={"grid_id": False},
                zoom=11,
                center={"lat": LAT_CENTER, "lon": LON_CENTER},
                map_style=map_type,
            )

        # Apply consistent layout styling
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
            clickmode="event+select",  # Enable box select and click events
            height=700,
        )

        # Apply trace-specific styling
        if render_mode == "points":
            # Point markers: semi-transparent, hide when unselected
            fig.update_traces(
                hovertemplate="Dominant Class: %{customdata[1]}<br>Confidence: %{customdata[2]}<extra></extra>",
                marker=dict(size=5, opacity=0.4),
                selected=dict(marker=dict(opacity=0.4)),
                unselected=dict(marker=dict(opacity=0.0)),
            )
        else:
            # Polygon markers: semi-transparent fill, hide background when unselected
            fig.update_traces(
                hovertemplate="Dominant Class: %{customdata[1]}<br>Confidence: %{customdata[2]}<extra></extra>",
                marker=dict(opacity=0.3),
                selected=dict(marker=dict(opacity=0.3)),
                unselected=dict(marker=dict(opacity=0.0)),
            )

        return fig, grid_df

    # Build prediction dataframes for selected year and future prediction
    if delta_selection == 0:
        # If selected year is 2021 (delta=0), use actual data for selected year
        # and predictions only for future
        predicted_dfs = [
            assign_group_dominant_class(base_df.copy(), CLASS_COLS),
            build_predicted_df(prediction_delta=time_delta),
        ]
    else:
        # If selected year is historical, predict both selected year and future
        predicted_dfs = [
            build_predicted_df(prediction_delta=0),
            build_predicted_df(prediction_delta=time_delta),
        ]

    # Build both maps and extract grid-level aggregates
    (selected_fig, selected_grid_df), (future_fig, future_grid_df) = [
        build_map(df_item) for df_item in predicted_dfs
    ]

    # Update global caches for downstream interaction handlers
    global \
        LAST_GRID_TABLES, \
        LAST_SUBGRID_TABLES, \
        LAST_BOROUGH_CHANGE_FIG, \
        LAST_MAP_FIGURES

    LAST_GRID_TABLES = {
        "selected_year": selected_grid_df[SELECTION_COLUMNS].copy(),
        "future_prediction": future_grid_df[SELECTION_COLUMNS].copy(),
    }

    # Build sub-grid tables for composition analysis
    selected_subgrid = build_subgrid_table(predicted_dfs[0])
    future_subgrid = build_subgrid_table(predicted_dfs[1])
    LAST_SUBGRID_TABLES = {
        "selected_year": selected_subgrid,
        "future_prediction": future_subgrid,
    }

    # Build borough-level composition change chart
    borough_change_fig = build_top_changed_boroughs_chart(
        selected_subgrid,
        future_subgrid,
    )
    LAST_BOROUGH_CHANGE_FIG = borough_change_fig

    # Cache map figures for deselect operations
    LAST_MAP_FIGURES = {
        "selected_year": go.Figure(selected_fig),
        "future_prediction": go.Figure(future_fig),
    }

    return selected_fig, future_fig, borough_change_fig


def selection_payload_to_outputs(payload: dict | None):
    def hidden_plot_update():
        return gr.update(value=None, visible=False)

    def visible_plot_update(fig: go.Figure):
        return gr.update(value=fig, visible=True)

    def visible_borough_or_placeholder():
        fig = LAST_BOROUGH_CHANGE_FIG
        if fig is None:
            return gr.update(value=None, visible=True)
        return gr.update(value=fig, visible=True)

    def selected_empty_update(message: str):
        return gr.update(
            value=build_empty_pie("Selected Year Composition", message),
            visible="hidden",
        )

    def future_empty_update(message: str):
        return gr.update(
            value=build_empty_pie("Future Prediction Composition", message),
            visible="hidden",
        )

    def delta_empty_update(message: str):
        return gr.update(
            value=build_empty_delta("Land-Cover Composition Delta", message),
            visible="hidden",
        )

    def build_land_cover_composition(subgrid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sub-grid land-cover composition into percentage breakdown.

        Sums class scores across all sub-grids within selection and converts to
        percentages for human-readable composition display.

        Args:
            subgrid_df: Sub-grid level data with CLASS_COLS scores

        Returns:
            DataFrame with columns: class (label), value (sum), percent (%)
        """
        if subgrid_df.empty:
            return pd.DataFrame(columns=["class", "value", "percent"])

        # Sum class scores across all selected sub-grids
        values = subgrid_df[CLASS_COLS].sum(axis=0).clip(lower=0)
        total = float(values.sum())
        if total <= 0:
            return pd.DataFrame(columns=["class", "value", "percent"])

        # Build composition DataFrame with human-readable labels
        comp_df = pd.DataFrame(
            {
                "class": [LAND_COVER_LABELS[col] for col in CLASS_COLS],
                "value": [float(values[col]) for col in CLASS_COLS],
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
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
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
            selected_empty_update("Waiting for selection"),
            future_empty_update("Waiting for selection"),
            delta_empty_update("Waiting for selection"),
            visible_borough_or_placeholder(),
        )

    payload_ts = payload.get("ts")
    event_kind = payload.get("event_kind")
    borough_name = payload.get("borough")
    try:
        payload_ts = int(payload_ts) if payload_ts is not None else None
    except Exception:
        payload_ts = None

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

    # Ignore only empty bridge events emitted during plot re-render right after submit.
    # Keep non-empty events so the very first real user selection is not dropped.
    if (
        event_kind != "borough_click"
        and payload_ts is not None
        and payload_ts < (LAST_SUBMIT_TS_MS + SELECTION_EVENT_GUARD_MS)
        and not grid_ids
    ):
        return (
            DEFAULT_SELECTION_MESSAGE,
            selected_empty_update("Waiting for selection"),
            future_empty_update("Waiting for selection"),
            delta_empty_update("Waiting for selection"),
            visible_borough_or_placeholder(),
        )

    if not grid_ids:
        return (
            DEFAULT_SELECTION_MESSAGE,
            selected_empty_update("Selection cleared"),
            future_empty_update("Selection cleared"),
            delta_empty_update("Selection cleared"),
            visible_borough_or_placeholder(),
        )

    selected_subgrid_table = LAST_SUBGRID_TABLES.get("selected_year")
    future_subgrid_table = LAST_SUBGRID_TABLES.get("future_prediction")
    if selected_subgrid_table is None or selected_subgrid_table.empty:
        return (
            "Selected-year sub-grid data is not available yet. Click Submit and select a region.",
            selected_empty_update("No selected-year sub-grid data"),
            future_empty_update("No future sub-grid data"),
            delta_empty_update("No selected sub-grid data"),
            visible_borough_or_placeholder(),
        )
    if future_subgrid_table is None or future_subgrid_table.empty:
        return (
            "Future-year sub-grid data is not available yet. Click Submit and select a region.",
            selected_empty_update("No selected-year sub-grid data"),
            future_empty_update("No future sub-grid data"),
            delta_empty_update("No selected sub-grid data"),
            visible_borough_or_placeholder(),
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

    summary_prefix = f"Selection ({borough_name}): " if borough_name else "Selection: "
    summary = (
        f"{summary_prefix}{len(grid_ids)} parent grids | "
        f"{len(selected_subgrid_rows)} selected-year sub-grids | "
        f"{len(future_subgrid_rows)} future sub-grids."
    )
    return (
        summary,
        visible_plot_update(selected_pie),
        visible_plot_update(future_pie),
        visible_plot_update(delta_plot),
        hidden_plot_update(),
    )


def reset_selection_insights() -> tuple:
    """
    Reset selection state by clearing all composition analysis plots.

    Called when user needs to see default empty state. Simply delegates to
    `selection_payload_to_outputs` with None payload.

    Returns:
        Tuple of all secondary plot outputs (pies, delta, borough chart) in empty state.
    """
    return selection_payload_to_outputs(None)


def clear_selection_without_recompute() -> tuple:
    """
    Clear grid selection on maps without recomputing dashboard or regenerating data.

    Used by Deselect button: clears selectedPoints from both maps and resets secondary
    plots to empty state. More efficient than full re-submission as it uses cached
    map figures.

    Returns:
        Tuple of map updates, selection summary, and secondary plots in empty state.
    """

    def clear_selectedpoints(fig: go.Figure | None) -> dict:
        """
        Clear selected points from a Plotly figure.

        Args:
            fig: Plotly figure or None

        Returns:
            Gradio update object (dict)
        """
        if fig is None:
            return gr.update()
        fig_out = go.Figure(fig)
        fig_out.update_traces(selectedpoints=None)
        return gr.update(value=fig_out)

    # Clear selections from both cached maps
    left_map_update = clear_selectedpoints(LAST_MAP_FIGURES.get("selected_year"))
    right_map_update = clear_selectedpoints(LAST_MAP_FIGURES.get("future_prediction"))

    # Reset all secondary plots to empty state
    summary, selected_pie_update, future_pie_update, delta_update, borough_update = (
        selection_payload_to_outputs({"event_kind": "deselect", "grid_ids": []})
    )

    return (
        left_map_update,
        right_map_update,
        {},
        summary,
        selected_pie_update,
        future_pie_update,
        delta_update,
        borough_update,
    )


def build_map_titles(start_year: int, time_delta: int) -> tuple[str, str]:
    """
    Generate descriptive titles for selected year and future prediction maps.

    Indicates whether the selected year map shows true labels (2021) or predictions (historical years).

    Args:
        start_year: Selected year for visualization
        time_delta: Prediction horizon in years

    Returns:
        Tuple of (selected_year_title, future_year_title) as markdown-formatted strings
    """
    start_year = int(start_year)
    time_delta = int(time_delta)

    # Only 2021 has true/validated labels; all other years are predictions
    selected_year_label_type = (
        "True Labels" if start_year == 2021 else "Predicted Labels"
    )
    selected_title = f"### {start_year} Land-Cover ({selected_year_label_type})"
    future_title = f"### {start_year + time_delta} Land-Cover (Predicted)"
    return selected_title, future_title


def update_dashboard_with_titles(
    start_year: int,
    time_delta: int,
    map_type: str,
    render_mode: str,
    grid_cell_size: int,
) -> tuple[go.Figure | None, go.Figure | None, go.Figure, str, str]:
    """
    Wrapper around update_dashboard that generates map titles.

    Calls update_dashboard and then builds corresponding titles based on selected year.

    Args:
        start_year: Selected year for visualization
        time_delta: Prediction horizon in years
        map_type: Map style (carto-voyager, satellite-streets, etc.)
        render_mode: Rendering mode (points or polygons)
        grid_cell_size: Grid resolution in meters

    Returns:
        Tuple of (selected_fig, future_fig, borough_fig, left_title, right_title)
    """
    selected_fig, future_fig, borough_change_fig = update_dashboard(
        start_year=start_year,
        time_delta=time_delta,
        map_type=map_type,
        render_mode=render_mode,
        grid_cell_size=grid_cell_size,
    )
    left_title, right_title = build_map_titles(start_year, time_delta)
    return selected_fig, future_fig, borough_change_fig, left_title, right_title


def submit_all_outputs(
    start_year: int,
    time_delta: int,
    map_type: str,
    render_mode: str,
    grid_cell_size: int,
) -> tuple:
    """
    Process Submit button click and generate all dashboard outputs.

    Main entry point from Gradio UI called on form submission. Updates dashboard
    computation, resets selection state to empty, and returns all visualizations
    and metadata for display.

    Args:
        start_year: Selected year for visualization (2016-2021)
        time_delta: Prediction horizon in years (0-4)
        map_type: Map style (carto-voyager, satellite-streets, etc.)
        render_mode: Rendering mode (points or polygons)
        grid_cell_size: Grid resolution in meters (20-200)

    Returns:
        Tuple of all Gradio outputs in order:
        (selected_map, future_map, selected_title, future_title, selection_bridge,
         summary, selected_pie, future_pie, delta_plot, borough_chart)
    """
    global LAST_SUBMIT_TS_MS
    # Record timestamp for event guard (filters stale re-render events from Plotly)
    LAST_SUBMIT_TS_MS = int(time.time() * 1000)

    # Update dashboard and get all visualizations with titles
    selected_fig, future_fig, borough_change_fig, left_title, right_title = (
        update_dashboard_with_titles(
            start_year=start_year,
            time_delta=time_delta,
            map_type=map_type,
            render_mode=render_mode,
            grid_cell_size=grid_cell_size,
        )
    )

    # Initialize secondary plots with default (empty selection) state
    summary = DEFAULT_SELECTION_MESSAGE
    selected_pie_update = gr.update(
        value=build_empty_pie("Selected Year Composition", "Waiting for selection"),
        visible="hidden",
    )
    future_pie_update = gr.update(
        value=build_empty_pie("Future Prediction Composition", "Waiting for selection"),
        visible="hidden",
    )
    delta_update = gr.update(
        value=build_empty_delta(
            "Land-Cover Composition Delta", "Waiting for selection"
        ),
        visible="hidden",
    )
    borough_update = gr.update(value=borough_change_fig, visible=True)

    return (
        selected_fig,
        future_fig,
        left_title,
        right_title,
        {},
        summary,
        selected_pie_update,
        future_pie_update,
        delta_update,
        borough_update,
    )


# Gradio UI
with gr.Blocks(fill_height=True) as app:
    gr.Markdown("# 🏙️ Nuremberg Future Land-Cover Prediction")

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
            value=2016,
            label="Start Year Selection",
        )
        time_delta_drop_down = gr.Dropdown(
            choices=[i for i in range(0, 5)],
            value=1,
            label="Future Time (Years)",
        )
        with gr.Column(scale=1, min_width=140):
            submit_button = gr.Button("Submit", variant="primary")
            deselect_button = gr.Button("Deselect", variant="secondary")

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
        borough_plot_id="borough_change_plot",
        borough_bounds=calculate_borough_bounds(),
        label="Selection Bridge",
        visible="hidden",
    )

    with gr.Row():
        with gr.Column():
            selected_year_pie = gr.Plot(
                show_label=False,
                value=build_empty_pie("Selected Year Composition", "Submit to begin"),
                visible="hidden",
            )
        with gr.Column():
            future_year_pie = gr.Plot(
                show_label=False,
                value=build_empty_pie(
                    "Future Prediction Composition", "Submit to begin"
                ),
                visible="hidden",
            )

    with gr.Row():
        delta_plot = gr.Plot(
            show_label=False,
            value=build_empty_delta("Land-Cover Composition Delta", "Submit to begin"),
            visible="hidden",
        )

    with gr.Row():
        borough_change_plot = gr.Plot(
            show_label=False,
            value=None,
            visible=True,
            elem_id="borough_change_plot",
        )

    with gr.Row():
        selection_summary = gr.Markdown(DEFAULT_SELECTION_MESSAGE)

    with gr.Row():
        gr.Markdown("""
            ---
            ### ⚠️ Limitations & Disclosures
            * **Not for Official Use:** This tool is designed for exploratory analysis and high-level insights, and is not intended to replace high-resolution surveys or official land-use assessments.
            * **Grid Resolution Constraints:** Predictions are aggregated at a 20x20 meter grid level, which may obscure fine-grained land-cover patterns. Predictions are generally less reliable in mixed-use transition regions where a single cell contains heterogeneous land cover.
            * **Historical Label Noise:** The model is trained on ESA WorldCover labels from 2020/2021, which have an estimated accuracy of 70-75%. Inherent noise in these satellite-derived labels impacts the model's baseline accuracy.
            * **Seasonal & Temporal Ambiguity:** Short-term seasonal fluctuations (e.g., temporary vegetation changes) can sometimes be mistaken by the model for genuine land-cover conversion. Furthermore, prediction uncertainty is largest significantly for short-term (0 years) and long-term forecasts (4 years out).
            * **Confidence vs. Reality:** The displayed "Confidence" estimate reflects the model's internal mathematical certainty, not necessarily true ground-truth reliability.
            """)

    update_args = {
        "fn": submit_all_outputs,
        "inputs": [
            start_year_dropdown,
            time_delta_drop_down,
            map_type_radio,
            render_mode_radio,
            grid_cell_size_dropdown,
        ],
        "outputs": [
            map_output_left,
            map_output_right,
            left_map_title,
            right_map_title,
            selection_bridge,
            selection_summary,
            selected_year_pie,
            future_year_pie,
            delta_plot,
            borough_change_plot,
        ],
    }
    submit_button.click(**update_args)
    deselect_button.click(
        fn=clear_selection_without_recompute,
        inputs=[],
        outputs=[
            map_output_left,
            map_output_right,
            selection_bridge,
            selection_summary,
            selected_year_pie,
            future_year_pie,
            delta_plot,
            borough_change_plot,
        ],
    )
    selection_bridge.change(
        fn=selection_payload_to_outputs,
        inputs=[selection_bridge],
        outputs=[
            selection_summary,
            selected_year_pie,
            future_year_pie,
            delta_plot,
            borough_change_plot,
        ],
    )

theme = Ocean()
if __name__ == "__main__":
    app.launch(theme=theme)
