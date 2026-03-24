"""Interactive Gradio dashboard for land-cover comparison and prediction in Nuremberg.

This module loads geospatial grid data, prepares model-based future predictions,
renders two synchronized interactive maps, and provides selection-driven insight
plots for land-cover composition changes.
"""

import json
import pickle
import time
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

NURNBERG_BOROUGH_NAMES = [
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

LAST_BOROUGH_CHANGE_FIG = None
LAST_MAP_FIGURES: dict[str, go.Figure | None] = {
    "selected_year": None,
    "future_prediction": None,
}
LAST_SUBMIT_TS_MS = 0
SELECTION_EVENT_GUARD_MS = 900


def build_empty_borough_change_plot(message: str) -> go.Figure:
    """Create a placeholder figure for borough-level change output.

    Args:
        message: Centered message shown when borough insights are unavailable.

    Returns:
        A Plotly figure with fixed layout and a descriptive annotation.
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
    """Create an empty pie-chart placeholder with a custom title and message.

    Args:
        title: Figure title for the placeholder panel.
        message: Centered message shown in the panel body.

    Returns:
        A Plotly figure configured as a placeholder state.
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
    """Create an empty delta-chart placeholder with axes and status text.

    Args:
        title: Figure title for the placeholder panel.
        message: Centered message shown when delta data is unavailable.

    Returns:
        A Plotly bar-chart layout with a centered status annotation.
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


BOROUGH_BOUNDARIES_GDF = load_nuremberg_borough_boundaries()


def calculate_borough_bounds() -> dict[str, dict]:
    """Calculate map bounds and zoom hints for each known borough polygon.

    Returns:
        Dictionary mapping borough names to
        {min_lat, max_lat, min_lon, max_lon, center_lat, center_lon, zoom}.
    """
    bounds = {}
    if BOROUGH_BOUNDARIES_GDF is None or BOROUGH_BOUNDARIES_GDF.empty:
        return bounds

    for _, row in BOROUGH_BOUNDARIES_GDF.iterrows():
        borough_name = row["borough"]
        geom = row["geometry"]
        minx, miny, maxx, maxy = geom.bounds  # (lon_min, lat_min, lon_max, lat_max)

        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

        # Calculate zoom level based on borough size
        # Smaller bounds = higher zoom
        lon_span = maxx - minx
        lat_span = maxy - miny
        max_span = max(lon_span, lat_span)
        # Empirical formula for Plotly zoom levels
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
    """Build a stacked chart of positive land-cover composition change by borough.

    The chart compares selected-year and future composition percentages per borough,
    keeps only positive deltas per class, and stacks class contributions for all
    ten expected boroughs.

    Args:
        selected_subgrid_df: Sub-grid table for the selected year.
        future_subgrid_df: Sub-grid table for the future prediction.

    Returns:
        A Plotly figure containing stacked positive deltas by borough or an
        explanatory placeholder if required inputs are missing.
    """
    if selected_subgrid_df.empty or future_subgrid_df.empty:
        return build_empty_borough_change_plot("No grid data available yet.")

    boundaries = BOROUGH_BOUNDARIES_GDF
    if boundaries is None or boundaries.empty:
        return build_empty_borough_change_plot("Borough boundaries are not available.")

    def attach_boroughs_to_subgrid(grid_df: pd.DataFrame) -> pd.DataFrame:
        """Attach borough names to sub-grid rows via point-in-polygon join."""
        points_gdf = gpd.GeoDataFrame(
            grid_df[["lat", "lon", *class_cols]].copy(),
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
        out = joined.drop(columns=["geometry", "index_right"], errors="ignore")
        out["borough"] = out["borough"].fillna("Unassigned")
        return out

    selected_with_borough = attach_boroughs_to_subgrid(selected_subgrid_df)
    future_with_borough = attach_boroughs_to_subgrid(future_subgrid_df)

    selected_scores = selected_with_borough.groupby("borough", as_index=False)[
        class_cols
    ].sum()
    future_scores = future_with_borough.groupby("borough", as_index=False)[
        class_cols
    ].sum()

    # Convert borough-level sums into percentages so changes are comparable.
    def to_percent(df_scores: pd.DataFrame) -> pd.DataFrame:
        out = df_scores.copy()
        sums = out[class_cols].sum(axis=1)
        for col in class_cols:
            out[col] = np.where(sums > 1e-12, (out[col] / sums) * 100.0, 0.0)
        return out

    selected_pct = to_percent(selected_scores).rename(
        columns={c: f"{c}_selected" for c in class_cols}
    )
    future_pct = to_percent(future_scores).rename(
        columns={c: f"{c}_future" for c in class_cols}
    )

    merged = selected_pct.merge(future_pct, on="borough", how="outer")
    if merged.empty:
        return build_empty_borough_change_plot("No borough stats available.")

    for col in class_cols:
        merged[f"{col}_selected"] = merged[f"{col}_selected"].fillna(0.0)
        merged[f"{col}_future"] = merged[f"{col}_future"].fillna(0.0)

    for col in class_cols:
        delta_col = f"{col}_delta_pp"
        merged[delta_col] = merged[f"{col}_future"] - merged[f"{col}_selected"]
        merged[f"{col}_positive_delta_pp"] = merged[delta_col].clip(lower=0.0)

    positive_cols = [f"{col}_positive_delta_pp" for col in class_cols]
    full = pd.DataFrame({"borough": NURNBERG_BOROUGH_NAMES}).merge(
        merged[["borough", *positive_cols]],
        on="borough",
        how="left",
    )
    for col in positive_cols:
        full[col] = full[col].fillna(0.0)

    if full.empty:
        return build_empty_borough_change_plot("No borough change values available.")

    class_label_map = {
        "built_up": "Built Up",
        "vegetation": "Vegetation",
        "water": "Water",
    }
    class_color_map = {
        "Built Up": color_map.get("Built Up", "#fa0000"),
        "Vegetation": color_map.get("Vegetation", "#006400"),
        "Water": color_map.get("Water", "#0064ff"),
    }
    positive_long = full.melt(
        id_vars="borough",
        value_vars=positive_cols,
        var_name="class_metric",
        value_name="positive_delta_pp",
    )
    positive_long["class"] = positive_long["class_metric"].str.replace(
        "_positive_delta_pp", "", regex=False
    )
    positive_long["class"] = positive_long["class"].map(class_label_map)
    borough_totals = full.assign(
        total_positive_change_pp=full[positive_cols].sum(axis=1)
    )

    fig = px.bar(
        positive_long,
        x="borough",
        y="positive_delta_pp",
        color="class",
        color_discrete_map=class_color_map,
        barmode="stack",
    )
    fig.update_traces(
        hovertemplate="%{x}<br>%{fullData.name} increase: %{y:.2f}%<extra></extra>",
    )
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
    fig.update_xaxes(tickangle=-30)
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
        """Initialize the bridge component and embed JavaScript event wiring.

        Args:
            left_plot_id: DOM element id of the selected-year map container.
            right_plot_id: DOM element id of the future map container.
            borough_plot_id: Optional DOM id of the borough chart container.
            borough_bounds: Optional borough metadata used for click-to-zoom.
            value: Initial payload value exposed by the component.
            label: Optional Gradio label.
            **kwargs: Additional keyword arguments forwarded to ``gr.HTML``.
        """
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
        """Describe the payload schema emitted by this custom component."""
        return {
            "type": "object",
            "description": "Selection payload with source, event_kind, and grid_ids.",
        }


def load_data_from_csv(data_path):
    """Load geospatial training/inference data and derive model-ready features.

    Args:
        data_path: Path to the parquet dataset that includes geometry and bands.

    Returns:
        GeoDataFrame in EPSG:4326 with centroids, vegetation target, and spectral
        index features.
    """
    # Load parquet data at full row-level detail.
    df = pd.read_parquet(data_path)

    # Parse per-row GeoJSON geometry payloads into shapely objects.
    df["geometry"] = pd.Series(
        [shape(json.loads(x)) for x in df[".geo"]],
        index=df.index,
        dtype="object",
    )

    # Keep projected metric CRS for geometry operations that depend on distance.
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:32632")

    # Create stable ids and centroids before converting to geographic CRS.
    gdf["grid_id"] = gdf.index
    centroids = gdf.geometry.centroid
    gdf = gdf.to_crs("EPSG:4326")
    centroids = centroids.to_crs("EPSG:4326")
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y

    # Collapse vegetation-related labels into a single model target column.
    gdf["vegetation"] = gdf[["tree_cover", "cropland", "grassland"]].sum(axis=1)

    # Derive spectral indices used by the prediction model.
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
    """Convert class index to a human-readable label.

    Args:
        cls: Integer index into the class column list.

    Returns:
        Title-cased class label, or ``"unclassified"`` if the index is invalid.
    """
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
    """Load the persisted prediction model if the artifact exists.

    Args:
        model_path: Relative or absolute path to the pickled model file.

    Returns:
        Deserialized model object, or ``None`` when the file is absent.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        return None

    with model_file.open("rb") as f:
        return pickle.load(f)


prediction_model = load_prediction_model()

try:
    gdf = load_data_from_csv("data_3x3/delta_table_2021_3x3.parquet")
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
    """Recompute map data, selection tables, and borough chart for the dashboard.

    Args:
        start_year: User-selected base year.
        time_delta: Forecast horizon in years from the base year.
        map_type: Plotly basemap style identifier.
        render_mode: ``"points"`` or ``"polygons"`` rendering mode.
        grid_cell_size: Parent grid size in meters for aggregation.

    Returns:
        Tuple of selected-year figure, future figure, and borough change figure.
        If base data is unavailable, map outputs are ``None`` and the borough
        output is an informative placeholder.
    """
    if gdf is None:
        return (
            None,
            None,
            build_empty_borough_change_plot("No grid data available yet."),
        )

    start_year = int(start_year)
    time_delta = int(time_delta)
    grid_cell_size = int(grid_cell_size)
    np.random.seed(start_year)

    # Select rows matching the requested start year. The stored dataset uses
    # year offsets where delta 0 corresponds to 2021.
    delta_selection = 2021 - start_year

    base_df = gdf[gdf["delta_years"].astype(int) == delta_selection].copy()

    # Compute parent-grid coordinates once and reuse them across all outputs.
    base_df["grid_x"] = (
        np.floor(base_df["x"] / grid_cell_size) * grid_cell_size
    ).astype(int)
    base_df["grid_y"] = (
        np.floor(base_df["y"] / grid_cell_size) * grid_cell_size
    ).astype(int)

    # Build stable parent-grid metadata; polygon GeoJSON is generated lazily.
    grid_metadata = (
        base_df[["delta_years", "grid_x", "grid_y"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid_metadata["grid_id"] = np.arange(len(grid_metadata), dtype=int)
    plotly_geojson = None

    def get_plotly_geojson():
        """Lazily build and cache parent-grid GeoJSON for polygon rendering."""
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
        """Build a dataframe for one target delta with optional model inference."""
        df_out = base_df.copy()
        df_out["delta_years"] = prediction_delta

        if prediction_model is not None and feature_names is not None:
            missing_features = [
                name for name in feature_names if name not in df_out.columns
            ]
            if not missing_features:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*Falling back to prediction using DMatrix.*",
                    )
                    predicted_targets = prediction_model.predict(df_out[feature_names])
                predicted_targets_df = pd.DataFrame(
                    predicted_targets, index=df_out.index, columns=class_cols
                )
                predicted_targets_df = normalize_class_scores(
                    predicted_targets_df,
                    class_cols,
                )
                df_out[class_cols] = predicted_targets_df[class_cols]

        df_out = assign_row_dominant_class(df_out, class_cols)
        df_out = assign_group_dominant_class(df_out, class_cols)
        df_out["Parent Dominant Class"] = df_out["Dominant Class"]
        return df_out

    def build_subgrid_table(df_in: pd.DataFrame) -> pd.DataFrame:
        """Create a normalized sub-grid table used by selection analytics."""
        subgrid_df = df_in.merge(
            grid_metadata[["grid_x", "grid_y", "grid_id"]],
            on=["grid_x", "grid_y"],
            how="left",
            validate="many_to_one",
        ).rename(columns={"grid_id_x": "subgrid_id", "grid_id_y": "parent_grid_id"})

        # Backfill required columns so downstream code can rely on a fixed schema.
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
        """Build one map figure and its parent-grid table from sub-grid input."""
        # Aggregate sub-grid scores into parent-grid scores, then derive dominant
        # class and confidence buckets for display.
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

        # Use average lat/lon as representative positions for each parent grid.
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

        if render_mode == "points":
            # Render as point markers for faster interaction on larger datasets.
            fig = px.scatter_map(
                grid_df,
                lat="lat",
                lon="lon",
                color="Dominant Class",
                custom_data=["grid_id", "Dominant Class", "Confidence", "Borough"],
                color_discrete_map=color_map,
                hover_name=None,
                hover_data={"grid_id": False, "lat": False, "lon": False},
                zoom=11,
                center={"lat": lat_center, "lon": lon_center},
                map_style=map_type,
            )
        else:
            # Render as polygons using parent-grid geometry for visual precision.
            fig = px.choropleth_map(
                grid_df,
                geojson=get_plotly_geojson(),
                locations="grid_id",
                featureidkey="properties.grid_id",
                color="Dominant Class",
                custom_data=["grid_id", "Dominant Class", "Confidence", "Borough"],
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
                marker=dict(size=5, opacity=0.4),
                selected=dict(marker=dict(opacity=0.4)),
                unselected=dict(marker=dict(opacity=0.0)),
            )
        else:
            fig.update_traces(
                hovertemplate="Dominant Class: %{customdata[1]}<br>Confidence: %{customdata[2]}<extra></extra>",
                marker=dict(opacity=0.3),
                selected=dict(marker=dict(opacity=0.3)),
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

    global \
        LAST_GRID_TABLES, \
        LAST_SUBGRID_TABLES, \
        LAST_BOROUGH_CHANGE_FIG, \
        LAST_MAP_FIGURES
    LAST_GRID_TABLES = {
        "selected_year": selected_grid_df[SELECTION_COLUMNS].copy(),
        "future_prediction": future_grid_df[SELECTION_COLUMNS].copy(),
    }
    selected_subgrid = build_subgrid_table(predicted_dfs[0])
    future_subgrid = build_subgrid_table(predicted_dfs[1])
    LAST_SUBGRID_TABLES = {
        "selected_year": selected_subgrid,
        "future_prediction": future_subgrid,
    }
    borough_change_fig = build_top_changed_boroughs_chart(
        selected_subgrid,
        future_subgrid,
    )
    LAST_BOROUGH_CHANGE_FIG = borough_change_fig
    LAST_MAP_FIGURES = {
        "selected_year": go.Figure(selected_fig),
        "future_prediction": go.Figure(future_fig),
    }

    return selected_fig, future_fig, borough_change_fig


def selection_payload_to_outputs(payload: dict | None):
    """Convert a selection payload into all selection-dependent UI updates.

    Args:
        payload: Bridge payload containing event metadata and selected grid ids.

    Returns:
        Tuple of updates for summary text, two composition plots, delta plot,
        and borough chart placeholder/visibility state.
    """

    def hidden_plot_update():
        """Return a hidden plot update with no value."""
        return gr.update(value=None, visible=False)

    def visible_plot_update(fig: go.Figure):
        """Return a visible plot update for the provided figure."""
        return gr.update(value=fig, visible=True)

    def visible_borough_or_placeholder():
        """Show borough chart if available; otherwise keep placeholder visible."""
        fig = LAST_BOROUGH_CHANGE_FIG
        if fig is None:
            return gr.update(value=None, visible=True)
        return gr.update(value=fig, visible=True)

    def selected_empty_update(message: str):
        """Create hidden selected-year composition placeholder update."""
        return gr.update(
            value=build_empty_pie("Selected Year Composition", message),
            visible="hidden",
        )

    def future_empty_update(message: str):
        """Create hidden future composition placeholder update."""
        return gr.update(
            value=build_empty_pie("Future Prediction Composition", message),
            visible="hidden",
        )

    def delta_empty_update(message: str):
        """Create hidden delta placeholder update."""
        return gr.update(
            value=build_empty_delta("Land-Cover Composition Delta", message),
            visible="hidden",
        )

    def build_land_cover_composition(subgrid_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sub-grid class columns into composition totals and percents."""
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
        """Build a combined pie-plus-bar composition visualization."""
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
        """Build a horizontal delta bar chart from per-class differences."""
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

    # Ignore only empty bridge events emitted during immediate post-submit
    # re-rendering. Real selections must still be processed.
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


def reset_selection_insights():
    """Reset selection-dependent outputs to their default waiting state."""
    return selection_payload_to_outputs(None)


def clear_selection_without_recompute():
    """Clear current visual selections while keeping computed map data intact."""

    def clear_selectedpoints(fig: go.Figure | None):
        """Return an updated figure with Plotly selectedpoints cleared."""
        if fig is None:
            return gr.update()
        fig_out = go.Figure(fig)
        fig_out.update_traces(selectedpoints=None)
        return gr.update(value=fig_out)

    left_map_update = clear_selectedpoints(LAST_MAP_FIGURES.get("selected_year"))
    right_map_update = clear_selectedpoints(LAST_MAP_FIGURES.get("future_prediction"))

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


def build_map_titles(start_year, time_delta):
    """Build markdown titles for selected-year and future-year map panels."""
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
    """Run dashboard recomputation and append map panel titles for UI outputs."""
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
    start_year,
    time_delta,
    map_type,
    render_mode,
    grid_cell_size,
):
    """Handle submit action by recomputing all outputs and resetting selection UI."""
    global LAST_SUBMIT_TS_MS
    LAST_SUBMIT_TS_MS = int(time.time() * 1000)

    selected_fig, future_fig, borough_change_fig, left_title, right_title = (
        update_dashboard_with_titles(
            start_year=start_year,
            time_delta=time_delta,
            map_type=map_type,
            render_mode=render_mode,
            grid_cell_size=grid_cell_size,
        )
    )
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


# Build the interactive Gradio layout and event wiring.
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
            value=2021,
            label="Start Year Selection",
        )
        time_delta_drop_down = gr.Dropdown(
            choices=[i for i in range(0, 5)],
            value=0,
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
