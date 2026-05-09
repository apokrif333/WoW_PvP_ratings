from __future__ import annotations

import gc
from math import ceil
from pathlib import Path
from textwrap import dedent
from time import monotonic
from typing import Any

import pandas as pd
from dash import ALL, Dash, Input, Output, State, dash_table, dcc, html
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme

from wowpvp.icons import icon_slug
from wowpvp.storage import dataset_version, read_processed_players


DATA_PATH = Path("data/processed/pvp_players.parquet")
DATA_REFRESH_CHECK_SECONDS = 60
RATING_COLUMNS = ["shuffle_rating", "blitz_rating", "rating_2v2", "rating_3v3", "rating_rbg"]
GAME_MODE_COLUMNS = {
    "Shuffle": "shuffle_rating",
    "Blitz BG": "blitz_rating",
    "2v2": "rating_2v2",
    "3v3": "rating_3v3",
    "RBG": "rating_rbg",
}
PAGE_SIZE_OPTIONS = [{"label": str(value), "value": value} for value in (10, 20, 50, 100)]
MAX_DYNAMIC_OPTIONS = 500
HIGH_RATING_THRESHOLD = 1800
PERCENTILE_LIFT_QUANTILE = 0.80
INTEGER_FORMAT = Format(precision=0, scheme=Scheme.fixed)
RATING_FORMAT = Format(precision=1, scheme=Scheme.fixed)
LIFT_FORMAT = Format(precision=2, scheme=Scheme.fixed)
PERCENT_FORMAT = FormatTemplate.percentage(2)

MAIN_TABLE_COLUMNS = [
    {"name": "Region", "id": "region"},
    {"name": "Name", "id": "character_name"},
    {"name": "Realm", "id": "realm"},
    {"name": "Class", "id": "class_name"},
    {"name": "Spec", "id": "spec_name"},
    {"name": "Shuffle", "id": "shuffle_rating", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "Blitz BG", "id": "blitz_rating", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "2v2", "id": "rating_2v2", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "3v3", "id": "rating_3v3", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "RBG", "id": "rating_rbg", "type": "numeric", "format": INTEGER_FORMAT},
]
MAIN_TABLE_COLUMN_IDS = [column["id"] for column in MAIN_TABLE_COLUMNS]
MAIN_STRING_COLUMNS = ["region", "character_name", "realm", "class_name", "spec_name"]
APP_DATA_COLUMNS = [*MAIN_STRING_COLUMNS, *RATING_COLUMNS]
APP_CATEGORY_COLUMNS = ["region", "realm", "class_name", "spec_name"]

SUMMARY_FIXED_COLUMN_IDS = [
    "spec_name",
    "class_name",
    "game_mode",
    "region_filter",
    "total_players",
    "n_0_1400",
    "n_1400_1800",
    "n_1800_2100",
    "n_2100_plus",
]

SUMMARY_COLUMNS = [
    {"name": "Spec", "id": "spec_name", "presentation": "markdown"},
    {"name": "Class", "id": "class_name", "presentation": "markdown"},
    {"name": "Game Mode", "id": "game_mode"},
    {"name": "Region", "id": "region_filter"},
    {"name": "Total Players", "id": "total_players", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "n_0_1400", "id": "n_0_1400", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "n_1400_1800", "id": "n_1400_1800", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "n_1800_2100", "id": "n_1800_2100", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "n_2100_plus", "id": "n_2100_plus", "type": "numeric", "format": INTEGER_FORMAT},
    {"name": "pct_0_1400", "id": "pct_0_1400", "type": "numeric", "format": PERCENT_FORMAT},
    {"name": "pct_1400_1800", "id": "pct_1400_1800", "type": "numeric", "format": PERCENT_FORMAT},
    {"name": "pct_1800_2100", "id": "pct_1800_2100", "type": "numeric", "format": PERCENT_FORMAT},
    {"name": "pct_2100_plus", "id": "pct_2100_plus", "type": "numeric", "format": PERCENT_FORMAT},
    {"name": "mean_rating_all", "id": "mean_rating_all", "type": "numeric", "format": RATING_FORMAT},
    {"name": "median_rating_all", "id": "median_rating_all", "type": "numeric", "format": RATING_FORMAT},
    {"name": "q20_rating_all", "id": "q20_rating_all", "type": "numeric", "format": RATING_FORMAT},
    {"name": "q80_rating_all", "id": "q80_rating_all", "type": "numeric", "format": RATING_FORMAT},
    {"name": "mean_rating_1800_plus", "id": "mean_rating_1800_plus", "type": "numeric", "format": RATING_FORMAT},
    {"name": "median_rating_1800_plus", "id": "median_rating_1800_plus", "type": "numeric", "format": RATING_FORMAT},
    {"name": "q20_rating_1800_plus", "id": "q20_rating_1800_plus", "type": "numeric", "format": RATING_FORMAT},
    {"name": "q80_rating_1800_plus", "id": "q80_rating_1800_plus", "type": "numeric", "format": RATING_FORMAT},
    {"name": "overall_spec_share", "id": "overall_spec_share", "type": "numeric", "format": PERCENT_FORMAT},
    {"name": "spec_share_1800_plus", "id": "spec_share_1800_plus", "type": "numeric", "format": PERCENT_FORMAT},
    {"name": "lift_1800_plus", "id": "lift_1800_plus", "type": "numeric", "format": LIFT_FORMAT},
    {"name": "lift_p80_plus", "id": "lift_p80_plus", "type": "numeric", "format": LIFT_FORMAT},
]
SUMMARY_COLUMN_BY_ID = {column["id"]: column for column in SUMMARY_COLUMNS}
SUMMARY_COLUMN_IDS = [column["id"] for column in SUMMARY_COLUMNS]
SUMMARY_NUMERIC_COLUMNS = [
    column["id"] for column in SUMMARY_COLUMNS if column.get("type") == "numeric"
]
SUMMARY_OPTIONAL_COLUMN_IDS = [
    column_id for column_id in SUMMARY_COLUMN_IDS if column_id not in SUMMARY_FIXED_COLUMN_IDS
]
SUMMARY_DEFAULT_OPTIONAL_COLUMN_IDS = ["lift_1800_plus", "lift_p80_plus"]
SUMMARY_STRING_COLUMNS = ["spec_name", "class_name", "game_mode", "region_filter"]

RANGE_LABELS = {
    "shuffle_rating": "Shuffle",
    "blitz_rating": "Blitz BG",
    "rating_2v2": "2v2",
    "rating_3v3": "3v3",
    "rating_rbg": "RBG",
}
SUMMARY_RANGE_LABELS = {
    "total_players": "Total Players",
    "n_0_1400": "n 0-1400",
    "n_1400_1800": "n 1400-1800",
    "n_1800_2100": "n 1800-2100",
    "n_2100_plus": "n 2100+",
    "pct_0_1400": "pct 0-1400",
    "pct_1400_1800": "pct 1400-1800",
    "pct_1800_2100": "pct 1800-2100",
    "pct_2100_plus": "pct 2100+",
    "mean_rating_all": "Mean all",
    "median_rating_all": "Median all",
    "q20_rating_all": "Q20 all",
    "q80_rating_all": "Q80 all",
    "mean_rating_1800_plus": "Mean 1800+",
    "median_rating_1800_plus": "Median 1800+",
    "q20_rating_1800_plus": "Q20 1800+",
    "q80_rating_1800_plus": "Q80 1800+",
    "overall_spec_share": "Overall share",
    "spec_share_1800_plus": "1800+ share",
    "lift_1800_plus": "Lift 1800+",
    "lift_p80_plus": "Lift P80+",
}

MAIN_COLUMN_TOOLTIPS = {
    "region": "Регион персонажа: EU или US.",
    "character_name": "Имя персонажа.",
    "realm": "Реалм персонажа.",
    "class_name": "Класс персонажа.",
    "spec_name": "Спек персонажа.",
    "shuffle_rating": "Рейтинг Solo Shuffle (из Blizzard ladder). Если нет участия: 0.",
    "blitz_rating": "Рейтинг Blitz BG (из Blizzard ladder). Если нет участия: 0.",
    "rating_2v2": "Рейтинг 2v2 (из check-pvp). Если нет участия: 0.",
    "rating_3v3": "Рейтинг 3v3 (из check-pvp). Если нет участия: 0.",
    "rating_rbg": "Рейтинг RBG (из check-pvp). Если нет участия: 0.",
}

SUMMARY_COLUMN_TOOLTIPS = {
    "spec_name": "Название спека (агрегация по spec + class).",
    "class_name": "Название класса.",
    "game_mode": "Игровой режим, по которому считается summary.",
    "region_filter": "Регион/срез: Both, EU или US.",
    "total_players": "total_players = COUNT(игроков спека после всех фильтров).",
    "n_0_1400": "n_0_1400 = COUNT(rating >= 0 AND rating < 1400).",
    "n_1400_1800": "n_1400_1800 = COUNT(rating >= 1400 AND rating < 1800).",
    "n_1800_2100": "n_1800_2100 = COUNT(rating >= 1800 AND rating < 2100).",
    "n_2100_plus": "n_2100_plus = COUNT(rating >= 2100).",
    "pct_0_1400": "pct_0_1400 = n_0_1400 / total_players.",
    "pct_1400_1800": "pct_1400_1800 = n_1400_1800 / total_players.",
    "pct_1800_2100": "pct_1800_2100 = n_1800_2100 / total_players.",
    "pct_2100_plus": "pct_2100_plus = n_2100_plus / total_players.",
    "mean_rating_all": "mean_rating_all = AVG(rating по всем игрокам спека).",
    "median_rating_all": "median_rating_all = MEDIAN(rating по всем игрокам спека).",
    "q20_rating_all": "q20_rating_all = QUANTILE(rating, 0.20) по всем игрокам спека.",
    "q80_rating_all": "q80_rating_all = QUANTILE(rating, 0.80) по всем игрокам спека.",
    "mean_rating_1800_plus": "mean_rating_1800_plus = AVG(rating WHERE rating >= 1800).",
    "median_rating_1800_plus": "median_rating_1800_plus = MEDIAN(rating WHERE rating >= 1800).",
    "q20_rating_1800_plus": "q20_rating_1800_plus = QUANTILE(rating, 0.20 WHERE rating >= 1800).",
    "q80_rating_1800_plus": "q80_rating_1800_plus = QUANTILE(rating, 0.80 WHERE rating >= 1800).",
    "overall_spec_share": "overall_spec_share = total_players_спека / total_players_всех_спеков.",
    "spec_share_1800_plus": "spec_share_1800_plus = n_(rating>=1800)_спека / n_(rating>=1800)_всех_спеков.",
    "lift_1800_plus": "lift_1800_plus = spec_share_1800_plus / overall_spec_share.",
    "lift_p80_plus": "lift_p80_plus = spec_share_p80_plus / active_spec_share. P80 cutoff is calculated per game mode.",
}

FILTER_OPERATORS = [
    ("ge", ">="),
    ("le", "<="),
    ("ne", "!="),
    ("lt", "<"),
    ("gt", ">"),
    ("eq", "="),
    ("contains", "contains"),
]

TABLE_STYLE_CELL = {
    "fontFamily": "Segoe UI, Arial, sans-serif",
    "fontSize": "13px",
    "padding": "8px 10px",
    "border": "1px solid #d8dee8",
    "backgroundColor": "#ffffff",
    "color": "#111827",
    "minWidth": "84px",
    "maxWidth": "220px",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}
TABLE_STYLE_HEADER = {
    "backgroundColor": "#eef3f8",
    "color": "#0f172a",
    "fontWeight": "700",
    "border": "1px solid #cbd5e1",
}
TABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
    {"if": {"state": "active"}, "backgroundColor": "#fff7ed", "border": "1px solid #ea580c"},
    {"if": {"filter_query": "{rating_3v3} >= 2400"}, "backgroundColor": "#e0f2fe"},
]
MAIN_STYLE_CELL_CONDITIONAL = [
    {"if": {"column_id": column}, "textAlign": "right"} for column in RATING_COLUMNS
]
SUMMARY_STYLE_CELL_CONDITIONAL = [
    {"if": {"column_id": column}, "textAlign": "right"} for column in SUMMARY_NUMERIC_COLUMNS
]


def load_data() -> pd.DataFrame:
    df = read_processed_players(DATA_PATH, columns=APP_DATA_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=APP_DATA_COLUMNS)

    for column in APP_DATA_COLUMNS:
        if column not in df.columns:
            df[column] = 0 if column in RATING_COLUMNS else ""
    df = df[APP_DATA_COLUMNS]

    for column in RATING_COLUMNS:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=65535)
            .astype("uint16")
        )
    for column in APP_CATEGORY_COLUMNS:
        df[column] = df[column].fillna("").astype(str).astype("category")
    df["character_name"] = df["character_name"].fillna("").astype("string")
    return df


def make_options(values: pd.Series) -> list[dict[str, str]]:
    clean_values = sorted(v for v in values.dropna().astype(str).unique() if v)
    return [{"label": value.upper() if value in {"eu", "us"} else value, "value": value} for value in clean_values]


def make_column_options(column_ids: list[str]) -> list[dict[str, str]]:
    return [
        {
            "label": SUMMARY_RANGE_LABELS.get(column_id, SUMMARY_COLUMN_BY_ID[column_id]["name"]),
            "value": column_id,
        }
        for column_id in column_ids
    ]


def summary_visible_column_ids(optional_columns: list[str] | None) -> list[str]:
    selected = [column for column in (optional_columns or []) if column in SUMMARY_OPTIONAL_COLUMN_IDS]
    return [*SUMMARY_FIXED_COLUMN_IDS, *selected]


def summary_visible_columns(optional_columns: list[str] | None) -> list[dict[str, Any]]:
    return [SUMMARY_COLUMN_BY_ID[column_id] for column_id in summary_visible_column_ids(optional_columns)]


def make_limited_options(
    values: pd.Series,
    selected: list[str] | str | None = None,
    search: str | None = None,
    limit: int = MAX_DYNAMIC_OPTIONS,
) -> list[dict[str, str]]:
    clean_values = pd.Series(values.dropna().astype(str).unique())
    if search:
        folded_search = search.casefold()
        clean_values = clean_values[
            clean_values.str.casefold().str.contains(folded_search, regex=False)
        ]

    sorted_values = sorted(value for value in clean_values.tolist() if value)
    limited_values = sorted_values[:limit]
    selected_values = selected if isinstance(selected, list) else [selected] if selected else []
    for value in selected_values:
        if value and value not in limited_values:
            limited_values.append(value)
    limited_values = sorted(limited_values)
    return [{"label": value, "value": value} for value in limited_values]


def icon_markdown(kind: str, filename: str, label: str) -> str:
    return f"![{label}](/assets/icons/{kind}/{filename}.jpg) {label}"


def make_multi_filter(
    component_id: str,
    label: str,
    options: list[dict[str, str]],
    placeholder: str | None = None,
    value: list[str] | None = None,
) -> html.Div:
    return html.Div(
        className="field",
        children=[
            html.Label(label),
            dcc.Dropdown(
                id=component_id,
                className="dropdown-control",
                options=options,
                value=value,
                multi=True,
                placeholder=placeholder or label,
                maxHeight=520,
                optionHeight=34,
            ),
        ],
    )


def make_text_filter(component_id: str, label: str, placeholder: str | None = None) -> html.Div:
    return html.Div(
        className="field",
        children=[
            html.Label(label),
            dcc.Input(
                id=component_id,
                className="text-filter",
                type="text",
                debounce=False,
                placeholder=placeholder or label,
                value="",
            ),
        ],
    )


def summary_table_records(df: pd.DataFrame) -> list[dict]:
    return summary_table_records_for_columns(df, SUMMARY_COLUMN_IDS)


def summary_table_records_for_columns(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    display = df[columns].copy()
    if not display.empty:
        raw_classes = df["class_name"].astype(str)
        raw_specs = df["spec_name"].astype(str)
        if "class_name" in display.columns:
            display["class_name"] = raw_classes.map(
                lambda value: icon_markdown("class", icon_slug(str(value)), str(value))
            )
        if "spec_name" in display.columns:
            display["spec_name"] = [
                icon_markdown(
                    "spec",
                    f"{icon_slug(class_name)}-{icon_slug(spec_name)}",
                    spec_name,
                )
                for class_name, spec_name in zip(raw_classes, raw_specs)
            ]
    display = display.astype(object).where(pd.notna(display), None)
    return display.to_dict("records")


def make_page_size_control(component_id: str, value: int = 50) -> html.Div:
    return html.Div(
        className="field page-size-field",
        children=[
            html.Label("Rows"),
            dcc.Dropdown(
                id=component_id,
                className="dropdown-control",
                options=PAGE_SIZE_OPTIONS,
                value=value,
                clearable=False,
                searchable=False,
                maxHeight=220,
                optionHeight=34,
            ),
        ],
    )


def range_step(column: str) -> float | int:
    if column.startswith("pct_") or column.endswith("_share") or "_share_" in column:
        return 0.001
    if column.startswith("lift_"):
        return 0.01
    if column.startswith("mean_") or column.startswith("median_") or column.startswith("q"):
        return 1
    return 1


def format_range_value(column: str, value: float | int) -> str:
    if column.startswith("pct_") or column.endswith("_share") or "_share_" in column:
        return f"{value:.0%}"
    if column.startswith("lift_"):
        return f"{value:.2f}"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.1f}"
    return f"{int(value):,}"


def make_range_bounds(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int]]:
    bounds: dict[str, dict[str, float | int]] = {}
    for column in columns:
        if column not in df.columns or df.empty:
            minimum: float | int = 0
            maximum: float | int = 1
        else:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty:
                minimum = 0
                maximum = 1
            else:
                minimum = float(series.min())
                maximum = float(series.max())
                if range_step(column) == 1:
                    minimum = int(minimum)
                    maximum = int(maximum)
        if minimum == maximum:
            maximum = minimum + range_step(column)
        bounds[column] = {"min": minimum, "max": maximum, "step": range_step(column)}
    return bounds


def make_summary_range_bounds(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    max_rating = 3000
    if not df.empty:
        for column in RATING_COLUMNS:
            if column not in df.columns:
                continue
            column_max = pd.to_numeric(df[column], errors="coerce").max()
            if not pd.isna(column_max):
                max_rating = max(max_rating, int(column_max))

    max_players = max(1, int(len(df)))
    bounds: dict[str, dict[str, float | int]] = {}
    for column in SUMMARY_NUMERIC_COLUMNS:
        if column.startswith("pct_") or column.endswith("_share") or "_share_" in column:
            minimum: float | int = 0
            maximum: float | int = 1
        elif column.startswith("lift_"):
            minimum = 0
            maximum = 5
        elif column.startswith(("mean_", "median_", "q")):
            minimum = 0
            maximum = max_rating
        else:
            minimum = 0
            maximum = max_players

        bounds[column] = {"min": minimum, "max": maximum, "step": range_step(column)}
    return bounds


def default_range_values(
    columns: list[str],
    bounds: dict[str, dict[str, float | int]],
) -> list[list[float | int]]:
    return [[bounds[column]["min"], bounds[column]["max"]] for column in columns]


def make_range_slider(
    prefix: str,
    column: str,
    label: str,
    bounds: dict[str, dict[str, float | int]],
) -> html.Div:
    minimum = bounds[column]["min"]
    maximum = bounds[column]["max"]
    return html.Div(
        className="range-field slider-field",
        children=[
            html.Div(
                className="slider-label-row",
                children=[
                    html.Label(label),
                    html.Span(
                        f"{format_range_value(column, minimum)} - {format_range_value(column, maximum)}",
                        className="slider-bounds",
                    ),
                ],
            ),
            html.Div(
                className="range-slider-wrap",
                children=[
                    dcc.RangeSlider(
                        id={"type": f"{prefix}-range", "column": column},
                        min=minimum,
                        max=maximum,
                        step=bounds[column]["step"],
                        value=[minimum, maximum],
                        marks={
                            minimum: format_range_value(column, minimum),
                            maximum: format_range_value(column, maximum),
                        },
                        tooltip={"placement": "bottom", "always_visible": False},
                        allowCross=False,
                    ),
                ],
            ),
        ],
    )


def parse_filter_expression(filter_part: str) -> tuple[str | None, str | None, str | float | None]:
    for operator_name, operator_token in FILTER_OPERATORS:
        spaced_operator = f" {operator_token} "
        if spaced_operator not in filter_part:
            continue

        name_part, value_part = filter_part.split(spaced_operator, 1)
        if "{" not in name_part or "}" not in name_part:
            return None, None, None

        column_id = name_part[name_part.find("{") + 1 : name_part.rfind("}")]
        value_part = value_part.strip()
        if not value_part:
            return column_id, operator_name, ""

        if value_part[0] in ("'", '"', "`") and value_part[-1] == value_part[0]:
            value: str | float = value_part[1:-1]
        else:
            try:
                value = float(value_part)
            except ValueError:
                value = value_part
        return column_id, operator_name, value

    return None, None, None


def apply_table_filter(
    df: pd.DataFrame,
    filter_query: str | None,
    numeric_columns: set[str],
) -> pd.DataFrame:
    if not filter_query:
        return df

    for filter_part in filter_query.split(" && "):
        column_id, operator, value = parse_filter_expression(filter_part)
        if not column_id or not operator or column_id not in df.columns:
            continue

        if column_id in numeric_columns:
            series = pd.to_numeric(df[column_id], errors="coerce")
            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric_value):
                continue
            if operator == "ge":
                df = df[series >= numeric_value]
            elif operator == "le":
                df = df[series <= numeric_value]
            elif operator == "gt":
                df = df[series > numeric_value]
            elif operator == "lt":
                df = df[series < numeric_value]
            elif operator == "ne":
                df = df[series != numeric_value]
            elif operator == "eq":
                df = df[series == numeric_value]
            elif operator == "contains":
                df = df[series.astype(str).str.contains(str(value), case=False, na=False)]
            continue

        text_series = df[column_id].astype(str).str.casefold()
        text_value = str(value).casefold()
        if operator == "contains":
            df = df[text_series.str.contains(text_value, na=False)]
        elif operator == "eq":
            df = df[text_series == text_value]
        elif operator == "ne":
            df = df[text_series != text_value]
        elif operator == "ge":
            df = df[text_series >= text_value]
        elif operator == "le":
            df = df[text_series <= text_value]
        elif operator == "gt":
            df = df[text_series > text_value]
        elif operator == "lt":
            df = df[text_series < text_value]

    return df


def apply_numeric_ranges(
    df: pd.DataFrame,
    columns: list[str],
    range_values: list[list[Any]] | None,
    bounds: dict[str, dict[str, float | int]],
) -> pd.DataFrame:
    range_values = range_values or []

    for index, column in enumerate(columns):
        if column not in df.columns:
            continue

        value = range_values[index] if index < len(range_values) else None
        if not value or len(value) != 2:
            continue

        min_value, max_value = value
        default_min = bounds[column]["min"]
        default_max = bounds[column]["max"]
        if min_value <= default_min and max_value >= default_max:
            continue

        numeric_min = pd.to_numeric(min_value, errors="coerce")
        if not pd.isna(numeric_min):
            series = pd.to_numeric(df[column], errors="coerce")
            df = df[series >= numeric_min]

        numeric_max = pd.to_numeric(max_value, errors="coerce")
        if not pd.isna(numeric_max):
            series = pd.to_numeric(df[column], errors="coerce")
            df = df[series <= numeric_max]

    return df


def apply_string_filters(
    df: pd.DataFrame,
    filters: dict[str, list[str] | None],
) -> pd.DataFrame:
    for column, values in filters.items():
        if column not in df.columns or not values:
            continue
        df = df[df[column].astype(str).isin(values)]
    return df


def apply_text_contains_filter(df: pd.DataFrame, column: str, value: str | None) -> pd.DataFrame:
    query = (value or "").strip()
    if not query or column not in df.columns:
        return df
    return df[df[column].astype(str).str.casefold().str.contains(query.casefold(), regex=False)]


def apply_table_sort(
    df: pd.DataFrame,
    sort_by: list[dict] | None,
    numeric_columns: set[str],
) -> pd.DataFrame:
    if not sort_by:
        return df

    for sort_rule in reversed(sort_by):
        column_id = sort_rule.get("column_id")
        if column_id not in df.columns:
            continue

        ascending = sort_rule.get("direction") == "asc"
        if column_id in numeric_columns:
            sortable = df.assign(__sort_key=pd.to_numeric(df[column_id], errors="coerce"))
            df = sortable.sort_values(
                "__sort_key",
                ascending=ascending,
                kind="mergesort",
                na_position="last",
            ).drop(columns="__sort_key")
        else:
            df = df.sort_values(
                column_id,
                ascending=ascending,
                kind="mergesort",
                na_position="last",
                key=lambda series: series.astype(str).str.casefold(),
            )

    return df


def page_dataframe(
    df: pd.DataFrame,
    page_current: int | None,
    page_size: int | None,
) -> tuple[pd.DataFrame, int]:
    total_rows = len(df)
    page_size = page_size or 50
    page_count = max(1, ceil(total_rows / page_size)) if total_rows else 0
    safe_page = min(page_current or 0, max(page_count - 1, 0))
    page_start = safe_page * page_size
    return df.iloc[page_start : page_start + page_size], page_count


def table_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    clean = df[columns].copy()
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict("records")


def make_quantile(series: pd.Series, q: float) -> float | None:
    if series.empty:
        return None
    return round(float(series.quantile(q)), 2)


def make_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.mean()), 2)


def active_mode_ratings(df: pd.DataFrame, rating_column: str) -> pd.Series:
    ratings = pd.to_numeric(df[rating_column], errors="coerce").fillna(0)
    return ratings[ratings > 0]


def mode_percentile_cutoff(df: pd.DataFrame, rating_column: str) -> float | None:
    return make_quantile(active_mode_ratings(df, rating_column), PERCENTILE_LIFT_QUANTILE)


def make_p80_lift_tooltip() -> str:
    if DATA.empty:
        return (
            "lift_p80_plus = spec_share_p80_plus / active_spec_share. "
            "P80 cutoff is calculated per game mode from active characters with rating > 0."
        )

    cutoffs = []
    for mode, rating_column in GAME_MODE_COLUMNS.items():
        cutoff = mode_percentile_cutoff(DATA, rating_column)
        if cutoff is not None:
            cutoffs.append(f"{mode} - lift cutoff {int(round(cutoff))}")

    cutoff_text = ", ".join(cutoffs) if cutoffs else "no active ratings yet"
    return (
        "lift_p80_plus = spec_share_p80_plus / active_spec_share. "
        "The cutoff is the 80th percentile rating among active characters "
        f"(rating > 0) in the same game mode and region. Current Both cutoffs: {cutoff_text}."
    )


def make_summary_for_mode_region(mode: str, region_filter: str) -> pd.DataFrame:
    rating_column = GAME_MODE_COLUMNS.get(mode, "shuffle_rating")
    region_label = region_filter or "Both"

    df = DATA[["region", "class_name", "spec_name", rating_column]].copy()
    if region_label != "Both":
        df = df[df["region"] == region_label.lower()]

    df[rating_column] = pd.to_numeric(df[rating_column], errors="coerce").fillna(0)
    total_players = len(df)
    total_1800_plus = int((df[rating_column] >= HIGH_RATING_THRESHOLD).sum())
    active_df = df[df[rating_column] > 0]
    total_active_players = len(active_df)
    p80_cutoff = mode_percentile_cutoff(df, rating_column)
    total_p80_plus = int((df[rating_column] >= p80_cutoff).sum()) if p80_cutoff is not None else 0
    rows: list[dict[str, Any]] = []

    for (class_name, spec_name), group in df.groupby(["class_name", "spec_name"], dropna=False):
        ratings = group[rating_column]
        total = int(len(group))
        active_total = int((ratings > 0).sum())
        n_0_1400 = int(((ratings >= 0) & (ratings < 1400)).sum())
        n_1400_1800 = int(((ratings >= 1400) & (ratings < 1800)).sum())
        n_1800_2100 = int(((ratings >= 1800) & (ratings < 2100)).sum())
        high_ratings = ratings[ratings >= HIGH_RATING_THRESHOLD]
        n_1800_plus = int(len(high_ratings))
        n_true_2100_plus = int((ratings >= 2100).sum())
        n_p80_plus = int((ratings >= p80_cutoff).sum()) if p80_cutoff is not None else 0

        overall_spec_share = total / total_players if total_players else None
        active_spec_share = active_total / total_active_players if total_active_players else None
        spec_share_1800_plus = (
            n_1800_plus / total_1800_plus if n_1800_plus and total_1800_plus else None
        )
        spec_share_p80_plus = n_p80_plus / total_p80_plus if n_p80_plus and total_p80_plus else None
        lift_1800_plus = (
            spec_share_1800_plus / overall_spec_share
            if spec_share_1800_plus is not None and overall_spec_share
            else None
        )
        lift_p80_plus = (
            spec_share_p80_plus / active_spec_share
            if spec_share_p80_plus is not None and active_spec_share
            else None
        )

        rows.append(
            {
                "spec_name": spec_name,
                "class_name": class_name,
                "game_mode": mode,
                "region_filter": region_label,
                "total_players": total,
                "n_0_1400": n_0_1400,
                "n_1400_1800": n_1400_1800,
                "n_1800_2100": n_1800_2100,
                "n_2100_plus": n_true_2100_plus,
                "pct_0_1400": round(n_0_1400 / total, 4) if total else None,
                "pct_1400_1800": round(n_1400_1800 / total, 4) if total else None,
                "pct_1800_2100": round(n_1800_2100 / total, 4) if total else None,
                "pct_2100_plus": round(n_true_2100_plus / total, 4) if total else None,
                "mean_rating_all": make_mean(ratings),
                "median_rating_all": make_quantile(ratings, 0.5),
                "q20_rating_all": make_quantile(ratings, 0.2),
                "q80_rating_all": make_quantile(ratings, 0.8),
                "mean_rating_1800_plus": make_mean(high_ratings),
                "median_rating_1800_plus": make_quantile(high_ratings, 0.5),
                "q20_rating_1800_plus": make_quantile(high_ratings, 0.2),
                "q80_rating_1800_plus": make_quantile(high_ratings, 0.8),
                "overall_spec_share": round(overall_spec_share, 6) if overall_spec_share else None,
                "spec_share_1800_plus": (
                    round(spec_share_1800_plus, 6) if spec_share_1800_plus is not None else None
                ),
                "lift_1800_plus": round(lift_1800_plus, 4) if lift_1800_plus is not None else None,
                "lift_p80_plus": round(lift_p80_plus, 4) if lift_p80_plus is not None else None,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMN_IDS)

    for column in SUMMARY_NUMERIC_COLUMNS:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    return summary.sort_values(["class_name", "spec_name"], kind="mergesort")


def make_summary(modes: list[str] | None, region_filters: list[str] | None) -> pd.DataFrame:
    modes = modes or list(GAME_MODE_COLUMNS.keys())
    region_filters = region_filters or ["Both", "US", "EU"]
    frames = [
        make_summary_for_mode_region(mode, region_filter)
        for mode in modes
        for region_filter in region_filters
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMN_IDS)
    return pd.concat(frames, ignore_index=True)


DATA = pd.DataFrame(columns=APP_DATA_COLUMNS)
MAIN_RANGE_BOUNDS: dict[str, dict[str, float | int]] = {}
SUMMARY_RANGE_BOUNDS: dict[str, dict[str, float | int]] = {}
DATA_VERSION: str | None = None
DATA_LAST_CHECK = 0.0


def reload_application_data() -> None:
    global DATA, MAIN_RANGE_BOUNDS, SUMMARY_RANGE_BOUNDS, DATA_VERSION

    DATA = pd.DataFrame(columns=APP_DATA_COLUMNS)
    gc.collect()
    DATA = load_data()
    MAIN_RANGE_BOUNDS = make_range_bounds(DATA, RATING_COLUMNS)
    SUMMARY_RANGE_BOUNDS = make_summary_range_bounds(DATA)
    DATA_VERSION = dataset_version(DATA_PATH)
    SUMMARY_COLUMN_TOOLTIPS["lift_p80_plus"] = make_p80_lift_tooltip()
    gc.collect()


def refresh_application_data_if_changed(force: bool = False) -> None:
    global DATA_LAST_CHECK

    now = monotonic()
    if not force and now - DATA_LAST_CHECK < DATA_REFRESH_CHECK_SECONDS:
        return

    DATA_LAST_CHECK = now
    current_version = dataset_version(DATA_PATH)
    if force or current_version != DATA_VERSION:
        reload_application_data()


reload_application_data()

app = Dash(__name__, title="WoW PvP Data", suppress_callback_exceptions=True)
server = app.server


PLAYER_TABLE_GUIDE_RU = dedent(
    """
    ### Таблица персонажей: что она показывает

    Эта таблица отвечает на вопрос: **какие конкретные персонажи есть в датасете и какие рейтинги у них в разных PvP-режимах**. Одна строка - один персонаж, объединенный по региону, реалму и имени. Данные собираются из Blizzard ladder для Solo Shuffle и Blitz BG, а также из check-pvp для 2v2, 3v3 и RBG.

    **Как пользоваться таблицей**

    - `Region` фильтрует EU/US. Пример: выбери `EU`, чтобы анализировать только европейский ладдер.
    - `Name` ищет подстроку по всем именам персонажей, а не только по выпадающему списку. Пример: введи `feet`, и таблица найдет `Feetup`, если персонаж есть в базе.
    - `Realm`, `Class`, `Spec` сужают выборку до конкретного реалма, класса или специализации.
    - `Rating ranges` задают числовые диапазоны по режимам. Пример: `3v3 >= 2400` через ползунок `3v3` оставит только персонажей с высоким 3v3-рейтингом.
    - Сортировка по заголовку колонки позволяет быстро находить топов по конкретному режиму.
    - `Rows` меняет размер страницы, чтобы смотреть больше или меньше строк за раз.

    **Колонки**

    - `Region` - регион персонажа: `eu` или `us`.
    - `Name` - имя персонажа.
    - `Realm` - игровой реалм персонажа.
    - `Class` - класс персонажа, например Druid, Mage, Warrior.
    - `Spec` - активная специализация, например Restoration, Fire, Arms.
    - `Shuffle` - текущий Solo Shuffle rating из Blizzard PvP leaderboard. Если персонаж не найден в этом режиме, значение `0`.
    - `Blitz BG` - текущий Battleground Blitz rating из Blizzard PvP leaderboard. Если участия нет, значение `0`.
    - `2v2`, `3v3`, `RBG` - рейтинги из check-pvp. Если режима нет в данных, значение `0`.

    **Как читать значения**

    `0` не всегда означает плохой рейтинг. В этой базе это чаще значит **нет найденного участия в конкретном режиме**. Например, персонаж может иметь `2600` в Shuffle и `0` в 3v3, потому что он не играет 3v3 или check-pvp не вернул для него рейтинг.

    **Примеры полезных вопросов**

    - Найти всех персонажей конкретного спека: выбери `Class = Druid`, `Spec = Restoration`.
    - Найти сильных игроков в одном режиме: выставь `Shuffle` от `2400` и отсортируй по `Shuffle`.
    - Сравнить режимы у одного персонажа: введи имя в `Name` и посмотри, где у него высокий рейтинг, а где `0`.
    - Найти реалмовые кластеры: выбери realm и посмотри, какие классы и спеки чаще встречаются в топе.

    **Аналитическая польза**

    Таблица персонажей полезна для точечного поиска: кто играет какой спек, где у персонажа основной рейтинг, какие спеки представлены на высоких рейтингах и есть ли пересечение между режимами. Это нижний уровень данных, из которого строится summary ниже.
    """
).strip()


PLAYER_TABLE_GUIDE_EN = dedent(
    """
    ### Player table: what it shows

    This table answers: **which individual characters are in the dataset and what ratings they have across PvP modes**. One row is one character, merged by region, realm, and character name. Solo Shuffle and Blitz BG come from Blizzard ladders; 2v2, 3v3, and RBG come from check-pvp.

    **How to use it**

    - `Region` filters EU/US. Example: choose `EU` to inspect only the European ladder.
    - `Name` searches across all character names by substring. Example: type `feet` to find `Feetup` if that character exists in the dataset.
    - `Realm`, `Class`, and `Spec` narrow the table to a realm, class, or specialization.
    - `Rating ranges` filter numeric ratings by mode. Example: set `3v3` to `2400+` to keep only high-rated 3v3 characters.
    - Click a column header to sort by that column.
    - `Rows` changes how many rows are shown per page.

    **Columns**

    - `Region` - character region: `eu` or `us`.
    - `Name` - character name.
    - `Realm` - character realm.
    - `Class` - class name, such as Druid, Mage, Warrior.
    - `Spec` - active specialization, such as Restoration, Fire, Arms.
    - `Shuffle` - current Solo Shuffle rating from Blizzard PvP leaderboards. Missing participation is shown as `0`.
    - `Blitz BG` - current Battleground Blitz rating from Blizzard PvP leaderboards. Missing participation is shown as `0`.
    - `2v2`, `3v3`, `RBG` - ratings from check-pvp. Missing mode data is shown as `0`.

    **How to interpret values**

    `0` does not always mean a weak character. In this dataset it usually means **no detected participation in that mode**. A character can have `2600` in Shuffle and `0` in 3v3 because they do not play 3v3 or because check-pvp did not return a rating for that mode.

    **Useful questions this table can answer**

    - Find all characters of a spec: choose `Class = Druid`, `Spec = Restoration`.
    - Find strong players in one mode: set `Shuffle` to `2400+` and sort by `Shuffle`.
    - Compare one character across modes: type a name and inspect where the character has ratings.
    - Inspect realm-level patterns: filter a realm and look at which specs/classes appear at high rating.

    **Analytical value**

    The player table is for exact lookup and micro-analysis: who plays what, which mode is a character's main mode, which specs appear at high ratings, and whether strong players overlap across modes. It is the raw character layer used to build the summary table below.
    """
).strip()


SUMMARY_TABLE_GUIDE_RU = dedent(
    """
    ### Spec Summary: что показывает вторая таблица

    Summary агрегирует персонажей по связке **Class + Spec + Game Mode + Region**. Она отвечает не на вопрос "кто конкретно играет", а на вопрос **как распределены спеки в режиме и насколько часто конкретный спек встречается среди сильных игроков**.

    **Как пользоваться**

    - `Game Mode` выбирает режим анализа: Shuffle, Blitz BG, 2v2, 3v3 или RBG.
    - `Region` выбирает срез: `Both`, `US`, `EU`.
    - `Class` и `Spec` позволяют смотреть конкретный класс/спек или сравнивать несколько.
    - `Extra Columns` добавляет статистические колонки: процентили, средние, доли, lift.
    - `Summary numeric ranges` фильтрует summary по числовым метрикам. Пример: можно оставить только спеки с `lift_p80_plus > 1.2`.

    **Базовые колонки**

    - `Spec`, `Class` - специализация и класс, по которым сделана агрегация.
    - `Game Mode` - режим, рейтинг которого используется в расчетах.
    - `Region` - срез региона: Both, EU или US.
    - `Total Players` - количество персонажей этого спека в выбранном режиме/регионе.
    - `n_0_1400`, `n_1400_1800`, `n_1800_2100`, `n_2100_plus` - сколько персонажей спека попало в каждый рейтинговый диапазон.

    **Процентные колонки**

    - `pct_0_1400 = n_0_1400 / total_players`.
    - `pct_1400_1800 = n_1400_1800 / total_players`.
    - `pct_1800_2100 = n_1800_2100 / total_players`.
    - `pct_2100_plus = n_2100_plus / total_players`.

    Эти колонки показывают внутреннюю структуру спека. Например, если у спека высокий `pct_2100_plus`, значит заметная часть его игроков находится в верхнем диапазоне рейтинга.

    **Средние и процентили**

    - `mean_rating_all` - средний рейтинг всех персонажей спека.
    - `median_rating_all` - медиана рейтинга всех персонажей спека.
    - `q20_rating_all` и `q80_rating_all` - 20-й и 80-й процентили рейтинга всех персонажей спека.
    - `mean_rating_1800_plus`, `median_rating_1800_plus`, `q20_rating_1800_plus`, `q80_rating_1800_plus` - те же метрики, но только среди персонажей с рейтингом `>= 1800`.

    Процентили полезнее среднего, когда распределение неровное. Например, один очень высокий игрок может поднять среднее, но медиана и q80 лучше показывают типичный верхний уровень спека.

    **Доли и lift**

    - `overall_spec_share = total_players_спека / total_players_всех_спеков`.
    - `spec_share_1800_plus = players_спека_с_rating>=1800 / players_всех_спеков_с_rating>=1800`.
    - `lift_1800_plus = spec_share_1800_plus / overall_spec_share`.

    Если `lift_1800_plus = 1.00`, спек представлен среди 1800+ примерно так же, как в общей популяции. Если `lift_1800_plus = 1.50`, спек встречается среди 1800+ на 50% чаще, чем ожидалось по его общей доле. Если `0.70`, спек недопредставлен.

    **Mode-specific P80 lift**

    `lift_p80_plus` похож на `lift_1800_plus`, но порог считается отдельно для каждого режима как **80-й процентиль рейтинга всех активных персонажей режима**. Это важно, потому что распределения в Shuffle, Blitz, 2v2, 3v3 и RBG разные. Один фиксированный порог `1800` может быть слишком мягким для одного режима и слишком жестким для другого.

    Расчет:

    - Берем всех активных персонажей режима с `rating > 0`.
    - Считаем 80-й процентиль рейтинга этого режима.
    - Считаем долю спека среди персонажей выше этого порога.
    - Делим ее на долю спека среди всех активных персонажей режима.

    **Какой вывод делать из lift**

    - `lift > 1` - спек чаще встречается в верхнем рейтинговом сегменте, чем в общей базе. Это может указывать на силу спека, высокий skill ceiling, популярность среди сильных игроков или метовую востребованность.
    - `lift около 1` - спек представлен примерно нейтрально.
    - `lift < 1` - спек реже доходит до верхнего сегмента, чем ожидалось по численности. Это может указывать на сложность спека, слабую мету, низкую популярность среди сильных игроков или специфику режима.

    Важно: lift не доказывает баланс сам по себе. Он показывает **представленность**, а не прямую причинность. Для надежного вывода смотри одновременно `Total Players`, `pct_2100_plus`, `q80_rating_all`, `lift_1800_plus` и `lift_p80_plus`.
    """
).strip()


SUMMARY_TABLE_GUIDE_EN = dedent(
    """
    ### Spec Summary: what the second table shows

    The summary table aggregates characters by **Class + Spec + Game Mode + Region**. It is not about individual players; it is about **how specs are distributed in a mode and how often a spec appears among stronger players**.

    **How to use it**

    - `Game Mode` selects the rating mode: Shuffle, Blitz BG, 2v2, 3v3, or RBG.
    - `Region` selects the slice: `Both`, `US`, or `EU`.
    - `Class` and `Spec` focus the table on specific classes/specs.
    - `Extra Columns` adds statistical columns: percentiles, averages, shares, and lift.
    - `Summary numeric ranges` filters the summary by metrics. Example: keep only specs with `lift_p80_plus > 1.2`.

    **Core columns**

    - `Spec`, `Class` - specialization and class used for aggregation.
    - `Game Mode` - the mode whose rating is being analyzed.
    - `Region` - Both, EU, or US.
    - `Total Players` - number of characters of that spec in the selected mode/region.
    - `n_0_1400`, `n_1400_1800`, `n_1800_2100`, `n_2100_plus` - counts of spec players in each rating band.

    **Percentage columns**

    - `pct_0_1400 = n_0_1400 / total_players`.
    - `pct_1400_1800 = n_1400_1800 / total_players`.
    - `pct_1800_2100 = n_1800_2100 / total_players`.
    - `pct_2100_plus = n_2100_plus / total_players`.

    These columns describe the internal rating structure of a spec. For example, a high `pct_2100_plus` means a meaningful share of that spec's players sit in the upper rating band.

    **Averages and percentiles**

    - `mean_rating_all` - average rating of all characters of the spec.
    - `median_rating_all` - median rating of all characters of the spec.
    - `q20_rating_all` and `q80_rating_all` - 20th and 80th rating percentiles for the spec.
    - `mean_rating_1800_plus`, `median_rating_1800_plus`, `q20_rating_1800_plus`, `q80_rating_1800_plus` - the same metrics, but only for characters with rating `>= 1800`.

    Percentiles are often more stable than the mean when distributions are uneven. One extreme player can pull the average up, while median and q80 better describe the typical upper range of a spec.

    **Shares and lift**

    - `overall_spec_share = spec_total_players / all_specs_total_players`.
    - `spec_share_1800_plus = spec_players_rating>=1800 / all_specs_players_rating>=1800`.
    - `lift_1800_plus = spec_share_1800_plus / overall_spec_share`.

    If `lift_1800_plus = 1.00`, the spec is represented among 1800+ players roughly as often as expected from its overall population share. If `lift_1800_plus = 1.50`, the spec appears among 1800+ players 50% more often than expected. If it is `0.70`, the spec is underrepresented.

    **Mode-specific P80 lift**

    `lift_p80_plus` is similar to `lift_1800_plus`, but the cutoff is calculated separately for each mode as the **80th percentile rating of all active characters in that mode**. This matters because Shuffle, Blitz, 2v2, 3v3, and RBG have different rating distributions. A fixed `1800` cutoff can be too easy in one mode and too strict in another.

    Calculation:

    - Take all active characters in the mode with `rating > 0`.
    - Calculate the mode's 80th percentile rating.
    - Calculate the spec's share among characters at or above that cutoff.
    - Divide it by the spec's share among all active characters in that mode.

    **How to interpret lift**

    - `lift > 1` - the spec appears in the upper rating segment more often than its population share predicts. This can suggest spec strength, high skill ceiling, popularity among strong players, or meta relevance.
    - `lift around 1` - neutral representation.
    - `lift < 1` - the spec reaches the upper segment less often than expected from its population size. This can point to difficulty, weaker meta position, lower adoption by strong players, or mode-specific limitations.

    Lift does not prove balance by itself. It measures **representation**, not direct causality. For stronger conclusions, look at `Total Players`, `pct_2100_plus`, `q80_rating_all`, `lift_1800_plus`, and `lift_p80_plus` together.
    """
).strip()


def make_info_tabs(component_id: str, russian_text: str, english_text: str) -> html.Div:
    return html.Div(
        className="info-panel",
        children=[
            dcc.Tabs(
                id=f"{component_id}-tabs",
                value="ru",
                className="info-tabs",
                parent_className="info-tabs-wrap",
                children=[
                    dcc.Tab(
                        label="Русский",
                        value="ru",
                        className="info-tab",
                        selected_className="info-tab info-tab-selected",
                        children=dcc.Markdown(russian_text, className="info-copy"),
                    ),
                    dcc.Tab(
                        label="English",
                        value="en",
                        className="info-tab",
                        selected_className="info-tab info-tab-selected",
                        children=dcc.Markdown(english_text, className="info-copy"),
                    ),
                ],
            ),
        ],
    )


def layout() -> html.Div:
    refresh_application_data_if_changed()

    if DATA.empty:
        return html.Div(
            className="page",
            children=[
                html.H1("WoW PvP Data"),
                html.Div(
                    className="empty-state",
                    children="No local dataset found. Run `python main.py` first.",
                ),
            ],
        )

    return html.Div(
        className="page",
        children=[
            html.Div(
                className="toolbar",
                children=[
                    html.Div(
                        children=[
                            html.H1("WoW PvP Data"),
                            html.Div(id="row-count", className="row-count"),
                        ],
                    ),
                    html.Button("Reset", id="reset-filters", className="reset-button", n_clicks=0),
                ],
            ),
            make_info_tabs("player-table-guide", PLAYER_TABLE_GUIDE_RU, PLAYER_TABLE_GUIDE_EN),
            html.Div(
                className="filters main-filters",
                children=[
                    make_multi_filter(
                        "region-filter",
                        "Region",
                        make_options(DATA["region"]),
                        "All regions",
                    ),
                    make_text_filter(
                        "character-filter",
                        "Name",
                        "Search name",
                    ),
                    make_multi_filter(
                        "realm-filter",
                        "Realm",
                        make_options(DATA["realm"]),
                        "All realms",
                    ),
                    make_multi_filter(
                        "class-filter",
                        "Class",
                        make_options(DATA["class_name"]),
                        "All classes",
                    ),
                    make_multi_filter(
                        "spec-filter",
                        "Spec",
                        make_options(DATA["spec_name"]),
                        "All specs",
                    ),
                    make_page_size_control("main-page-size", 50),
                ],
            ),
            html.Details(
                className="range-panel",
                open=True,
                children=[
                    html.Summary("Rating ranges"),
                    html.Div(
                        className="range-grid",
                        children=[
                            make_range_slider(
                                "main-rating",
                                column,
                                RANGE_LABELS[column],
                                MAIN_RANGE_BOUNDS,
                            )
                            for column in RATING_COLUMNS
                        ],
                    ),
                ],
            ),
            dash_table.DataTable(
                id="pvp-table",
                columns=MAIN_TABLE_COLUMNS,
                tooltip_header=MAIN_COLUMN_TOOLTIPS,
                tooltip_delay=250,
                tooltip_duration=None,
                data=[],
                page_action="custom",
                page_current=0,
                page_size=50,
                page_count=0,
                sort_action="custom",
                sort_by=[],
                sort_mode="single",
                fixed_rows={"headers": True},
                style_as_list_view=True,
                style_table={"height": "520px", "overflowX": "auto", "overflowY": "auto"},
                style_cell=TABLE_STYLE_CELL,
                style_cell_conditional=MAIN_STYLE_CELL_CONDITIONAL,
                style_header=TABLE_STYLE_HEADER,
                style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
            ),
            html.Div(
                className="section-heading",
                children=[
                    html.Div(
                        children=[
                            html.H2("Spec Summary"),
                            html.Div(id="summary-row-count", className="row-count"),
                        ],
                    ),
                ],
            ),
            make_info_tabs("summary-table-guide", SUMMARY_TABLE_GUIDE_RU, SUMMARY_TABLE_GUIDE_EN),
            html.Div(
                className="filters summary-filters",
                children=[
                    make_multi_filter(
                        "summary-mode-filter",
                        "Game Mode",
                        [{"label": label, "value": label} for label in GAME_MODE_COLUMNS.keys()],
                        "All modes",
                        ["Shuffle"],
                    ),
                    make_multi_filter(
                        "summary-region-filter",
                        "Region",
                        [
                            {"label": "Both", "value": "Both"},
                            {"label": "US", "value": "US"},
                            {"label": "EU", "value": "EU"},
                        ],
                        "All region views",
                        ["Both"],
                    ),
                    make_multi_filter(
                        "summary-class-filter",
                        "Class",
                        make_options(DATA["class_name"]),
                        "All classes",
                    ),
                    make_multi_filter(
                        "summary-spec-filter",
                        "Spec",
                        make_options(DATA["spec_name"]),
                        "All specs",
                    ),
                    make_multi_filter(
                        "summary-column-filter",
                        "Extra Columns",
                        make_column_options(SUMMARY_OPTIONAL_COLUMN_IDS),
                        "Add stats",
                        SUMMARY_DEFAULT_OPTIONAL_COLUMN_IDS,
                    ),
                    make_page_size_control("summary-page-size", 50),
                ],
            ),
            html.Details(
                className="range-panel",
                children=[
                    html.Summary("Summary numeric ranges"),
                    html.Div(
                        className="range-grid summary-range-grid",
                        children=[
                            make_range_slider(
                                "summary-range",
                                column,
                                SUMMARY_RANGE_LABELS.get(column, column),
                                SUMMARY_RANGE_BOUNDS,
                            )
                            for column in SUMMARY_NUMERIC_COLUMNS
                        ],
                    ),
                ],
            ),
            dash_table.DataTable(
                id="summary-table",
                columns=summary_visible_columns(SUMMARY_DEFAULT_OPTIONAL_COLUMN_IDS),
                tooltip_header=SUMMARY_COLUMN_TOOLTIPS,
                tooltip_delay=250,
                tooltip_duration=None,
                data=[],
                page_action="custom",
                page_current=0,
                page_size=50,
                page_count=0,
                sort_action="custom",
                sort_by=[],
                sort_mode="single",
                fixed_rows={"headers": True},
                style_as_list_view=True,
                style_table={"height": "520px", "overflowX": "auto", "overflowY": "auto"},
                style_cell={**TABLE_STYLE_CELL, "minWidth": "116px", "maxWidth": "240px"},
                style_cell_conditional=SUMMARY_STYLE_CELL_CONDITIONAL,
                style_header=TABLE_STYLE_HEADER,
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
                    {"if": {"state": "active"}, "backgroundColor": "#fff7ed", "border": "1px solid #ea580c"},
                ],
                markdown_options={"link_target": "_self"},
            ),
        ],
    )


app.layout = layout


@app.callback(Output("pvp-table", "page_size"), Input("main-page-size", "value"))
def sync_main_page_size(page_size: int | None) -> int:
    return page_size or 50


@app.callback(Output("summary-table", "page_size"), Input("summary-page-size", "value"))
def sync_summary_page_size(page_size: int | None) -> int:
    return page_size or 50


@app.callback(
    Output("pvp-table", "page_current"),
    Input("region-filter", "value"),
    Input("character-filter", "value"),
    Input("realm-filter", "value"),
    Input("class-filter", "value"),
    Input("spec-filter", "value"),
    Input({"type": "main-rating-range", "column": ALL}, "value"),
    Input("main-page-size", "value"),
    Input("pvp-table", "sort_by"),
)
def reset_main_page_on_query_change(
    _regions: list[str] | None,
    _character_query: str | None,
    _realms: list[str] | None,
    _classes: list[str] | None,
    _specs: list[str] | None,
    _rating_ranges: list[list[Any]] | None,
    _page_size: int | None,
    _sort_by: list[dict] | None,
) -> int:
    return 0


@app.callback(
    Output("summary-table", "page_current"),
    Input("summary-mode-filter", "value"),
    Input("summary-region-filter", "value"),
    Input("summary-class-filter", "value"),
    Input("summary-spec-filter", "value"),
    Input("summary-column-filter", "value"),
    Input({"type": "summary-range-range", "column": ALL}, "value"),
    Input("summary-page-size", "value"),
    Input("summary-table", "sort_by"),
)
def reset_summary_page_on_query_change(
    _modes: list[str] | None,
    _regions: list[str] | None,
    _classes: list[str] | None,
    _specs: list[str] | None,
    _columns: list[str] | None,
    _ranges: list[list[Any]] | None,
    _page_size: int | None,
    _sort_by: list[dict] | None,
) -> int:
    return 0


@app.callback(
    Output("pvp-table", "data"),
    Output("pvp-table", "page_count"),
    Output("row-count", "children"),
    Input("region-filter", "value"),
    Input("character-filter", "value"),
    Input("realm-filter", "value"),
    Input("class-filter", "value"),
    Input("spec-filter", "value"),
    Input({"type": "main-rating-range", "column": ALL}, "value"),
    Input("pvp-table", "page_current"),
    Input("pvp-table", "page_size"),
    Input("pvp-table", "sort_by"),
)
def filter_main_rows(
    regions: list[str] | None,
    character_query: str | None,
    realms: list[str] | None,
    classes: list[str] | None,
    specs: list[str] | None,
    rating_ranges: list[list[Any]] | None,
    page_current: int | None,
    page_size: int | None,
    sort_by: list[dict] | None,
) -> tuple[list[dict], int, str]:
    refresh_application_data_if_changed()
    df = DATA
    df = apply_string_filters(
        df,
        {
            "region": regions,
            "realm": realms,
            "class_name": classes,
            "spec_name": specs,
        },
    )
    df = apply_text_contains_filter(df, "character_name", character_query)
    df = apply_numeric_ranges(df, RATING_COLUMNS, rating_ranges, MAIN_RANGE_BOUNDS)
    df = apply_table_sort(df, sort_by, set(RATING_COLUMNS))
    page_df, page_count = page_dataframe(df, page_current, page_size)

    return table_records(page_df, MAIN_TABLE_COLUMN_IDS), page_count, f"{len(df):,} rows"


@app.callback(
    Output("summary-table", "data"),
    Output("summary-table", "page_count"),
    Output("summary-row-count", "children"),
    Output("summary-table", "columns"),
    Input("summary-mode-filter", "value"),
    Input("summary-region-filter", "value"),
    Input("summary-class-filter", "value"),
    Input("summary-spec-filter", "value"),
    Input("summary-column-filter", "value"),
    Input({"type": "summary-range-range", "column": ALL}, "value"),
    Input("summary-table", "page_current"),
    Input("summary-table", "page_size"),
    Input("summary-table", "sort_by"),
)
def update_summary_rows(
    modes: list[str] | None,
    regions: list[str] | None,
    classes: list[str] | None,
    specs: list[str] | None,
    optional_columns: list[str] | None,
    range_values: list[list[Any]] | None,
    page_current: int | None,
    page_size: int | None,
    sort_by: list[dict] | None,
) -> tuple[list[dict], int, str, list[dict[str, Any]]]:
    refresh_application_data_if_changed()
    visible_ids = summary_visible_column_ids(optional_columns)
    visible_columns = summary_visible_columns(optional_columns)
    df = make_summary(modes, regions)
    df = apply_string_filters(df, {"class_name": classes, "spec_name": specs})
    df = apply_numeric_ranges(df, SUMMARY_NUMERIC_COLUMNS, range_values, SUMMARY_RANGE_BOUNDS)
    visible_sort_by = [
        sort_rule for sort_rule in (sort_by or []) if sort_rule.get("column_id") in visible_ids
    ]
    df = apply_table_sort(df, visible_sort_by, set(SUMMARY_NUMERIC_COLUMNS))
    page_df, page_count = page_dataframe(df, page_current, page_size)

    return (
        summary_table_records_for_columns(page_df, visible_ids),
        page_count,
        f"{len(df):,} specs",
        visible_columns,
    )


@app.callback(
    Output("region-filter", "value"),
    Output("character-filter", "value"),
    Output("realm-filter", "value"),
    Output("class-filter", "value"),
    Output("spec-filter", "value"),
    Output({"type": "main-rating-range", "column": ALL}, "value"),
    Output("pvp-table", "sort_by"),
    Output("summary-mode-filter", "value"),
    Output("summary-region-filter", "value"),
    Output("summary-class-filter", "value"),
    Output("summary-spec-filter", "value"),
    Output("summary-column-filter", "value"),
    Output({"type": "summary-range-range", "column": ALL}, "value"),
    Output("summary-table", "sort_by"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(
    _clicks: int,
) -> tuple[
    None,
    str,
    None,
    None,
    None,
    list[list[float | int]],
    list,
    list[str],
    list[str],
    None,
    None,
    list,
    list[list[float | int]],
    list,
]:
    refresh_application_data_if_changed()
    return (
        None,
        "",
        None,
        None,
        None,
        default_range_values(RATING_COLUMNS, MAIN_RANGE_BOUNDS),
        [],
        ["Shuffle"],
        ["Both"],
        None,
        None,
        SUMMARY_DEFAULT_OPTIONAL_COLUMN_IDS,
        default_range_values(SUMMARY_NUMERIC_COLUMNS, SUMMARY_RANGE_BOUNDS),
        [],
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
