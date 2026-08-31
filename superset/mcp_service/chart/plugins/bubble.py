# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Bubble chart type plugin."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from typing import Any, ClassVar, Literal

from superset.mcp_service.chart.chart_utils import (
    _bubble_chart_what,
    _summarize_filters,
    map_bubble_config,
)
from superset.mcp_service.chart.plugin import BaseChartPlugin
from superset.mcp_service.chart.schemas import BubbleChartConfig, ChartError, ColumnRef
from superset.mcp_service.chart.validation.dataset_validator import (
    DatasetValidator,
    is_numeric_column,
)
from superset.mcp_service.common.error_schemas import ChartGenerationError

BubbleMetricOutputStatus = Literal["numeric", "nonnumeric", "unknown"]

_COUNT_AGGREGATES = {"COUNT", "COUNT_DISTINCT"}
_NUMERIC_AGGREGATES = {
    "SUM",
    "AVG",
    "MEDIAN",
    "PERCENTILE",
    "STDDEV",
    "STDDEV_SAMP",
    "VAR",
    "VAR_SAMP",
}
_SET_AGGREGATES = _COUNT_AGGREGATES | _NUMERIC_AGGREGATES | {"MIN", "MAX"}

# Saved metrics are stored as MediumText and do not pass through ColumnRef's
# 2,000-character bound. Keep static inference deliberately smaller than a SQL
# parser: every input is scanned once and any exceeded budget fails closed to a
# validating query.
_MAX_SQL_EXPRESSION_LENGTH = 16_384
_MAX_SQL_EXPRESSION_TOKENS = 2_048
_MAX_SQL_EXPRESSION_DEPTH = 64
_MAX_SQL_CLASSIFICATION_WORK = 4_096

_SqlTokenKind = Literal[
    "identifier",
    "quoted_identifier",
    "number",
    "string",
    "left_parenthesis",
    "right_parenthesis",
    "comma",
    "comment",
    "other",
]


@dataclass(frozen=True, slots=True)
class _SqlToken:
    """One structural SQL token used by conservative Bubble type inference."""

    kind: _SqlTokenKind
    text: str


@dataclass(frozen=True, slots=True)
class _TokenizedSql:
    """Bounded token stream plus precomputed parenthesis relationships."""

    tokens: tuple[_SqlToken, ...]
    matching_parentheses: dict[int, int]
    comment_prefix: tuple[int, ...]
    has_nested_set_aggregate: bool


@dataclass(slots=True)
class _ClassificationBudget:
    """Deterministic work allowance for token-span classification."""

    remaining: int = _MAX_SQL_CLASSIFICATION_WORK

    def spend(self, amount: int = 1) -> bool:
        self.remaining -= amount
        return self.remaining >= 0


def _column_output_status(name: str, dataset_context: Any) -> BubbleMetricOutputStatus:
    """Classify a physical column from authoritative dataset metadata."""
    for column in dataset_context.available_columns:
        if str(column.get("name", "")).lower() != name.lower():
            continue
        type_name = str(column.get("type") or "").strip().upper()
        if is_numeric_column(column):
            return "numeric"
        if type_name not in {"", "UNKNOWN"}:
            return "nonnumeric"
        return "unknown"
    return "unknown"


def _unquote_identifier(value: str) -> str:
    if len(value) >= 2 and (value[0], value[-1]) in {
        ('"', '"'),
        ("`", "`"),
        ("[", "]"),
    }:
        return value[1:-1]
    return value


def _is_identifier_start(char: str) -> bool:
    return char == "_" or char.isalpha()


def _is_identifier_part(char: str) -> bool:
    return char in {"_", "$"} or char.isalnum()


def _tokenize_sql(expression: str) -> _TokenizedSql | None:  # noqa: C901
    """Tokenize SQL once, returning ``None`` when a safety budget is exceeded."""
    if len(expression) > _MAX_SQL_EXPRESSION_LENGTH:
        return None

    tokens: list[_SqlToken] = []
    parentheses: list[int] = []
    matching_parentheses: dict[int, int] = {}
    index = 0
    length = len(expression)

    while index < length:
        char = expression[index]
        if char.isspace():
            index += 1
            continue

        start = index
        kind: _SqlTokenKind
        if expression.startswith("--", index):
            index += 2
            while index < length and expression[index] not in {"\r", "\n"}:
                index += 1
            kind = "comment"
        elif expression.startswith("/*", index):
            comment_end = expression.find("*/", index + 2)
            if comment_end < 0:
                return None
            index = comment_end + 2
            kind = "comment"
        elif char in {"'", '"', "`", "["}:
            closing_quote = "]" if char == "[" else char
            index += 1
            while index < length:
                current = expression[index]
                if current == "\\" and closing_quote != "]":
                    index += 2
                    continue
                if current == closing_quote:
                    if index + 1 < length and expression[index + 1] == closing_quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                return None
            kind = "string" if char == "'" else "quoted_identifier"
        elif _is_identifier_start(char):
            index += 1
            while index < length and _is_identifier_part(expression[index]):
                index += 1
            kind = "identifier"
        elif (
            char.isdigit()
            or (char == "." and index + 1 < length and expression[index + 1].isdigit())
            or (
                char in {"+", "-"}
                and index + 1 < length
                and (
                    expression[index + 1].isdigit()
                    or (
                        expression[index + 1] == "."
                        and index + 2 < length
                        and expression[index + 2].isdigit()
                    )
                )
            )
        ):
            if char in {"+", "-"}:
                index += 1
            while index < length and expression[index].isdigit():
                index += 1
            if index < length and expression[index] == ".":
                index += 1
                while index < length and expression[index].isdigit():
                    index += 1
            kind = "number"
        else:
            index += 1
            if char == "(":
                kind = "left_parenthesis"
            elif char == ")":
                kind = "right_parenthesis"
            elif char == ",":
                kind = "comma"
            else:
                kind = "other"

        token_index = len(tokens)
        tokens.append(_SqlToken(kind, expression[start:index]))
        if len(tokens) > _MAX_SQL_EXPRESSION_TOKENS:
            return None
        if kind == "left_parenthesis":
            parentheses.append(token_index)
            if len(parentheses) > _MAX_SQL_EXPRESSION_DEPTH:
                return None
        elif kind == "right_parenthesis":
            if not parentheses:
                return None
            matching_parentheses[parentheses.pop()] = token_index

    if parentheses or not tokens:
        return None

    comment_prefix = [0]
    for token in tokens:
        comment_prefix.append(comment_prefix[-1] + (token.kind == "comment"))

    active_aggregate_closings: list[int] = []
    has_nested_set_aggregate = False
    for token_index, token in enumerate(tokens):
        while active_aggregate_closings and token_index > active_aggregate_closings[-1]:
            active_aggregate_closings.pop()
        if (
            token.kind == "identifier"
            and token.text.upper() in _SET_AGGREGATES
            and token_index + 1 < len(tokens)
            and tokens[token_index + 1].kind == "left_parenthesis"
        ):
            closing = matching_parentheses.get(token_index + 1)
            if closing is None:
                return None
            if active_aggregate_closings:
                has_nested_set_aggregate = True
                break
            active_aggregate_closings.append(closing)

    return _TokenizedSql(
        tokens=tuple(tokens),
        matching_parentheses=matching_parentheses,
        comment_prefix=tuple(comment_prefix),
        has_nested_set_aggregate=has_nested_set_aggregate,
    )


def _parse_cast_type(  # noqa: C901
    parsed: _TokenizedSql,
    start: int,
    end: int,
    budget: _ClassificationBudget,
) -> str | None:
    """Return a simple, comment-free CAST target type from an argument span."""
    if start > end or parsed.comment_prefix[end + 1] != parsed.comment_prefix[start]:
        return None

    tokens = parsed.tokens
    as_position: int | None = None
    index = start
    while index <= end:
        if not budget.spend():
            return None
        token = tokens[index]
        if token.kind == "left_parenthesis":
            closing = parsed.matching_parentheses.get(index)
            if closing is None or closing > end:
                return None
            index = closing + 1
            continue
        if token.kind == "identifier" and token.text.upper() == "AS":
            if as_position is not None:
                return None
            as_position = index
        index += 1

    if as_position is None:
        return None
    type_tokens = tokens[as_position + 1 : end + 1]
    if not type_tokens or type_tokens[0].kind != "identifier":
        return None

    base_type = type_tokens[0].text.upper()
    type_index = 1
    if (
        base_type == "DOUBLE"
        and len(type_tokens) > 1
        and type_tokens[1].kind == "identifier"
        and type_tokens[1].text.upper() == "PRECISION"
    ):
        base_type = "DOUBLE PRECISION"
        type_index = 2
    if type_index == len(type_tokens):
        return base_type

    parameters = type_tokens[type_index:]
    if (
        len(parameters) not in {3, 5}
        or parameters[0].kind != "left_parenthesis"
        or parameters[-1].kind != "right_parenthesis"
        or not parameters[1].text.isdigit()
        or parameters[1].kind != "number"
    ):
        return None
    if len(parameters) == 5 and (
        parameters[2].kind != "comma"
        or parameters[3].kind != "number"
        or not parameters[3].text.isdigit()
    ):
        return None
    values = [parameters[1].text]
    if len(parameters) == 5:
        values.append(parameters[3].text)
    return f"{base_type}({', '.join(values)})"


def _sql_expression_output_status(  # noqa: C901
    expression: str | None, dataset_context: Any
) -> BubbleMetricOutputStatus:
    """Conservatively infer whether a SQL metric produces numeric values.

    This intentionally recognizes only proofs that are portable across SQL
    engines. Expressions that cannot be proven statically are validated from a
    small query result by the compile/preview paths.
    """
    if not expression:
        return "unknown"
    parsed = _tokenize_sql(expression)
    if parsed is None or parsed.has_nested_set_aggregate:
        return "unknown"

    tokens = parsed.tokens
    start = 0
    end = len(tokens) - 1
    budget = _ClassificationBudget()
    while start <= end and budget.spend():
        while (
            start < end
            and tokens[start].kind == "left_parenthesis"
            and parsed.matching_parentheses.get(start) == end
        ):
            if not budget.spend():
                return "unknown"
            start += 1
            end -= 1

        if start == end:
            token = tokens[start]
            if token.kind in {"identifier", "quoted_identifier"}:
                return _column_output_status(
                    _unquote_identifier(token.text), dataset_context
                )
            if token.kind == "number":
                return "numeric"
            if token.kind == "string":
                return "nonnumeric"
            return "unknown"

        if (
            tokens[start].kind != "identifier"
            or start + 1 > end
            or tokens[start + 1].kind != "left_parenthesis"
            or parsed.matching_parentheses.get(start + 1) != end
        ):
            return "unknown"
        function = tokens[start].text.upper()
        argument_start = start + 2
        argument_end = end - 1

        if function in {"CAST", "TRY_CAST"}:
            cast_type = _parse_cast_type(parsed, argument_start, argument_end, budget)
            if cast_type is None:
                return "unknown"
            if is_numeric_column({"type": cast_type}):
                return "numeric"
            if cast_type.split("(", 1)[0] in {
                "CHAR",
                "VARCHAR",
                "STRING",
                "TEXT",
                "BOOLEAN",
                "BOOL",
                "DATE",
                "TIME",
                "TIMESTAMP",
            }:
                return "nonnumeric"
            return "unknown"
        if function in _COUNT_AGGREGATES:
            return "numeric"
        if function in _NUMERIC_AGGREGATES or function in {"MIN", "MAX"}:
            # Continue through one wrapper chain without recursive calls or
            # substring rescans. Unknown compound arguments fail closed.
            start = argument_start
            end = argument_end
            continue
        return "unknown"
    return "unknown"


def bubble_metric_output_status(
    metric: ColumnRef, dataset_context: Any
) -> BubbleMetricOutputStatus:
    """Classify one typed Bubble metric's result as numeric/non-numeric/unknown."""
    if metric.aggregate in _COUNT_AGGREGATES:
        return "numeric"
    if metric.aggregate:
        if metric.name is None:
            return "unknown"
        return _column_output_status(metric.name, dataset_context)
    if metric.sql_expression:
        return _sql_expression_output_status(metric.sql_expression, dataset_context)
    if metric.saved_metric and metric.name:
        for saved_metric in dataset_context.available_metrics:
            if str(saved_metric.get("name", "")).lower() != metric.name.lower():
                continue
            status = _sql_expression_output_status(
                saved_metric.get("expression"), dataset_context
            )
            if status != "unknown":
                return status
            return "unknown"
    return "unknown"


def bubble_metrics_requiring_query_validation(
    config: Any, dataset_context: Any
) -> list[str]:
    """Return Bubble quantitative channels lacking static numeric proof."""
    if not isinstance(config, BubbleChartConfig):
        return []
    return [
        field
        for field in ("x", "y", "size")
        if bubble_metric_output_status(getattr(config, field), dataset_context)
        == "unknown"
    ]


def _is_finite_numeric(value: Any) -> bool:
    """Return whether a value is a finite real number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Number):
        return False
    try:
        return not isinstance(value, complex) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def bubble_metric_field(metric: Any) -> str | None:
    """Return the query-result field name for a native Bubble metric."""
    if isinstance(metric, str):
        return metric
    if not isinstance(metric, dict):
        return None
    label = metric.get("label")
    return label if isinstance(label, str) and label else None


def validate_bubble_query_output(
    data: list[Any],
    form_data: Mapping[str, Any],
    *,
    require_runtime_numeric_proof: bool,
) -> ChartError | None:
    """Validate Bubble query shape and quantitative output values.

    An empty result is a valid chart result when dataset metadata or portable
    expression inference already proves the three metric outputs are numeric.
    Callers that still need runtime proof must fail closed on an empty result.
    """
    entity = form_data.get("entity")
    metric_fields = {
        channel: bubble_metric_field(form_data.get(channel))
        for channel in ("x", "y", "size")
    }
    missing_config = [
        field
        for field, value in {"entity": entity, **metric_fields}.items()
        if not isinstance(value, str) or not value
    ]
    if missing_config:
        return ChartError(
            error=(
                "Bubble query requires entity, x, y, and size form fields; "
                f"missing or invalid: {', '.join(missing_config)}"
            ),
            error_type="InvalidChart",
        )
    if not data:
        if require_runtime_numeric_proof:
            return ChartError(
                error=(
                    "Bubble query returned no rows, so the numeric output of "
                    "an unproven saved or custom metric could not be verified"
                ),
                error_type="InvalidChartData",
            )
        return None
    if not all(isinstance(row, Mapping) for row in data):
        return ChartError(
            error="Bubble query returned rows in an unsupported shape",
            error_type="InvalidChartData",
        )

    result_fields = {key for row in data for key in row}
    required_result_fields = {"entity": entity, **metric_fields}
    series = form_data.get("series")
    if isinstance(series, str) and series:
        required_result_fields["series"] = series
    missing_results = [
        f"{channel} ({field})"
        for channel, field in required_result_fields.items()
        if field not in result_fields
    ]
    if missing_results:
        return ChartError(
            error=(
                "Bubble query result is missing required column(s): "
                + ", ".join(missing_results)
            ),
            error_type="InvalidChartData",
        )

    for channel, field in metric_fields.items():
        assert isinstance(field, str)
        samples = [row.get(field) for row in data if row.get(field) is not None]
        if not samples:
            return ChartError(
                error=(
                    f"Bubble {channel} result column '{field}' has no non-null "
                    "numeric values"
                ),
                error_type="InvalidChartData",
            )
        if any(not _is_finite_numeric(value) for value in samples):
            return ChartError(
                error=(
                    f"Bubble {channel} result column '{field}' must contain "
                    "finite numeric, non-boolean values"
                ),
                error_type="InvalidChartData",
            )
    return None


def _invalid_bubble_metric_output(
    field: str, metric: ColumnRef
) -> ChartGenerationError:
    label = metric.label or metric.name or metric.sql_expression or field
    return ChartGenerationError(
        error_type="invalid_bubble_metric_output",
        message=f"Bubble {field} metric '{label}' does not produce numeric values",
        details=(
            f"Bubble's {field} channel is quantitative. COUNT and COUNT_DISTINCT "
            "may aggregate any column, but SUM/AVG/MIN/MAX and other numeric "
            "Bubble metrics must produce numbers."
        ),
        suggestions=[
            "Use COUNT or COUNT_DISTINCT to count text values",
            "Choose a numeric input column for the metric",
            "Use a saved or SQL metric whose output is numeric",
        ],
        error_code="INVALID_BUBBLE_METRIC_OUTPUT",
    )


class BubbleChartPlugin(BaseChartPlugin):
    """Plugin for bubble chart type."""

    chart_type = "bubble"
    display_name = "Bubble Chart"
    native_viz_types: ClassVar[Mapping[str, str]] = {
        "bubble_v2": "Bubble Chart",
    }

    def pre_validate(
        self,
        config: dict[str, Any],
    ) -> ChartGenerationError | None:
        missing_fields = []

        if "entity" not in config:
            missing_fields.append("'entity' (category column per bubble)")
        if "x" not in config:
            missing_fields.append("'x' (metric for horizontal position)")
        if "y" not in config:
            missing_fields.append("'y' (metric for vertical position)")
        if "size" not in config:
            missing_fields.append("'size' (metric for bubble area)")

        if missing_fields:
            return ChartGenerationError(
                error_type="missing_bubble_fields",
                message=(
                    f"Bubble chart missing required fields: {', '.join(missing_fields)}"
                ),
                details=(
                    "Bubble charts plot an entity by three metrics: x and y "
                    "position each bubble and size sets its area"
                ),
                suggestions=[
                    "Add 'entity': {'name': 'country'}",
                    "Add 'x': {'name': 'gdp', 'aggregate': 'AVG'}",
                    "Add 'y': {'name': 'life_expectancy', 'aggregate': 'AVG'}",
                    "Add 'size': {'name': 'population', 'aggregate': 'SUM'}",
                ],
                error_code="MISSING_BUBBLE_FIELDS",
            )

        return None

    def extract_column_refs(self, config: Any) -> list[ColumnRef]:
        if not isinstance(config, BubbleChartConfig):
            return []
        refs: list[ColumnRef] = [config.entity, config.x, config.y, config.size]
        if config.series:
            refs.append(config.series)
        if config.order_by:
            refs.append(config.order_by)
        if config.filters:
            for f in config.filters:
                refs.append(ColumnRef(name=f.column))
        return refs

    def validate_dataset(
        self, config: Any, dataset_context: Any
    ) -> ChartGenerationError | None:
        """Require every quantitative Bubble channel to have numeric output."""
        if not isinstance(config, BubbleChartConfig):
            return None
        for field in ("x", "y", "size"):
            metric = getattr(config, field)
            if bubble_metric_output_status(metric, dataset_context) == "nonnumeric":
                return _invalid_bubble_metric_output(field, metric)
        return None

    def to_form_data(
        self, config: Any, dataset_id: int | str | None = None
    ) -> dict[str, Any]:
        return map_bubble_config(config)

    def generate_name(self, config: Any, dataset_name: str | None = None) -> str:
        what = _bubble_chart_what(config)
        context = _summarize_filters(config.filters)
        return self._with_context(what, context)

    def resolve_viz_type(self, config: Any) -> str:
        return "bubble_v2"

    def normalize_column_refs(self, config: Any, dataset_context: Any) -> Any:
        config_dict = config.model_dump()

        for key in ("entity", "series"):
            col = config_dict.get(key)
            if col and not col.get("sql_expression") and not col.get("saved_metric"):
                col["name"] = DatasetValidator.get_canonical_column_name(
                    col["name"], dataset_context
                )
        for key in ("x", "y", "size", "order_by"):
            metric = config_dict.get(key)
            if not metric:
                continue
            if metric.get("sql_expression"):
                continue
            if metric.get("saved_metric"):
                metric["name"] = DatasetValidator.get_canonical_metric_name(
                    metric["name"], dataset_context
                )
            else:
                metric["name"] = DatasetValidator.get_canonical_column_name(
                    metric["name"], dataset_context
                )
        DatasetValidator.normalize_filters(config_dict, dataset_context)
        return BubbleChartConfig.model_validate(config_dict)

    def schema_error_hint(self) -> ChartGenerationError | None:
        return ChartGenerationError(
            error_type="bubble_validation_error",
            message="Bubble chart configuration validation failed",
            details=(
                "The bubble chart configuration is missing required "
                "fields or has invalid structure"
            ),
            suggestions=[
                "Ensure 'entity' has a 'name'",
                "Ensure 'x', 'y', and 'size' each have 'name' and 'aggregate'",
                "Example: {'chart_type': 'bubble', "
                "'entity': {'name': 'country'}, "
                "'x': {'name': 'gdp', 'aggregate': 'AVG'}, "
                "'y': {'name': 'life_expectancy', 'aggregate': 'AVG'}, "
                "'size': {'name': 'population', 'aggregate': 'SUM'}}",
            ],
            error_code="BUBBLE_VALIDATION_ERROR",
        )
