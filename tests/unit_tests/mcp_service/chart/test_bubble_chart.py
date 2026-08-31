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

"""Tests for the bubble chart type plugin.

Schema validation, form_data mapping (matching the frontend Bubble buildQuery
contract for viz_type ``bubble_v2`` — an ``entity`` dimension plus three
separate metric keys ``x``/``y``/``size`` and an optional ``series``), and
registry integration.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from superset.charts.schemas import ChartDataQueryObjectSchema
from superset.mcp_service.chart.chart_helpers import apply_bubble_ordering
from superset.mcp_service.chart.chart_utils import (
    analyze_chart_capabilities,
    analyze_chart_semantics,
    map_bubble_config,
    map_config_to_form_data,
)
from superset.mcp_service.chart.compile import _compile_chart
from superset.mcp_service.chart.preview_utils import (
    _build_query_fields,
    _generate_vega_lite_preview_from_data,
)
from superset.mcp_service.chart.schemas import (
    BubbleChartConfig,
    ChartConfig,
    ChartError,
    GenerateChartRequest,
    GetChartPreviewRequest,
    UpdateChartRequest,
)
from superset.mcp_service.chart.tool.get_chart_preview import VegaLitePreviewStrategy
from superset.utils import json


def _base(**overrides):
    cfg = {
        "chart_type": "bubble",
        "entity": {"name": "country"},
        "x": {"name": "gdp", "aggregate": "AVG"},
        "y": {"name": "life_expectancy", "aggregate": "AVG"},
        "size": {"name": "population", "aggregate": "SUM"},
    }
    cfg.update(overrides)
    return cfg


class TestBubbleChartConfigSchema:
    """BubbleChartConfig schema validation."""

    def test_basic_bubble_config(self) -> None:
        config = BubbleChartConfig(**_base())
        assert config.entity.name == "country"
        assert config.x.name == "gdp"
        assert config.series is None  # series grouping is optional
        assert config.row_limit == 10000  # shared control default

    @pytest.mark.parametrize("missing", ["entity", "x", "y", "size"])
    def test_bubble_missing_required(self, missing: str) -> None:
        cfg = _base()
        del cfg[missing]
        with pytest.raises(ValidationError):
            BubbleChartConfig(**cfg)

    def test_bubble_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            BubbleChartConfig(**_base(bogus=1))

    def test_bubble_entity_rejects_saved_metric(self) -> None:
        with pytest.raises(ValidationError):
            BubbleChartConfig(**_base(entity={"name": "c", "saved_metric": True}))

    def test_bubble_series_rejects_saved_metric(self) -> None:
        with pytest.raises(ValidationError):
            BubbleChartConfig(**_base(series={"name": "c", "saved_metric": True}))

    def test_bubble_entity_rejects_aggregate(self) -> None:
        """An aggregate makes entity metric-like; entity is a dimension."""
        with pytest.raises(ValidationError):
            BubbleChartConfig(**_base(entity={"name": "country", "aggregate": "SUM"}))

    def test_bubble_series_rejects_aggregate(self) -> None:
        with pytest.raises(ValidationError):
            BubbleChartConfig(
                **_base(series={"name": "continent", "aggregate": "COUNT"})
            )

    def test_bubble_x_accepts_saved_metric(self) -> None:
        """A saved metric is a valid x/y/size value."""
        config = BubbleChartConfig(
            **_base(x={"name": "gdp_index", "saved_metric": True})
        )
        assert config.x.saved_metric is True

    @pytest.mark.parametrize("field", ["x", "y", "size"])
    def test_bubble_metrics_reject_plain_dimension_refs(self, field: str) -> None:
        with pytest.raises(ValidationError, match=rf"{field} must define an aggregate"):
            BubbleChartConfig(**_base(**{field: {"name": "gdp"}}))

    def test_chart_config_union_dispatches_bubble(self) -> None:
        config = TypeAdapter(ChartConfig).validate_python(_base())
        assert isinstance(config, BubbleChartConfig)

    def test_native_viz_type_and_form_data_round_trip(self) -> None:
        native_form_data = map_bubble_config(
            BubbleChartConfig(
                **_base(filters=[{"column": "year", "op": "=", "value": 2026}])
            )
        )

        generate_request = GenerateChartRequest.model_validate(
            {"dataset_id": 7, "config": dict(native_form_data)}
        )
        update_request = UpdateChartRequest.model_validate(
            {"identifier": 9, "config": dict(native_form_data)}
        )

        for config in (generate_request.config, update_request.config):
            assert isinstance(config, BubbleChartConfig)
            assert config.chart_type == "bubble"
            assert config.entity.name == "country"
            assert config.x.aggregate == "AVG"
            assert config.filters is not None
            assert config.filters[0].column == "year"
            assert config.filters[0].op == "="

    def test_native_saved_and_sql_metrics_round_trip(self) -> None:
        request = UpdateChartRequest.model_validate(
            {
                "identifier": 9,
                "config": {
                    "viz_type": "bubble_v2",
                    "entity": "country",
                    "x": "saved_x",
                    "y": {
                        "expressionType": "SQL",
                        "sqlExpression": "AVG(revenue / NULLIF(cost, 0))",
                        "label": "Efficiency",
                    },
                    "size": {
                        "expressionType": "SIMPLE",
                        "column": {"column_name": "population"},
                        "aggregate": "SUM",
                        "label": "Population",
                        "hasCustomLabel": True,
                    },
                },
            }
        )

        assert isinstance(request.config, BubbleChartConfig)
        assert request.config.x.saved_metric is True
        assert request.config.y.sql_expression is not None
        assert request.config.size.label == "Population"

    @pytest.mark.parametrize("order_desc", [False, True])
    @pytest.mark.parametrize(
        "native_order_by",
        [
            "saved_order_metric",
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": "population"},
                "aggregate": "SUM",
                "label": "SUM(population)",
            },
        ],
    )
    def test_native_scalar_orderby_round_trips(
        self, native_order_by: Any, order_desc: bool
    ) -> None:
        native = map_bubble_config(BubbleChartConfig(**_base()))
        native.update({"orderby": native_order_by, "order_desc": order_desc})

        config = UpdateChartRequest.model_validate(
            {"identifier": 9, "config": native}
        ).config

        assert isinstance(config, BubbleChartConfig)
        assert config.order_by is not None
        assert config.order_by.saved_metric is isinstance(native_order_by, str)
        remapped = map_bubble_config(config)
        assert isinstance(remapped["orderby"], str) is isinstance(native_order_by, str)
        assert remapped["order_desc"] is order_desc

    @pytest.mark.parametrize("clause", [None, 7])
    def test_native_filter_clause_requires_string(self, clause: Any) -> None:
        native = map_bubble_config(BubbleChartConfig(**_base()))
        native["adhoc_filters"] = [
            {
                "expressionType": "SIMPLE",
                "clause": clause,
                "subject": "country",
                "operator": "==",
                "comparator": "France",
            }
        ]

        with pytest.raises(ValidationError, match="clause must be a string"):
            UpdateChartRequest.model_validate({"identifier": 9, "config": native})

    @pytest.mark.parametrize("invalid_filters", [{}, "bad", 17])
    def test_native_malformed_filter_container_rejected(
        self, invalid_filters: object
    ) -> None:
        with pytest.raises(ValidationError, match="adhoc_filters must be a list"):
            UpdateChartRequest.model_validate(
                {
                    "identifier": 9,
                    "config": {
                        "viz_type": "bubble_v2",
                        "entity": "country",
                        "x": "saved_x",
                        "y": "saved_y",
                        "size": "saved_size",
                        "adhoc_filters": invalid_filters,
                    },
                }
            )

    def test_native_null_filter_container_is_accepted(self) -> None:
        request = GenerateChartRequest.model_validate(
            {
                "dataset_id": 7,
                "config": {
                    "viz_type": "bubble_v2",
                    "entity": "country",
                    "x": "saved_x",
                    "y": "saved_y",
                    "size": "saved_size",
                    "adhoc_filters": None,
                },
            }
        )
        assert isinstance(request.config, BubbleChartConfig)
        assert request.config.filters is None

    def test_native_typo_rejected_without_weakening_typed_schema(self) -> None:
        with pytest.raises(ValidationError, match="opacitiy.*opacity"):
            UpdateChartRequest.model_validate(
                {
                    "identifier": 9,
                    "config": {
                        "viz_type": "bubble_v2",
                        "entity": "country",
                        "x": "saved_x",
                        "y": "saved_y",
                        "size": "saved_size",
                        "opacitiy": 0.4,
                    },
                }
            )
        with pytest.raises(ValidationError, match="Unknown field 'opacity'"):
            BubbleChartConfig.model_validate(_base(opacity=0.4))

    @pytest.mark.parametrize(
        "fixture_path",
        [
            "superset/examples/featured_charts/charts/Bubble.yaml",
            "superset/examples/world_health/charts/Life_Expectancy_VS_Rural.yaml",
        ],
    )
    @pytest.mark.parametrize(
        ("request_type", "request_fields"),
        [
            (GenerateChartRequest, {"dataset_id": 7}),
            (UpdateChartRequest, {"identifier": 9}),
        ],
    )
    def test_repository_bubble_yaml_native_params_are_accepted_and_preserved(
        self,
        fixture_path: str,
        request_type,
        request_fields: dict[str, int],
    ) -> None:
        params = yaml.safe_load(Path(fixture_path).read_text())["params"]

        request = request_type.model_validate(
            {**request_fields, "config": deepcopy(params)}
        )

        assert isinstance(request.config, BubbleChartConfig)
        remapped = map_bubble_config(request.config)
        for key in (
            "annotation_layers",
            "legendOrientation",
            "legendType",
            "max_bubble_size",
            "opacity",
            "show_legend",
            "tooltipSizeFormat",
            "truncateXAxis",
        ):
            if key in params:
                assert remapped[key] == params[key]
        if params.get("time_range"):
            assert remapped["time_range"] == params["time_range"]

    def test_cached_native_state_accepts_server_fields_and_preserves_ui(self) -> None:
        native = map_bubble_config(BubbleChartConfig(**_base()))
        native.update(
            {
                "datasource": "7__table",
                "slice_id": 9,
                "extra_form_data": {"filters": []},
                "extra_filters": [],
                "dashboardId": 4,
                "force": False,
                "granularity_sqla": "event_date",
                "queryFields": {"entity": "groupby"},
                "result_format": "json",
                "result_type": "full",
                "annotation_layers": [],
                "max_bubble_size": "75",
                "opacity": 0.4,
                "legendMargin": 12,
                "legendSort": "asc",
                "xAxisFormat": "$,.2f",
                "tooltipSizeFormat": ",.0f",
            }
        )

        request = UpdateChartRequest.model_validate({"identifier": 9, "config": native})
        assert isinstance(request.config, BubbleChartConfig)
        assert request.config.temporal_column == "event_date"
        remapped = map_bubble_config(request.config)
        assert remapped["max_bubble_size"] == "75"
        assert remapped["opacity"] == 0.4
        assert remapped["legendMargin"] == 12
        assert remapped["legendSort"] == "asc"
        assert remapped["xAxisFormat"] == "$,.2f"
        assert remapped["tooltipSizeFormat"] == ",.0f"
        assert "datasource" not in remapped

    @pytest.mark.parametrize(
        ("request_type", "request_fields"),
        [
            (GenerateChartRequest, {"dataset_id": 7}),
            (UpdateChartRequest, {"identifier": 9}),
        ],
    )
    @patch(
        "superset.mcp_service.chart.chart_utils._is_temporal_for_dashboard_binding",
        return_value=True,
    )
    @patch("superset.mcp_service.chart.chart_utils._find_dataset_by_id_or_uuid")
    def test_temporal_native_form_data_round_trips_through_requests(
        self,
        mock_find_dataset: MagicMock,
        mock_is_temporal: MagicMock,
        request_type,
        request_fields: dict[str, int],
    ) -> None:
        mock_find_dataset.return_value = Mock(main_dttm_col="ds")
        native_form_data = map_config_to_form_data(
            BubbleChartConfig(
                **_base(filters=[{"column": "year", "op": "=", "value": 2026}])
            ),
            dataset_id=7,
        )
        mock_is_temporal.assert_called()

        assert native_form_data["_mcp_dashboard_time_filter_subject"] == "ds"
        request = request_type.model_validate(
            {**request_fields, "config": deepcopy(native_form_data)}
        )
        config = request.config

        assert isinstance(config, BubbleChartConfig)
        assert config.temporal_column == "ds"
        assert config.filters is not None
        assert [(filter_.column, filter_.op) for filter_ in config.filters] == [
            ("year", "=")
        ]

        remapped = map_config_to_form_data(config, dataset_id=7)
        assert remapped["_mcp_dashboard_time_filter_subject"] == "ds"
        assert [
            (filter_["subject"], filter_["operator"])
            for filter_ in remapped["adhoc_filters"]
        ] == [("year", "=="), ("ds", "TEMPORAL_RANGE")]

    @pytest.mark.parametrize(
        ("request_type", "request_fields"),
        [
            (GenerateChartRequest, {"dataset_id": 7}),
            (UpdateChartRequest, {"identifier": 9}),
        ],
    )
    @patch(
        "superset.mcp_service.chart.chart_utils._is_temporal_for_dashboard_binding",
        return_value=True,
    )
    @patch("superset.mcp_service.chart.chart_utils._find_dataset_by_id_or_uuid")
    def test_temporal_round_trip_rejects_unrelated_native_filter(
        self,
        mock_find_dataset: MagicMock,
        mock_is_temporal: MagicMock,
        request_type,
        request_fields: dict[str, int],
    ) -> None:
        mock_find_dataset.return_value = Mock(main_dttm_col="ds")
        native_form_data = map_config_to_form_data(
            BubbleChartConfig(**_base()), dataset_id=7
        )
        mock_is_temporal.assert_called()
        native_form_data["adhoc_filters"].append(
            {
                "clause": "WHERE",
                "expressionType": "SIMPLE",
                "subject": "other_ds",
                "operator": "TEMPORAL_RANGE",
                "comparator": "No filter",
            }
        )

        with pytest.raises(ValidationError, match="TEMPORAL_RANGE"):
            request_type.model_validate(
                {**request_fields, "config": deepcopy(native_form_data)}
            )


class TestMapBubbleConfig:
    """form_data mapping must match the frontend Bubble buildQuery."""

    def test_basic_bubble_form_data(self) -> None:
        config = BubbleChartConfig(**_base())
        form_data = map_bubble_config(config)
        assert form_data["viz_type"] == "bubble_v2"
        assert form_data["entity"] == "country"
        # x/y/size are three separate metric keys (not a metrics array)
        assert form_data["x"]["label"] == "AVG(gdp)"
        assert form_data["y"]["label"] == "AVG(life_expectancy)"
        assert form_data["size"]["label"] == "SUM(population)"
        assert form_data["row_limit"] == 10000
        assert "series" not in form_data  # omitted when not set

    def test_bubble_form_data_with_series_and_filters(self) -> None:
        config = BubbleChartConfig(
            **_base(
                series={"name": "continent"},
                filters=[{"column": "year", "op": "=", "value": 2026}],
            )
        )
        form_data = map_bubble_config(config)
        assert form_data["series"] == "continent"
        assert form_data["adhoc_filters"], "filters must map to adhoc_filters"

    def test_bubble_saved_metric_maps_to_name_string(self) -> None:
        config = BubbleChartConfig(
            **_base(size={"name": "headcount", "saved_metric": True})
        )
        assert map_bubble_config(config)["size"] == "headcount"


class TestBubbleMetricsResolution:
    """The MCP query path must fold x/y/size into metrics for bubble_v2.

    The mapper emits viz_type 'bubble_v2', so resolve_metrics must recognize
    it (not just the legacy 'bubble' key) or the query drops all three metrics.
    """

    def test_bubble_v2_metrics_resolved(self) -> None:
        from superset.mcp_service.chart.chart_helpers import resolve_metrics

        form_data = map_bubble_config(BubbleChartConfig(**_base()))
        metrics = resolve_metrics(form_data, "bubble_v2")
        labels = [m["label"] if isinstance(m, dict) else m for m in metrics]
        assert labels == ["AVG(gdp)", "AVG(life_expectancy)", "SUM(population)"]

    def test_compile_and_preview_fields_include_bubble_contract(self) -> None:
        form_data = map_bubble_config(
            BubbleChartConfig(**_base(series={"name": "continent"}))
        )

        columns, metrics = _build_query_fields(form_data)
        labels = [metric["label"] for metric in metrics]

        assert columns == ["country", "continent"]
        assert labels == ["AVG(gdp)", "AVG(life_expectancy)", "SUM(population)"]

    @pytest.mark.parametrize("order_desc", [False, True])
    @pytest.mark.parametrize(
        "order_by",
        [
            "saved_order_metric",
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": "population"},
                "aggregate": "SUM",
                "label": "SUM(population)",
            },
        ],
    )
    def test_bubble_ordering_matches_query_object_schema(
        self, order_by: Any, order_desc: bool
    ) -> None:
        form_data = {"viz_type": "bubble_v2", "orderby": order_by}
        form_data["order_desc"] = order_desc
        query: dict[str, Any] = {"columns": ["country"], "metrics": ["count"]}

        apply_bubble_ordering(query, form_data)
        loaded = ChartDataQueryObjectSchema().load(query)

        assert query["orderby"] == [[order_by, not order_desc]]
        assert loaded["orderby"] == [(order_by, not order_desc)]

    def test_bubble_ordering_uses_frontend_descending_default(self) -> None:
        query: dict[str, Any] = {}

        apply_bubble_ordering(
            query,
            {"viz_type": "bubble_v2", "orderby": "saved_order_metric"},
        )

        assert query["orderby"] == [["saved_order_metric", False]]

    @patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
    @patch("superset.common.query_context_factory.QueryContextFactory")
    def test_compile_builds_non_empty_bubble_query(
        self, mock_factory_cls: MagicMock, mock_command_cls: MagicMock
    ) -> None:
        mock_factory_cls.return_value.create.return_value = MagicMock()
        mock_command_cls.return_value.run.return_value = {
            "queries": [
                {
                    "data": [
                        {
                            "country": "France",
                            "AVG(gdp)": 44000,
                            "AVG(life_expectancy)": 82.3,
                            "SUM(population)": 68_000_000,
                        }
                    ]
                }
            ]
        }
        form_data = map_bubble_config(BubbleChartConfig(**_base()))

        result = _compile_chart(form_data, dataset_id=7)

        assert result.success is True
        query = mock_factory_cls.return_value.create.call_args.kwargs["queries"][0]
        assert query["columns"] == ["country"]
        assert [metric["label"] for metric in query["metrics"]] == [
            "AVG(gdp)",
            "AVG(life_expectancy)",
            "SUM(population)",
        ]


class TestBubbleVegaLitePreview:
    """Bubble previews must preserve all visual encodings."""

    def test_bubble_preview_uses_position_size_entity_and_series(self) -> None:
        form_data = map_bubble_config(
            BubbleChartConfig(**_base(series={"name": "continent"}))
        )
        data = [
            {
                "country": "France",
                "continent": "Europe",
                "AVG(gdp)": 44000,
                "AVG(life_expectancy)": 82.3,
                "SUM(population)": 68_000_000,
            }
        ]

        preview = _generate_vega_lite_preview_from_data(data, form_data)
        spec = preview.specification

        assert spec["mark"]["type"] == "point"
        assert spec["encoding"]["x"] == {
            "field": "AVG(gdp)",
            "type": "quantitative",
            "title": "AVG(gdp)",
        }
        assert spec["encoding"]["y"]["field"] == "AVG(life_expectancy)"
        assert spec["encoding"]["size"]["field"] == "SUM(population)"
        assert spec["encoding"]["detail"] == {
            "field": "country",
            "type": "nominal",
        }
        assert spec["encoding"]["color"]["field"] == "continent"
        assert [item["field"] for item in spec["encoding"]["tooltip"]] == [
            "country",
            "continent",
            "AVG(gdp)",
            "AVG(life_expectancy)",
            "SUM(population)",
        ]

    def _saved_strategy(self, form_data: dict[str, Any]) -> VegaLitePreviewStrategy:
        chart = Mock(
            id=9,
            viz_type="bubble_v2",
            params=json.dumps(form_data),
            slice_name="Economic outlook",
            datasource_id=7,
            datasource_type="table",
        )
        return VegaLitePreviewStrategy(
            chart,
            GetChartPreviewRequest(identifier=9, format="vega_lite"),
        )

    @pytest.mark.parametrize(
        ("mutate_form_data", "data", "error_type"),
        [
            (lambda fd: fd.pop("x"), [], "InvalidChart"),
            (lambda fd: None, [], "NoDataError"),
            (
                lambda fd: None,
                [
                    {
                        "country": "France",
                        "AVG(gdp)": 44000,
                        "AVG(life_expectancy)": 82.3,
                    }
                ],
                "InvalidChartData",
            ),
            (
                lambda fd: None,
                [
                    {
                        "country": "France",
                        "AVG(gdp)": "high",
                        "AVG(life_expectancy)": 82.3,
                        "SUM(population)": 68_000_000,
                    }
                ],
                "InvalidChartData",
            ),
            (
                lambda fd: None,
                [
                    {
                        "country": "France",
                        "AVG(gdp)": True,
                        "AVG(life_expectancy)": 82.3,
                        "SUM(population)": 68_000_000,
                    }
                ],
                "InvalidChartData",
            ),
            (
                lambda fd: None,
                [
                    {
                        "country": "France",
                        "AVG(gdp)": None,
                        "AVG(life_expectancy)": 82.3,
                        "SUM(population)": 68_000_000,
                    }
                ],
                "InvalidChartData",
            ),
        ],
    )
    def test_saved_and_unsaved_bubble_preview_errors_match(
        self, mutate_form_data, data: list[dict[str, Any]], error_type: str
    ) -> None:
        form_data = map_bubble_config(BubbleChartConfig(**_base()))
        mutate_form_data(form_data)

        unsaved = _generate_vega_lite_preview_from_data(data, form_data)
        with (
            patch(
                "superset.mcp_service.chart.tool.get_chart_preview."
                "build_query_context_from_form_data",
                return_value=MagicMock(),
            ),
            patch(
                "superset.commands.chart.data.get_data_command.ChartDataCommand"
            ) as command_cls,
        ):
            command_cls.return_value.run.return_value = {"queries": [{"data": data}]}
            saved = self._saved_strategy(form_data).generate()

        assert isinstance(unsaved, ChartError)
        assert isinstance(saved, ChartError)
        assert unsaved.error_type == saved.error_type == error_type
        assert unsaved.message == saved.message

    def test_custom_metric_aliases_are_required_result_fields(self) -> None:
        form_data = map_bubble_config(
            BubbleChartConfig(
                **_base(x={"name": "gdp", "aggregate": "AVG", "label": "GDP"})
            )
        )
        valid_data = [
            {
                "country": "France",
                "GDP": 44000,
                "AVG(life_expectancy)": 82.3,
                "SUM(population)": 68_000_000,
            }
        ]
        preview = _generate_vega_lite_preview_from_data(valid_data, form_data)
        assert not isinstance(preview, ChartError)
        assert preview.specification["encoding"]["x"]["field"] == "GDP"

        invalid_data = [
            {key: value for key, value in valid_data[0].items() if key != "GDP"}
        ]
        invalid_data[0]["AVG(gdp)"] = 44000
        error = _generate_vega_lite_preview_from_data(invalid_data, form_data)
        assert isinstance(error, ChartError)
        assert error.error_type == "InvalidChartData"

    def test_saved_bubble_preview_uses_native_metric_fields(self) -> None:
        form_data = map_bubble_config(BubbleChartConfig(**_base()))
        chart = Mock(
            viz_type="bubble_v2",
            params=json.dumps(form_data),
            slice_name="Economic outlook",
        )
        strategy = VegaLitePreviewStrategy(
            chart,
            GetChartPreviewRequest(identifier=9, format="vega_lite"),
        )

        spec = strategy._create_vega_lite_spec(
            [
                {
                    "country": "France",
                    "AVG(gdp)": 44000,
                    "AVG(life_expectancy)": 82.3,
                    "SUM(population)": 68_000_000,
                }
            ]
        )

        assert spec["mark"]["type"] == "point"
        assert spec["encoding"]["x"]["field"] == "AVG(gdp)"
        assert spec["encoding"]["y"]["field"] == "AVG(life_expectancy)"
        assert spec["encoding"]["size"]["field"] == "SUM(population)"


class TestBubblePluginRegistry:
    """Plugin registration and viz-type resolution."""

    def test_bubble_plugin_registered(self) -> None:
        from superset.mcp_service.chart import registry

        plugin = registry.get("bubble")
        assert plugin is not None
        assert plugin.resolve_viz_type(None) == "bubble_v2"

    def test_display_name_resolves(self) -> None:
        from superset.mcp_service.chart.registry import display_name_for_viz_type

        assert display_name_for_viz_type("bubble_v2") == "Bubble Chart"

    def test_pre_validate_missing_fields(self) -> None:
        from superset.mcp_service.chart import registry

        plugin = registry.get("bubble")
        assert plugin is not None
        error = plugin.pre_validate({"chart_type": "bubble"})
        assert error is not None
        assert "entity" in error.message
        assert "x" in error.message

    def test_update_preserves_native_viz_type(self) -> None:
        from superset.mcp_service.chart.tool.update_chart import _build_update_payload

        config = BubbleChartConfig(**_base())
        request = UpdateChartRequest(identifier=9, config=config)
        chart = Mock(datasource_id=7, slice_name="Bubble", params="{}")

        payload = _build_update_payload(request, chart, parsed_config=config)

        assert isinstance(payload, dict)
        assert payload["viz_type"] == "bubble_v2"
        assert json.loads(payload["params"])["viz_type"] == "bubble_v2"

    def test_capabilities_and_semantics_cover_bubble(self) -> None:
        config = BubbleChartConfig(**_base())

        capabilities = analyze_chart_capabilities("bubble_v2", config)
        semantics = analyze_chart_semantics("bubble_v2", config)

        assert capabilities.supports_interaction is True
        assert set(capabilities.data_types) == {"categorical", "metric"}
        assert "three metrics" in semantics.primary_insight
        assert "country" in semantics.data_story


class TestBubbleRecommendationCategory:
    """Bubble is categorized for chart recommendations and schema discovery."""

    def test_bubble_in_recommendation_category_map(self) -> None:
        from superset.mcp_service.chart.tool.get_chart_data import _VIZ_CATEGORY

        assert _VIZ_CATEGORY.get("bubble_v2") == "bubble"

    def test_get_chart_type_schema_includes_bubble(self) -> None:
        from superset.mcp_service.chart.tool.get_chart_type_schema import (
            _CHART_TYPE_ADAPTERS,
        )

        assert "bubble" in _CHART_TYPE_ADAPTERS
