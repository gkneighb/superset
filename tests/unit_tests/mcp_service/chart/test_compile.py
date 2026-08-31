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

"""
Integration-style tests for ``validate_and_compile``.

These tests exercise the real ``DatasetValidator.validate_against_dataset``
path so fast-path tools (``generate_explore_link``, ``update_chart_preview``)
that only use Tier-1 validation are exercised end-to-end.
"""

import time
from unittest.mock import Mock, patch

import pytest

from superset.mcp_service.chart.chart_utils import map_bubble_config
from superset.mcp_service.chart.compile import (
    CompileResult,
    validate_and_compile,
)
from superset.mcp_service.chart.plugins.bubble import (
    _MAX_SQL_EXPRESSION_DEPTH,
    _MAX_SQL_EXPRESSION_LENGTH,
    _MAX_SQL_EXPRESSION_TOKENS,
    _tokenize_sql,
    bubble_metric_output_status,
    bubble_metrics_requiring_query_validation,
)
from superset.mcp_service.chart.schemas import (
    BigNumberChartConfig,
    BubbleChartConfig,
    ColumnRef,
    FilterConfig,
    PieChartConfig,
    PivotTableChartConfig,
    TableChartConfig,
    XYChartConfig,
)
from superset.mcp_service.chart.validation.dataset_validator import (
    build_dataset_context_from_orm,
)


def _orm_dataset(
    *,
    column_names: list[str] | None = None,
    metric_names: list[str] | None = None,
    has_database: bool = True,
) -> Mock:
    """Build a Mock dataset that satisfies build_dataset_context_from_orm."""
    columns = []
    for name in column_names or ["ds", "gender", "name", "num"]:
        col = Mock()
        col.column_name = name
        col.type = "TEXT"
        col.is_temporal = name == "ds"
        col.is_numeric = name == "num"
        columns.append(col)

    metrics = []
    for name in metric_names or ["sum_boys", "sum_girls"]:
        m = Mock()
        m.metric_name = name
        m.expression = f"SUM({name})"
        m.description = None
        metrics.append(m)

    dataset = Mock()
    dataset.id = 3
    dataset.table_name = "birth_names"
    dataset.schema = None
    dataset.columns = columns
    dataset.metrics = metrics
    if has_database:
        db = Mock()
        db.database_name = "examples"
        dataset.database = db
    else:
        dataset.database = None
    return dataset


class TestBuildDatasetContextFromOrm:
    """Cover the helper that converts ORM dataset → DatasetContext."""

    def test_handles_missing_database_relationship(self):
        """``database_name`` defaults to '' when ``dataset.database`` is None
        so Pydantic validation doesn't blow up."""
        ds = _orm_dataset(has_database=False)
        ctx = build_dataset_context_from_orm(ds)
        assert ctx is not None
        assert ctx.database_name == ""
        assert ctx.id == 3
        assert {c["name"] for c in ctx.available_columns} == {
            "ds",
            "gender",
            "name",
            "num",
        }
        assert {m["name"] for m in ctx.available_metrics} == {
            "sum_boys",
            "sum_girls",
        }

    def test_returns_none_for_none_input(self):
        assert build_dataset_context_from_orm(None) is None


class TestValidateAndCompileChartTypeCoverage:
    """Tier-1 validation must catch bad column refs in every supported
    chart-config variant — not just XY and table."""

    def test_xy_bad_metric_column_rejected(self):
        ds = _orm_dataset()
        config = XYChartConfig(
            chart_type="xy",
            x=ColumnRef(name="ds"),
            y=[ColumnRef(name="num_boys", aggregate="SUM")],
            kind="line",
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success
        assert result.tier == "validation"
        assert result.error_obj is not None
        assert any("sum_boys" in s for s in (result.error_obj.suggestions or []))

    def test_pie_bad_metric_column_rejected(self):
        ds = _orm_dataset()
        config = PieChartConfig(
            dimension=ColumnRef(name="gender"),
            metric=ColumnRef(name="num_boys", aggregate="SUM"),
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success, "Pie chart with bad metric column should fail"
        assert result.tier == "validation"
        assert result.error_obj is not None
        assert any("sum_boys" in s for s in (result.error_obj.suggestions or []))

    def test_pie_valid_dimension_and_saved_metric_passes(self):
        ds = _orm_dataset()
        config = PieChartConfig(
            dimension=ColumnRef(name="gender"),
            metric=ColumnRef(name="sum_boys", saved_metric=True),
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success, result.error

    @pytest.mark.parametrize("field", ["x", "y", "size"])
    def test_bubble_sum_on_non_numeric_column_rejected(self, field):
        ds = _orm_dataset()
        metrics = {
            "x": ColumnRef(name="num", aggregate="AVG"),
            "y": ColumnRef(name="num", aggregate="MAX"),
            "size": ColumnRef(name="num", aggregate="SUM"),
        }
        metrics[field] = ColumnRef(name="gender", aggregate="SUM")
        config = BubbleChartConfig(entity=ColumnRef(name="name"), **metrics)

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_AGGREGATION"

    def test_bubble_numeric_metrics_pass(self):
        ds = _orm_dataset()
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="num", aggregate="AVG"),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success, result.error

    @pytest.mark.parametrize(
        "sql_type",
        ["INT2", "INT4", "INT8", "FLOAT8", "MEDIUMINT", "SMALLMONEY"],
    )
    def test_bubble_uses_shared_numeric_sql_type_detection(self, sql_type):
        ds = _orm_dataset()
        numeric = next(column for column in ds.columns if column.column_name == "num")
        numeric.is_numeric = False
        numeric.type = sql_type
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="num", aggregate="AVG"),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success, result.error

    @pytest.mark.parametrize("aggregate", ["MIN", "MAX"])
    @pytest.mark.parametrize("field", ["x", "y", "size"])
    def test_bubble_min_max_on_text_rejected(self, aggregate, field):
        ds = _orm_dataset()
        metrics = {
            "x": ColumnRef(name="num", aggregate="AVG"),
            "y": ColumnRef(name="num", aggregate="MAX"),
            "size": ColumnRef(name="num", aggregate="SUM"),
        }
        metrics[field] = ColumnRef(name="gender", aggregate=aggregate)

        result = validate_and_compile(
            BubbleChartConfig(entity=ColumnRef(name="name"), **metrics),
            {},
            ds,
            run_compile_check=False,
        )

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_BUBBLE_METRIC_OUTPUT"

    @pytest.mark.parametrize("aggregate", ["COUNT", "COUNT_DISTINCT"])
    def test_bubble_count_over_text_is_numeric(self, aggregate):
        ds = _orm_dataset()
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="gender", aggregate=aggregate),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success, result.error

    @pytest.mark.parametrize(
        ("metric", "passes"),
        [
            (ColumnRef(name="saved_text", saved_metric=True), False),
            (ColumnRef(name="saved_count", saved_metric=True), True),
            (ColumnRef(name="saved_sum", saved_metric=True), True),
            (ColumnRef(sql_expression="MAX(gender)", label="Text max"), False),
            (ColumnRef(sql_expression="COUNT(gender)", label="Count"), True),
            (
                ColumnRef(sql_expression="SUM(num)", label="Custom numeric alias"),
                True,
            ),
        ],
    )
    @pytest.mark.parametrize("field", ["x", "y", "size"])
    def test_bubble_saved_and_sql_metric_output_inference(self, metric, passes, field):
        ds = _orm_dataset(metric_names=["saved_text", "saved_count", "saved_sum"])
        expressions = {
            "saved_text": "MAX(gender)",
            "saved_count": "COUNT(gender)",
            "saved_sum": "SUM(num)",
        }
        for saved_metric in ds.metrics:
            saved_metric.expression = expressions[saved_metric.metric_name]
            saved_metric.metric_type = None
            saved_metric.d3format = None
        metrics = {
            "x": ColumnRef(name="num", aggregate="AVG"),
            "y": ColumnRef(name="num", aggregate="MAX"),
            "size": ColumnRef(name="num", aggregate="SUM"),
        }
        metrics[field] = metric
        config = BubbleChartConfig(entity=ColumnRef(name="name"), **metrics)

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success is passes
        if not passes:
            assert result.error_obj is not None
            assert result.error_obj.error_code == "INVALID_BUBBLE_METRIC_OUTPUT"

    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_unproven_saved_metric_forces_query_on_tier_one_path(self, mock_compile):
        ds = _orm_dataset(metric_names=["complex_metric"])
        ds.metrics[0].expression = "SOME_VENDOR_FUNCTION(gender)"
        ds.metrics[0].metric_type = "sum"
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="complex_metric", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_compile.return_value = CompileResult(success=True)

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success
        mock_compile.assert_called_once_with(
            {}, ds.id, bubble_runtime_validation_required=True
        )

    @patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
    @patch("superset.common.query_context_factory.QueryContextFactory")
    def test_empty_bubble_compile_passes_with_static_numeric_proof(
        self, mock_factory, mock_command
    ):
        ds = _orm_dataset()
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="num", aggregate="AVG"),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_factory.return_value.create.return_value = Mock()
        mock_command.return_value.run.return_value = {"queries": [{"data": []}]}

        result = validate_and_compile(
            config,
            map_bubble_config(config),
            ds,
            run_compile_check=True,
        )

        assert result.success, result.error
        assert result.row_count == 0

    @patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
    @patch("superset.common.query_context_factory.QueryContextFactory")
    def test_empty_bubble_compile_fails_when_runtime_proof_is_required(
        self, mock_factory, mock_command
    ):
        ds = _orm_dataset(metric_names=["complex_metric"])
        ds.metrics[0].expression = "SOME_VENDOR_FUNCTION(gender)"
        ds.metrics[0].metric_type = "sum"
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="complex_metric", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_factory.return_value.create.return_value = Mock()
        mock_command.return_value.run.return_value = {"queries": [{"data": []}]}

        result = validate_and_compile(
            config,
            map_bubble_config(config),
            ds,
            run_compile_check=False,
        )

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_BUBBLE_QUERY_DATA"
        assert "could not be verified" in result.error_obj.message

    @pytest.mark.parametrize(
        "expression",
        [
            "SUM(CAST(name AS VARCHAR /* INT */))",
            "AVG(CAST(name AS TEXT -- DOUBLE\n))",
            "MEDIAN(CAST(name AS VARCHAR /* DECIMAL */))",
            "STDDEV(CAST(name AS VARCHAR /* DECIMAL */))",
            "STDDEV_SAMP(CAST(name AS VARCHAR /* DECIMAL */))",
            "VAR(CAST(name AS VARCHAR /* DECIMAL */))",
            "VAR_SAMP(CAST(name AS VARCHAR /* DECIMAL */))",
            "PERCENTILE(CAST(name AS VARCHAR /* DECIMAL */), 0.5)",
            "SUM(((CAST(name AS VARCHAR /* misleading ) INT ( */))))",
        ],
    )
    def test_bubble_numeric_aggregate_requires_proven_nested_argument(
        self, expression: str
    ) -> None:
        """Aggregate wrappers must not hide ambiguous nested CAST targets."""
        ds = _orm_dataset(metric_names=["nested_cast"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        context = build_dataset_context_from_orm(ds)
        assert context is not None
        metric = ColumnRef(name="nested_cast", saved_metric=True)
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=metric,
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )

        assert bubble_metric_output_status(metric, context) == "unknown"
        assert bubble_metrics_requiring_query_validation(config, context) == ["x"]

    @pytest.mark.parametrize(
        "expression",
        [
            "SUM(CAST(name AS VARCHAR /* INT */))",
            "AVG(CAST(name AS TEXT -- DOUBLE\n))",
        ],
    )
    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_bubble_nested_commented_cast_forces_query_on_no_compile_path(
        self, mock_compile: Mock, expression: str
    ) -> None:
        """A requested no-query path must execute when nested proof is unknown."""
        ds = _orm_dataset(metric_names=["nested_cast"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="nested_cast", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_compile.return_value = CompileResult(success=True)

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success
        mock_compile.assert_called_once_with(
            {}, ds.id, bubble_runtime_validation_required=True
        )

    @pytest.mark.parametrize(
        "expression",
        [
            "SUM(AVG(num))",
            "MAX(SUM(num))",
            "SUM(AVG(CAST(num AS DECIMAL(10, 2))))",
        ],
    )
    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_bubble_nested_set_aggregate_forces_query_on_no_compile_path(
        self, mock_compile: Mock, expression: str
    ) -> None:
        """Invalid nested set aggregates must not receive static numeric proof."""
        ds = _orm_dataset(metric_names=["nested_aggregate"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="nested_aggregate", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_compile.return_value = CompileResult(success=True)

        result = validate_and_compile(config, {}, ds, run_compile_check=False)

        assert result.success
        mock_compile.assert_called_once_with(
            {}, ds.id, bubble_runtime_validation_required=True
        )

    @pytest.mark.parametrize(
        "expression",
        [
            "SUM(AVG(num))",
            "MAX(SUM(num))",
            "SUM(AVG(CAST(num AS DECIMAL(10, 2))))",
        ],
    )
    @patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
    @patch("superset.common.query_context_factory.QueryContextFactory")
    def test_empty_bubble_nested_set_aggregate_fails_closed(
        self,
        mock_factory: Mock,
        mock_command: Mock,
        expression: str,
    ) -> None:
        """An empty result cannot validate an invalid nested set aggregate."""
        ds = _orm_dataset(metric_names=["nested_aggregate"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="nested_aggregate", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_factory.return_value.create.return_value = Mock()
        mock_command.return_value.run.return_value = {"queries": [{"data": []}]}

        result = validate_and_compile(
            config,
            map_bubble_config(config),
            ds,
            run_compile_check=False,
        )

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_BUBBLE_QUERY_DATA"
        assert "could not be verified" in result.error_obj.message

    @pytest.mark.parametrize(
        "expression",
        [
            "SUM(" * 1_200 + "num" + ")" * 1_200,
            "(" * 4_000 + "SUM(num)" + ")" * 4_000,
        ],
    )
    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_long_saved_metric_fails_to_runtime_proof_quickly(
        self, mock_compile: Mock, expression: str
    ) -> None:
        """MediumText metrics beyond ColumnRef's limit remain bounded and safe."""
        assert len(expression) > 2_000
        ds = _orm_dataset(metric_names=["deep_metric"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="deep_metric", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_compile.return_value = CompileResult(success=True)

        started = time.process_time()
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        elapsed = time.process_time() - started

        assert result.success
        assert elapsed < 1.0
        mock_compile.assert_called_once_with(
            {}, ds.id, bubble_runtime_validation_required=True
        )

    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_saved_metric_depth_and_length_budget_boundaries(
        self, mock_compile: Mock
    ) -> None:
        """The documented static-inference depth and work bounds fail closed."""
        ds = _orm_dataset(metric_names=["boundary_metric"])
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="boundary_metric", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )

        within_depth = (
            "(" * (_MAX_SQL_EXPRESSION_DEPTH - 1)
            + "SUM(num)"
            + ")" * (_MAX_SQL_EXPRESSION_DEPTH - 1)
        )
        ds.metrics[0].expression = within_depth
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success
        mock_compile.assert_not_called()

        beyond_depth = f"({within_depth})"
        ds.metrics[0].expression = beyond_depth
        mock_compile.return_value = CompileResult(success=True)
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success
        mock_compile.assert_called_once_with(
            {}, ds.id, bubble_runtime_validation_required=True
        )

        mock_compile.reset_mock()
        within_length = "SUM(num)".ljust(_MAX_SQL_EXPRESSION_LENGTH)
        ds.metrics[0].expression = within_length
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success
        mock_compile.assert_not_called()

        ds.metrics[0].expression = within_length + " "
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success
        mock_compile.assert_called_once_with(
            {}, ds.id, bubble_runtime_validation_required=True
        )

    def test_sql_token_complexity_budget_boundary(self) -> None:
        """The tokenizer accepts its exact token budget and rejects one more."""
        at_budget = ",".join("x" for _ in range(_MAX_SQL_EXPRESSION_TOKENS // 2))
        at_budget += ","
        parsed = _tokenize_sql(at_budget)
        assert parsed is not None
        assert len(parsed.tokens) == _MAX_SQL_EXPRESSION_TOKENS
        assert _tokenize_sql(f"{at_budget}x") is None

    @pytest.mark.parametrize(
        ("expression", "expected_status"),
        [
            ("SUM(num)", "numeric"),
            ("AVG((num))", "numeric"),
            ("AVG(CAST(num AS DECIMAL))", "numeric"),
            ("SUM(CAST(num AS DECIMAL(10, 2)))", "numeric"),
            ("AVG(TRY_CAST(num AS DOUBLE PRECISION))", "numeric"),
            ("SUM(CAST(num AS INT8))", "numeric"),
            ("AVG(TRY_CAST(num AS FLOAT8))", "numeric"),
            (
                "SUM(CAST(COALESCE(num, '/* VARCHAR ) */ -- TEXT (') "
                "AS DECIMAL(10, 2)))",
                "numeric",
            ),
            ("CAST(MAX(name) AS VARCHAR)", "nonnumeric"),
            ("SUM(CAST(name AS VARCHAR))", "nonnumeric"),
        ],
    )
    def test_bubble_nested_aggregate_static_proofs(
        self, expression: str, expected_status: str
    ) -> None:
        """Clean numeric proofs survive nesting and quoted comment-like text."""
        ds = _orm_dataset(metric_names=["static_proof"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        context = build_dataset_context_from_orm(ds)
        assert context is not None
        metric = ColumnRef(name="static_proof", saved_metric=True)

        assert bubble_metric_output_status(metric, context) == expected_status

    @pytest.mark.parametrize(
        ("metric_kind", "expression"),
        [
            ("saved", "COUNT(*) || COALESCE(MAX(name), '')"),
            ("saved", "COUNT(*) > COALESCE(MAX(num), 0)"),
            ("adhoc", "SUM(num) || MAX(name)"),
            ("adhoc", "SUM(num) > COALESCE(MAX(num), 0)"),
        ],
    )
    @patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
    @patch("superset.common.query_context_factory.QueryContextFactory")
    def test_empty_bubble_compound_sql_requires_runtime_numeric_proof(
        self,
        mock_factory: Mock,
        mock_command: Mock,
        metric_kind: str,
        expression: str,
    ) -> None:
        """String/boolean compounds must not inherit a numeric function's type."""
        ds = _orm_dataset(
            metric_names=["compound_metric"] if metric_kind == "saved" else None
        )
        if metric_kind == "saved":
            ds.metrics[0].expression = expression
            ds.metrics[0].metric_type = None
            ds.metrics[0].d3format = None
            metric = ColumnRef(name="compound_metric", saved_metric=True)
        else:
            metric = ColumnRef(sql_expression=expression, label="Compound metric")
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=metric,
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_factory.return_value.create.return_value = Mock()
        mock_command.return_value.run.return_value = {"queries": [{"data": []}]}

        result = validate_and_compile(
            config,
            map_bubble_config(config),
            ds,
            run_compile_check=False,
        )

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_BUBBLE_QUERY_DATA"
        assert "could not be verified" in result.error_obj.message

    @pytest.mark.parametrize(
        ("expression", "expected_error"),
        [
            ("CAST(MAX(name) AS VARCHAR /* INT */)", "INVALID_BUBBLE_QUERY_DATA"),
            (
                "CAST(MAX(name) AS BOOLEAN /* DECIMAL */)",
                "INVALID_BUBBLE_QUERY_DATA",
            ),
            (
                "TRY_CAST(MAX(name) AS TEXT /* DOUBLE */)",
                "INVALID_BUBBLE_QUERY_DATA",
            ),
            (
                "CAST(MAX(name) AS VARCHAR -- INT\n)",
                "INVALID_BUBBLE_QUERY_DATA",
            ),
            ("CAST(MAX(name) AS VARCHAR INT)", "INVALID_BUBBLE_QUERY_DATA"),
            ("CAST(MAX(name) AS VARCHAR)", "INVALID_BUBBLE_METRIC_OUTPUT"),
            (
                "CAST(COALESCE(MAX(name), '/* INT */ -- DECIMAL') AS TEXT)",
                "INVALID_BUBBLE_METRIC_OUTPUT",
            ),
            (
                "SUM(CAST(name AS VARCHAR /* INT */))",
                "INVALID_BUBBLE_QUERY_DATA",
            ),
            (
                "AVG(CAST(name AS TEXT -- DOUBLE\n))",
                "INVALID_BUBBLE_QUERY_DATA",
            ),
            (
                "SUM(((CAST(name AS VARCHAR /* misleading ) INT ( */))))",
                "INVALID_BUBBLE_QUERY_DATA",
            ),
            ("SUM(CAST(name AS VARCHAR))", "INVALID_BUBBLE_METRIC_OUTPUT"),
            ("CAST(MAX(num) AS INT)", None),
            ("TRY_CAST(MAX(num) AS DOUBLE PRECISION)", None),
            ("CAST((COALESCE(MAX(num), 0)) AS DECIMAL(10, 2))", None),
            ("SUM(num)", None),
            ("AVG(CAST(num AS DECIMAL(10, 2)))", None),
            (
                "SUM(CAST(COALESCE(num, '/* VARCHAR ) */ -- TEXT (') "
                "AS DECIMAL(10, 2)))",
                None,
            ),
        ],
    )
    @patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
    @patch("superset.common.query_context_factory.QueryContextFactory")
    def test_empty_bubble_saved_metric_cast_inference_is_unambiguous(
        self,
        mock_factory: Mock,
        mock_command: Mock,
        expression: str,
        expected_error: str | None,
    ) -> None:
        """Only simple, comment-free CAST targets provide static type proof."""
        ds = _orm_dataset(metric_names=["cast_metric"])
        ds.metrics[0].expression = expression
        ds.metrics[0].metric_type = None
        ds.metrics[0].d3format = None
        config = BubbleChartConfig(
            entity=ColumnRef(name="name"),
            x=ColumnRef(name="cast_metric", saved_metric=True),
            y=ColumnRef(name="num", aggregate="MAX"),
            size=ColumnRef(name="num", aggregate="SUM"),
        )
        mock_factory.return_value.create.return_value = Mock()
        mock_command.return_value.run.return_value = {"queries": [{"data": []}]}

        result = validate_and_compile(
            config,
            map_bubble_config(config),
            ds,
            run_compile_check=True,
        )

        assert result.success is (expected_error is None)
        if expected_error is not None:
            assert result.error_obj is not None
            assert result.error_obj.error_code == expected_error

    def test_pivot_table_bad_row_rejected(self):
        ds = _orm_dataset()
        config = PivotTableChartConfig(
            rows=[ColumnRef(name="bogus_dim")],
            metrics=[ColumnRef(name="sum_boys", saved_metric=True)],
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success
        assert result.error_obj is not None

    def test_big_number_bad_temporal_column_rejected(self):
        ds = _orm_dataset()
        config = BigNumberChartConfig(
            chart_type="big_number",
            metric=ColumnRef(name="sum_boys", saved_metric=True),
            temporal_column="not_a_real_temporal",
            show_trendline=True,
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success, "BigNumber temporal_column must be validated"
        assert result.error_obj is not None
        assert "not_a_real_temporal" in (result.error_obj.message or "")

    def test_pie_with_sum_on_non_numeric_column_rejected(self):
        """Tier-1 aggregation compatibility now runs for non-Table/XY too —
        a pie ``metric={"name": "gender", "aggregate": "SUM"}`` would emit
        ``SUM(gender)`` which the DB rejects, so the validator must catch it
        before we hand back an explore URL."""
        ds = _orm_dataset()
        config = PieChartConfig(
            dimension=ColumnRef(name="name"),
            metric=ColumnRef(name="gender", aggregate="SUM"),
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success, "SUM on a TEXT column must reject"
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_AGGREGATION"

    def test_pivot_table_sum_on_non_numeric_column_rejected(self):
        ds = _orm_dataset()
        config = PivotTableChartConfig(
            rows=[ColumnRef(name="gender")],
            metrics=[ColumnRef(name="name", aggregate="SUM")],
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "INVALID_AGGREGATION"

    def test_pivot_table_min_on_non_numeric_column_passes(self):
        """MIN and MAX are not numeric-only (valid on dates/text in SQL).

        They are left to the Tier-2 compile check rather than being rejected
        by Tier-1 schema validation.
        """
        ds = _orm_dataset()
        config = PivotTableChartConfig(
            rows=[ColumnRef(name="gender")],
            metrics=[ColumnRef(name="name", aggregate="MIN")],
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success, (
            "MIN on a text column should not be rejected by Tier-1 validation"
        )

    def test_table_with_invalid_filter_column_rejected(self):
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table",
            columns=[ColumnRef(name="gender")],
            filters=[FilterConfig(column="bogus", op="=", value="x")],
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success
        assert result.error_obj is not None

    def test_inert_stale_filter_column_is_ignored(self):
        """A No filter placeholder produces no predicate and cannot block edits."""
        ds = _orm_dataset()
        config = TableChartConfig(columns=[ColumnRef(name="gender")])
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "subject": "dropped_column",
                    "operator": "TEMPORAL_RANGE",
                    "comparator": "No filter",
                }
            ]
        }

        result = validate_and_compile(config, form_data, ds, run_compile_check=False)

        assert result.success

    def test_no_filter_literal_with_non_temporal_operator_is_validated(self):
        """A literal value of No filter is not generally an inert predicate."""
        ds = _orm_dataset()
        config = TableChartConfig(columns=[ColumnRef(name="gender")])
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "subject": "dropped_column",
                    "operator": "==",
                    "comparator": "No filter",
                }
            ]
        }

        result = validate_and_compile(config, form_data, ds, run_compile_check=False)

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_type == "invalid_column"


class TestSavedMetricNotMarked:
    """A non-saved-metric ColumnRef whose name matches a saved metric is a
    common LLM mistake (forgetting to set ``saved_metric=true``). The
    validator should surface a tailored hint instead of letting the bad SQL
    through."""

    def test_table_metric_name_without_saved_metric_flag_rejected(self):
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table",
            columns=[
                ColumnRef(name="gender"),
                # ``sum_boys`` is a saved metric on the dataset, but
                # saved_metric=False (default) would render as
                # ``SUM(sum_boys)`` ad-hoc SQL — broken.
                ColumnRef(name="sum_boys", aggregate="SUM"),
            ],
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success, (
            "ref.name matches a saved metric but saved_metric=False -> reject"
        )
        assert result.error_obj is not None
        assert result.error_obj.error_code == "SAVED_METRIC_NOT_MARKED"
        # Suggestion should point the LLM at the right correction.
        suggestions_text = " ".join(result.error_obj.suggestions or [])
        assert "saved_metric" in suggestions_text
        assert "sum_boys" in suggestions_text

    def test_pie_metric_name_without_saved_metric_flag_rejected(self):
        ds = _orm_dataset()
        config = PieChartConfig(
            dimension=ColumnRef(name="gender"),
            metric=ColumnRef(name="sum_boys", aggregate="SUM"),
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_code == "SAVED_METRIC_NOT_MARKED"

    def test_explicit_saved_metric_passes(self):
        ds = _orm_dataset()
        config = PieChartConfig(
            dimension=ColumnRef(name="gender"),
            metric=ColumnRef(name="sum_boys", saved_metric=True),
        )
        result = validate_and_compile(config, {}, ds, run_compile_check=False)
        assert result.success, result.error


class TestAdhocFiltersFromFormData:
    """Filters merged into form_data (not present on the typed config) must
    also be validated. Without this hook, ``update_chart_preview`` could
    smuggle bad column refs through preserved adhoc filters."""

    def test_unknown_adhoc_filter_subject_rejected(self):
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table", columns=[ColumnRef(name="gender")]
        )
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "subject": "removed_column",
                    "operator": "==",
                    "comparator": "x",
                }
            ]
        }
        result = validate_and_compile(config, form_data, ds, run_compile_check=False)
        assert not result.success
        assert result.error_obj is not None
        assert "removed_column" in (result.error_obj.message or "")

    def test_known_adhoc_filter_subject_passes(self):
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table", columns=[ColumnRef(name="gender")]
        )
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "subject": "gender",
                    "operator": "==",
                    "comparator": "boy",
                }
            ]
        }
        result = validate_and_compile(config, form_data, ds, run_compile_check=False)
        assert result.success, result.error

    @pytest.mark.parametrize("clause", [None, 7])
    def test_malformed_cached_filter_clause_is_actionable(self, clause):
        ds = _orm_dataset()
        config = TableChartConfig(columns=[ColumnRef(name="gender")])
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "clause": clause,
                    "subject": "gender",
                    "operator": "==",
                    "comparator": "boy",
                }
            ]
        }

        result = validate_and_compile(config, form_data, ds, run_compile_check=False)

        assert not result.success
        assert result.error_obj is not None
        assert result.error_obj.error_type == "invalid_filter"
        assert "clause must be a string" in result.error_obj.message

    def test_sql_expression_filter_skipped(self):
        """SQL-expression filters carry a free-form ``sqlExpression`` we can't
        safely parse, so they should pass Tier-1 untouched."""
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table", columns=[ColumnRef(name="gender")]
        )
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SQL",
                    "clause": "WHERE",
                    "sqlExpression": "1 = 1",
                }
            ]
        }
        result = validate_and_compile(config, form_data, ds, run_compile_check=False)
        assert result.success

    def test_where_filter_with_metric_name_rejected(self):
        """A saved-metric name used as a WHERE filter subject must be rejected.

        WHERE filters need a physical column; metric names are only valid in
        HAVING clauses where Superset can resolve them.
        """
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table", columns=[ColumnRef(name="gender")]
        )
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "clause": "WHERE",
                    "subject": "sum_boys",  # saved metric, not a physical column
                    "operator": ">",
                    "comparator": "0",
                }
            ]
        }
        result = validate_and_compile(config, form_data, ds, run_compile_check=False)
        assert not result.success, (
            "A saved-metric name used in a WHERE filter must not pass Tier-1"
        )
        assert result.error_obj is not None
        assert "sum_boys" in (result.error_obj.message or "")

    def test_having_filter_with_metric_name_passes(self):
        """A saved-metric name used in a HAVING filter must be accepted.

        HAVING filters are aggregate-level conditions; Superset resolves metric
        names there so they are valid references.
        """
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table", columns=[ColumnRef(name="gender")]
        )
        form_data = {
            "adhoc_filters": [
                {
                    "expressionType": "SIMPLE",
                    "clause": "HAVING",
                    "subject": "sum_boys",  # saved metric — valid in HAVING
                    "operator": ">",
                    "comparator": "0",
                }
            ]
        }
        result = validate_and_compile(config, form_data, ds, run_compile_check=False)
        assert result.success, (
            "A saved-metric name in a HAVING filter should pass Tier-1 validation"
        )


class TestValidateAndCompileTier2:
    """When ``run_compile_check=True`` and Tier-1 passes, the helper must
    invoke ``_compile_chart`` and surface its outcome."""

    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_tier2_runs_when_tier1_passes(self, mock_compile):
        mock_compile.return_value = CompileResult(success=True)
        ds = _orm_dataset()
        config = TableChartConfig(
            chart_type="table", columns=[ColumnRef(name="gender")]
        )
        result = validate_and_compile(
            config, {"adhoc_filters": []}, ds, run_compile_check=True
        )
        assert result.success
        mock_compile.assert_called_once()

    @patch("superset.mcp_service.chart.compile._compile_chart")
    def test_tier2_skipped_on_tier1_failure(self, mock_compile):
        ds = _orm_dataset()
        config = TableChartConfig(chart_type="table", columns=[ColumnRef(name="bogus")])
        result = validate_and_compile(config, {}, ds, run_compile_check=True)
        assert not result.success
        assert result.tier == "validation"
        mock_compile.assert_not_called()

    def test_dataset_none_returns_dataset_not_found(self):
        result = validate_and_compile(None, {}, None, run_compile_check=True)
        assert not result.success
        assert result.error_code == "DATASET_NOT_FOUND"


@patch("superset.daos.dataset.DatasetDAO")
@patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
@patch("superset.common.query_context_factory.QueryContextFactory")
def test_compile_chart_returns_database_error_when_wrapped_in_query_failed(
    mock_factory, mock_cmd_cls, mock_dataset_dao
):
    """ChartDataCommand converts OperationalError to a string inside
    ChartDataQueryFailedError (no __cause__ set). _classify_as_database_error
    should use db_engine_spec.extract_errors() to detect the DB error."""
    from superset.commands.chart.exceptions import ChartDataQueryFailedError
    from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
    from superset.mcp_service.chart.compile import _compile_chart

    mock_factory.return_value.create.return_value = Mock()
    mock_cmd_cls.return_value.validate.return_value = None

    # Real scenario: __cause__ is NOT set, error is just a string
    wrapped = ChartDataQueryFailedError(
        "Error: (psycopg2.OperationalError) connection to server at '10.0.0.1',"
        " port 5432 failed: FATAL: tenant not found"
    )
    mock_cmd_cls.return_value.run.side_effect = wrapped

    # Mock the dataset's db_engine_spec to return GENERIC_DB_ENGINE_ERROR
    mock_db = Mock()
    mock_db.db_engine_spec.extract_errors.return_value = [
        SupersetError(
            error_type=SupersetErrorType.GENERIC_DB_ENGINE_ERROR,
            message="connection to server failed",
            level=ErrorLevel.ERROR,
            extra={"engine_name": "PostgreSQL"},
        )
    ]
    mock_dataset = Mock()
    mock_dataset.database = mock_db
    mock_dataset_dao.find_by_id.return_value = mock_dataset

    result = _compile_chart(
        form_data={
            "metrics": [{"label": "count", "expressionType": "SIMPLE"}],
            "adhoc_filters": [],
        },
        dataset_id=1,
    )

    assert not result.success
    assert "Database connection error" in result.error
    assert result.error_obj is not None
    assert result.error_obj.error_type == "database_connection_error"
    assert result.error_obj.error_code == "DATABASE_CONNECTION_ERROR"
    mock_db.db_engine_spec.extract_errors.assert_called_once()


@patch("superset.commands.chart.data.get_data_command.ChartDataCommand")
@patch("superset.common.query_context_factory.QueryContextFactory")
def test_compile_chart_returns_database_error_on_raw_sqlalchemy_error(
    mock_factory, mock_cmd_cls
):
    """When SQLAlchemyError escapes unwrapped, _compile_chart should
    catch it and return a database_connection_error."""
    from sqlalchemy.exc import OperationalError

    from superset.mcp_service.chart.compile import _compile_chart

    mock_factory.return_value.create.return_value = Mock()
    mock_cmd_cls.return_value.validate.return_value = None
    mock_cmd_cls.return_value.run.side_effect = OperationalError(
        "connection to server at '10.0.0.1', port 5432 failed: Connection timed out",
        None,
        None,
    )

    result = _compile_chart(
        form_data={
            "metrics": [{"label": "count", "expressionType": "SIMPLE"}],
            "adhoc_filters": [],
        },
        dataset_id=1,
    )

    assert not result.success
    assert "Database connection error" in result.error
    assert result.error_obj is not None
    assert result.error_obj.error_type == "database_connection_error"
    assert result.error_obj.error_code == "DATABASE_CONNECTION_ERROR"


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: PieChartConfig(
            dimension=ColumnRef(name="gender"),
            metric=ColumnRef(name="sum_boys", saved_metric=True),
        ),
        lambda: TableChartConfig(
            chart_type="table",
            columns=[
                ColumnRef(name="gender"),
                ColumnRef(name="sum_boys", saved_metric=True),
            ],
        ),
    ],
)
def test_valid_configs_pass_tier1(config_factory):
    ds = _orm_dataset()
    result = validate_and_compile(config_factory(), {}, ds, run_compile_check=False)
    assert result.success, result.error
