# stdlib
import base64
import json
from enum import Enum
from typing import Any, Dict, List, Optional

# third party
import plotly.express as px
import pyarrow as pa
import streamlit as st
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field
from snowflake.snowpark.context import get_active_session

st.set_page_config(layout="wide")

CHART_TYPE_FIELDS = {
    "line": ["x", "y", "color", "facet_row", "facet_col", "y2"],
    "bar": ["x", "y", "color", "orientation", "barmode", "y2"],
    "pie": ["values", "names"],
    "area": ["x", "y", "color", "y2"],
    "scatter": ["x", "y", "color", "size", "facet_col", "facet_row", "trendline"],
    "histogram": ["x", "nbins", "histfunc"],
}

EXAMPLE_PROMPT = """
The result should only contain a dictionary - nothing more!

Available metrics: {metrics}.
Available dimensions: {dimensions}.

User question: {question}
Result: {result}
"""



def _can_add_field(selections, available):
    return len(selections) < len(available)


def _available_options(selections, available):
    return [option for option in available if option not in selections]


def _sort_dataframe(df, query):
    try:
        time_dimensions = [
            col for col in df.columns if col in query.time_dimension_names
        ]
    except KeyError:
        return df
    else:
        if len(time_dimensions) > 0:
            col = time_dimensions[0]
            is_sorted = df[col].is_monotonic_increasing
            if not is_sorted:
                df = df.sort_values(by=col)
        return df


def _add_secondary_yaxis(df, fig, dct):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    chart_map = {
        "line": "Scatter",
        "bar": "Bar",
        "area": "Scatter",
    }

    new_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # add traces from plotly express figure to first figure
    for t in fig.select_traces():
        new_fig.add_trace(t, secondary_y=False)

    addl_config = {}
    if dct["chart_type"] == "line":
        addl_config["mode"] = "lines"
    elif dct["chart_type"] == "area":
        addl_config["fill"] = "tozeroy"

    new_fig.add_trace(
        getattr(go, chart_map[dct["chart_type"]])(
            x=df[dct["x"]], y=df[dct["y"]], **addl_config
        ),
        secondary_y=True,
    )
    return new_fig


def create_chart(df, query):
    col1, col2 = st.columns([0.2, 0.8])

    # Create default chart types
    if query.has_time_dimension:
        chart_types = ["line", "area", "bar"]
    elif query.has_multiple_metrics:
        chart_types = ["line", "scatter", "bar", "area"]
    else:
        chart_types = ["bar", "pie", "histogram", "scatter"]

    selected_chart_type = col1.selectbox(
        label="Select Chart Type",
        options=chart_types,
        key="selected_chart_type",
    )

    chart_config = {}

    for field in CHART_TYPE_FIELDS[selected_chart_type]:
        selected_dimensions = [
            col for col in chart_config.values() if col in query.dimension_names
        ]
        selected_metrics = [
            col for col in chart_config.values() if col in query.metric_names
        ]

        if field == "x":
            if selected_chart_type in ["scatter", "histogram"]:
                options = query.metric_names
            elif query.has_time_dimension:
                options = query.time_dimension_names
            else:
                options = query.dimension_names
            x = col1.selectbox(
                label="X-Axis",
                options=options,
                key="chart_config_x",
            )
            chart_config["x"] = x

        if field == "y":
            if len(query.metric_names) == 1 or selected_chart_type != "line":
                widget = "selectbox"
                y_kwargs = {}
            else:
                widget = "multiselect"
                y_kwargs = {"default": query.metric_names[0]}
            y = getattr(col1, widget)(
                label="Y-Axis",
                options=[
                    m for m in query.metric_names if m not in chart_config.values()
                ],
                key="chart_config_y",
                **y_kwargs,
            )
            chart_config["y"] = y

        if (
            len(query.metric_names) > 1
            and field == "y2"
            and len([m for m in query.metric_names if m not in chart_config.values()])
            > 0
        ):
            chart_config["y2"] = {}
            expander = col1.expander("Secondary Axis Options")
            y2 = expander.selectbox(
                label="Secondary Axis",
                options=[None]
                + [m for m in query.metric_names if m not in chart_config.values()],
                key="chart_config_y2",
            )
            chart_config["y2"]["metric"] = y2
            y2_chart = expander.selectbox(
                label="Secondary Axis Chart Type",
                options=chart_types,
                index=chart_types.index(selected_chart_type),
                key="chart_config_y2_chart_type",
            )
            chart_config["y2"]["chart_type"] = y2_chart

        if field == "values":
            values = col1.selectbox(
                label="Values",
                options=query.metric_names,
                key="chart_config_values",
            )
            chart_config["values"] = values

        if field == "names":
            names = col1.selectbox(
                label="Select Dimension",
                options=query.dimension_names,
                key="chart_config_names",
            )
            chart_config["names"] = names

        if field == "color":
            color = col1.selectbox(
                label="Color",
                options=[None] + query.all_names,
                key="chart_config_color",
            )
            chart_config["color"] = color

        if _can_add_field(selected_metrics, query.metric_names) and field == "size":
            size = col1.selectbox(
                label="Size",
                options=[None]
                + _available_options(selected_metrics, query.metric_names),
                key="chart_config_size",
            )
            chart_config["size"] = size

        if (
            _can_add_field(selected_dimensions, query.dimension_names)
            and field == "facet_col"
        ):
            facet_col = col1.selectbox(
                label="Facet Column",
                options=[None]
                + _available_options(selected_dimensions, query.dimension_names),
                key="chart_config_facet_col",
            )
            chart_config["facet_col"] = facet_col

        if (
            _can_add_field(selected_dimensions, query.dimension_names)
            and field == "facet_row"
        ):
            facet_row = col1.selectbox(
                label="Facet Row",
                options=[None]
                + _available_options(selected_dimensions, query.dimension_names),
                key="chart_config_facet_row",
            )
            chart_config["facet_row"] = facet_row

        if field == "histfunc":
            histfunc = col1.selectbox(
                label="Histogram Function",
                options=["sum", "count", "avg"],
                key="chart_config_histfunc",
            )
            chart_config["histfunc"] = histfunc

        if field == "nbins":
            nbins = col1.number_input(
                label="Number of Bins",
                min_value=0,
                key="chart_config_nbins",
                value=0,
                help="If set to 0, the number of bins will be determined automatically",
            )
            chart_config["nbins"] = nbins

        # if field == 'trendline':
        #     trendline = col1.selectbox(
        #         label='Select Trendline',
        #         options=[None, 'ols'],
        #         key='chart_config_trendline',
        #     )
        #     chart_config['trendline'] = trendline

        if field == "orientation":
            orientation = col1.selectbox(
                label="Select Orientation",
                options=["Vertical", "Horizontal"],
                key="chart_config_orientation",
            )
            chart_config["orientation"] = orientation[:1].lower()
            if chart_config["orientation"] == "h":
                x = chart_config.pop("x")
                y = chart_config.pop("y")
                chart_config["x"] = y
                chart_config["y"] = x

        if field == "barmode" and len(query.dimension_names) > 1:
            barmode = col1.selectbox(
                label="Select Bar Mode",
                options=["group", "stack"],
                key="chart_config_barmode",
            )
            chart_config["barmode"] = barmode

    with col2:
        df = _sort_dataframe(df, query)
        y2_dict = chart_config.pop("y2", None)
        fig = getattr(px, selected_chart_type)(df, **chart_config)
        if y2_dict is not None and y2_dict["metric"] is not None:
            dct = {
                "y": y2_dict["metric"],
                "x": chart_config["x"],
                "chart_type": y2_dict["chart_type"],
            }
            fig = _add_secondary_yaxis(df, fig, dct)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)



GQL_MAP: Dict = {
    "metrics": {
        "kwarg": "$metrics",
        "argument": "[MetricInput!]!",
    },
    "groupBy": {
        "kwarg": "$groupBy",
        "argument": "[GroupByInput!]",
    },
    "where": {
        "kwarg": "$where",
        "argument": "[WhereInput!]",
    },
    "orderBy": {
        "kwarg": "$orderBy",
        "argument": "[OrderByInput!]",
    },
    "limit": {"kwarg": "$limit", "argument": "Int"},
}


class TimeGranularity(str, Enum):
    day = "DAY"
    week = "WEEK"
    month = "MONTH"
    quarter = "QUARTER"
    year = "YEAR"


class MetricInput(BaseModel):
    name: str = Field(
        description=(
            "Metric name defined by the user.  A metric can generally be thought of "
            "as a descriptive statistic, indicator, or figure of merit used to "
            "describe or measure something quantitatively."
        )
    )


class GroupByInput(BaseModel):
    name: str = Field(
        description=(
            "Dimension name defined by the user.  They often contain qualitative "
            "values (such as names, dates, or geographical data). You can use "
            "dimensions to categorize, segment, and reveal the details in your data. "
            "A common dimension used here will be metric_time.  This will ALWAYS have "
            "an associated grain."
        )
    )
    grain: Optional[TimeGranularity] = Field(
        default=None,
        description=(
            "The grain is the time interval represented by a single point in the data"
        ),
    )

    class Config:
        use_enum_values = True


class OrderByInput(BaseModel):
    """
    Important note:  Only one of metric or groupBy is allowed to be specified
    """

    metric: Optional[MetricInput] = None
    groupBy: Optional[GroupByInput] = None
    descending: Optional[bool] = None

    class Config:
        exclude_none = True


class WhereInput(BaseModel):
    sql: str


class Query(BaseModel):
    """Data model for a graphQL query."""

    metrics: List[MetricInput]
    groupBy: Optional[List[GroupByInput]] = None
    where: Optional[List[WhereInput]] = None
    orderBy: Optional[List[OrderByInput]] = None
    limit: Optional[int] = None

    @property
    def all_names(self):
        return self.metric_names + self.dimension_names

    @property
    def metric_names(self):
        return [m.name for m in self.metrics]

    @property
    def dimension_names(self):
        return [
            f"{g.name}__{g.grain.lower()}" if g.grain is not None else g.name
            for g in self.groupBy or []
        ]

    @property
    def time_dimension_names(self):
        return [
            f"{g.name}__{g.grain.lower()}"
            for g in self.groupBy or []
            if g.grain is not None
        ]

    @property
    def has_time_dimension(self):
        if self.groupBy is not None:
            return any([g.grain is not None for g in self.groupBy])

        return False

    @property
    def has_multiple_metrics(self):
        return len(self.metrics) > 1

    @property
    def used_inputs(self) -> List[str]:
        inputs = []
        for key in GQL_MAP.keys():
            prop = getattr(self, key)
            if prop is not None:
                try:
                    if len(prop) > 0:
                        inputs.append(key)
                except TypeError:
                    inputs.append(key)

        return inputs

    @property
    def _jdbc_text(self) -> str:
        text = f"metrics={[m.name for m in self.metrics]}"
        if self.groupBy is not None:
            group_by = [
                f"{g.name}__{g.grain.lower()}" if g.grain is not None else g.name
                for g in self.groupBy
            ]
            text += f",\n        group_by={group_by}"
        if self.where is not None:
            where = " AND ".join([w.sql for w in self.where])
            text += f',\n        where="{where}"'
        if self.orderBy is not None:
            names = []
            for order in self.orderBy:
                obj = order.metric if order.metric else order.groupBy
                if hasattr(obj, "grain") and obj.grain is not None:
                    name = f"{obj.name}__{obj.grain.lower()}"
                else:
                    name = obj.name
                if order.descending:
                    name = f"-{name}"
                names.append(name)
            text += f",\n        order_by={names}"
        if self.limit is not None:
            text += f",\n        limit={self.limit}"
        return text

    @property
    def jdbc_query(self):
        sql = f"""
select *
from {{{{
    semantic_layer.query(
        {self._jdbc_text}
    )
}}}}
        """
        return sql

    @property
    def gql(self) -> str:
        query = GRAPHQL_QUERIES["create_query"]
        kwargs = {"environmentId": "$environmentId"}
        arguments = {"environmentId": "BigInt!"}
        for input in self.used_inputs:
            kwargs[input] = GQL_MAP[input]["kwarg"]
            arguments[input] = GQL_MAP[input]["argument"]
        return query.format(
            **{
                "arguments": ", ".join(f"${k}: {v}" for k, v in arguments.items()),
                "kwargs": ", ".join([f"{k}: {v}" for k, v in kwargs.items()]),
            }
        )

    @property
    def variables(self) -> Dict[str, List[Any]]:
        variables = {}
        for input in self.used_inputs:
            data = getattr(self, input)
            if isinstance(data, list):
                variables[input] = [m.dict(exclude_none=True) for m in data]
            else:
                try:
                    variables[input] = getattr(self, input).dict(
                        exclude_none=True
                    )
                except AttributeError:
                    variables[input] = getattr(self, input)
        return variables



def to_arrow_table(byte_string: str, to_pandas: bool = True) -> pa.Table:
    with pa.ipc.open_stream(base64.b64decode(byte_string)) as reader:
        arrow_table = pa.Table.from_batches(reader, reader.schema)

    if to_pandas:
        return arrow_table.to_pandas()

    return arrow_table



GRAPHQL_QUERIES = {
    "metrics": """
query GetMetrics($environmentId: BigInt!) {
  metrics(environmentId: $environmentId) {
    description
    name
    queryableGranularities
    type
    dimensions {
      description
      name
      type
    }
  }
}
    """,
    "dimensions": """
query GetDimensions($environmentId: BigInt!, $metrics: [MetricInput!]!) {
  dimensions(environmentId: $environmentId, metrics: $metrics) {
    description
    expr
    isPartition
    metadata {
      fileSlice {
        content
        endLineNumber
        filename
        startLineNumber
      }
      repoFilePath
    }
    name
    qualifiedName
    type
    typeParams {
      timeGranularity
      validityParams {
        isEnd
        isStart
      }
    }
  }
}
    """,
    "dimension_values": """
mutation GetDimensionValues($environmentId: BigInt!, $groupBy: [GroupByInput!]!, $metrics: [MetricInput!]!) {
  createDimensionValuesQuery(
    environmentId: $environmentId
    groupBy: $groupBy
    metrics: $metrics
  ) {
    queryId
  }
}
    """,
    "metric_for_dimensions": """
query GetMetricsForDimensions($environmentId: BigInt!, $dimensions: [GroupByInput!]) {
  metricsForDimensions(environmentId: $environmentId, dimensions: $dimensions) {
    description
    name
    queryableGranularities
    type
  }
}
    """,
    "create_query": """mutation CreateQuery({arguments}) {{createQuery({kwargs}) {{queryId}}}}""",
    "get_results": """
query GetResults($environmentId: BigInt!, $queryId: String!) {
  query(environmentId: $environmentId, queryId: $queryId) {
    arrowResult
    error
    queryId
    sql
    status
  }
}
    """,
    "queryable_granularities": """
query GetQueryableGranularities($environmentId: BigInt!, $metrics:[MetricInput!]!) {
  queryableGranularities(environmentId: $environmentId, metrics: $metrics)
}
    """,
    "metrics_for_dimensions": """
query GetMetricsForDimensions($environmentId: BigInt!, $dimensions:[GroupByInput!]!) {
  metricsForDimensions(environmentId: $environmentId, dimensions: $dimensions) {
    description
    name
    queryableGranularities
    type
    filter {
      whereSqlTemplate
    }
  }
}
    """,
}

JDBC_QUERIES = {
    "metrics": """
select *
from {{
    semantic_layer.metrics()
}}
""",
    "dimensions": """
select *
from {{{{
    semantic_layer.dimensions(
        metrics={metrics}
    )
}}}}
""",
    "dimension_values": """
select *
from {{{{
    semantic_layer.dimension_values(
        metrics={metrics},
        group_by='{dimension}'
    )
}}}}
""",
    "queryable_granularities": """
select *
from {{{{
    semantic_layer.queryable_granularities(
        metrics={metrics}
    )
}}}}
""",
    "metrics_for_dimensions": """
select *
from {{{{
    semantic_layer.metrics_for_dimensions(
        group_by={dimensions}
    )
}}}}
""",
}


EXAMPLES = [
    {
        "metrics": "total_revenue, total_expense, total_profit, monthly_customers",
        "dimensions": "metric_time, customer__customer_region, customer__customer_country",
        "question": "What is total revenue by month for customers in the United States?",
        "result": Query.parse_obj(
            {
                "metrics": [{"name": "total_revenue"}],
                "groupBy": [{"name": "metric_time", "grain": "MONTH"}],
                "where": [
                    {
                        "sql": "{{ Dimension('customer__customer_country') }} ilike 'United States'"
                    }
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_cost, total_sales, avg_sales_per_cust, avg_cost_per_cust",
        "dimensions": "metric_time, order_date, customer__country, customer__region, customer__city, customer__is_active",
        "question": "What is the average revenue and cost per customer by quarter for customers in Denver ordered by average revenue descending?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "avg_sales_per_cust"},
                    {"name": "avg_cost_per_cust"},
                ],
                "groupBy": [{"name": "metric_time", "grain": "QUARTER"}],
                "where": [{"sql": "{{ Dimension('customer__city') }} ilike 'Denver'"}],
                "orderBy": [
                    {"metric": {"name": "avg_sales_per_cust"}, "descending": True}
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "sales, expenses, profit, sales_ttm, expenses_ttm, profit_ttm, sales_yoy, expenses_yoy, profit_yoy",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, product__product_subcategory, product__product_name",
        "question": "What is the total sales, expenses, and profit for the last 3 months by product category?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "sales"},
                    {"name": "expenses"},
                    {"name": "profit"},
                ],
                "groupBy": [
                    {"name": "metric_time", "grain": "MONTH"},
                    {"name": "product__product_category"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'MONTH') }} >= dateadd('month', -3, current_date)"
                    }
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "revenue, costs, profit",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category",
        "question": "What is the total revenue by department this year?",
        "result": Query.parse_obj(
            {
                "metrics": [{"name": "revenue"}],
                "groupBy": [{"name": "department"}],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} >= date_trunc('year', current_date)"
                    }
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "What is total revenue by region and balance segment year-to-date over time ordered by time and region?",
        "result": Query.parse_obj(
            {
                "metrics": [{"name": "total_revenue"}],
                "groupBy": [
                    {"name": "customer__region"},
                    {"name": "customer__balance_segment"},
                    {"name": "metric_time", "grain": "YEAR"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} >= date_trunc('year', current_date)"
                    }
                ],
                "orderBy": [
                    {"groupBy": {"name": "metric_time", "grain": "YEAR"}},
                    {"groupBy": {"name": "region"}},
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "Which region has the most revenue in the past 365 days?",
        "result": Query.parse_obj(
            {
                "metrics": [{"name": "total_revenue"}],
                "groupBy": [
                    {"name": "customer__region"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} >= dateadd('d', -365, current_date)"
                    }
                ],
                "orderBy": [
                    {"metric": {"name": "total_revenue"}, "descending": True},
                ],
                "limit": 1,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "Which country has the most revenue in the past 365 days?",
        "result": Query.parse_obj(
            {
                "metrics": [{"name": "total_revenue"}],
                "groupBy": [
                    {"name": "customer__nation"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} >= dateadd('d', -365, current_date)"
                    }
                ],
                "orderBy": [
                    {"metric": {"name": "total_revenue"}, "descending": True},
                ],
                "limit": 1,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "Can you give me revenue, expense, and profit in 2023?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_revenue"},
                    {"name": "total_expense"},
                    {"name": "total_profit"},
                ],
                "where": [
                    {"sql": "year({{ TimeDimension('metric_time', 'DAY') }}) = 2023"}
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "Can you give me the top 10 sales people by revenue in September 2023?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_revenue"},
                ],
                "groupBy": [
                    {"name": "salesperson"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} between '2023-09-01' and '2023-09-30'"
                    }
                ],
                "orderBy": [
                    {"metric": {"name": "total_revenue"}, "descending": True},
                ],
                "limit": 10,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__nation, customer__balance_segment",
        "question": "What are the top 5 nations by total profit in 2023?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_profit"},
                ],
                "groupBy": [
                    {"name": "customer__nation"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} between '2023-01-01' and '2023-12-31'"
                    }
                ],
                "orderBy": [
                    {"metric": {"name": "total_profit"}, "descending": True},
                ],
                "limit": 5,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "Can you give me revenue by salesperson in the first quarter of 2023 where product category is either cars or motorcycles?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_revenue"},
                ],
                "groupBy": [
                    {"name": "salesperson"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} between '2023-01-01' and '2023-03-31'"
                    },
                    {
                        "sql": "{{ Dimension('product__product_category') }} ilike any ('cars', 'motorcycles')"
                    },
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "campaigns, impressions, clicks, conversions, roi, run_rate, arr, mrr, ltv, cac",
        "dimensions": "customer__geography__country, customer__market_segment, customer__industry, customer__sector, metric_time",
        "question": "What is the annual recurring revenue and customer acquisition cost by month and country in the United States?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "arr"},
                    {"name": "cac"},
                ],
                "groupBy": [
                    {"name": "customer__geography__country"},
                    {"name": "metric_time", "grain": "MONTH"},
                ],
                "where": [
                    {
                        "sql": "{{ Dimension('geography__country, entity_path=['customer']) }} ilike 'United States'"
                    },
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "campaigns, impressions, clicks, conversions, roi, run_rate, arr, mrr, ltv, cac",
        "dimensions": "customer__geography__country, customer__market_segment, customer__industry, customer__sector, metric_time, close_date",
        "question": "What are the 5 worst performing sectors by arr last month in the enterprise segment?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "arr"},
                ],
                "groupBy": [
                    {"name": "customer__sector"},
                ],
                "where": [
                    {
                        "sql": "{{ Dimension('customer__market_segment') }} ilike 'enterprise'"
                    },
                    {
                        "sql": "date_trunc('month', {{ TimeDimension('metric_time', 'DAY') }}) = date_trunc('month', dateadd('month', -1, current_date))"
                    },
                ],
                "orderBy": [
                    {"metric": {"name": "arr"}},
                ],
                "limit": 5,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "What is total profit in 2022 by quarter?",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_revenue"},
                ],
                "groupBy": [
                    {"name": "metric_time", "grain": "QUARTER"},
                ],
                "where": [
                    {
                        "sql": "{{ TimeDimension('metric_time', 'DAY') }} between '2022-01-01' and '2022-12-31'"
                    },
                ],
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer__region, customer__balance_segment",
        "question": "What day had the highest revenue",
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_revenue"},
                ],
                "groupBy": [
                    {"name": "metric_time", "grain": "DAY"},
                ],
                "orderBy": [
                    {
                        "metric": {
                            "name": "total_revenue"
                        },
                        "descending": True,
                    }
                ],
                "limit": 1
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
    {
        "metrics": "total_revenue, total_expense, total_profit, total_customers, monthly_customers, weekly_customers, daily_customers",
        "dimensions": "department, salesperson, cost_center, metric_time, product__product_category, customer_order__customer__customer_market_segment, customer_order__clerk_on_order",
        "question": "Who is the top clerk by total profit in 2023 in the automobile market segment?",
        "incorrect_result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_profit"},
                ],
                "groupBy": [
                    {"name": "customer_order__clerk_on_order"},
                ],
                "where": [
                    {"sql": "year({{ TimeDimension('metric_time', 'DAY') }}) = 2023"},
                    {
                        "sql": "{{ Dimension('customer_order__customer__customer_market_segment') }} ilike 'automobile'"
                    },
                ],
                "orderBy": [
                    {"metric": {"name": "total_profit"}, "descending": True},
                ],
                "limit": 1,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
        "result": Query.parse_obj(
            {
                "metrics": [
                    {"name": "total_profit"},
                ],
                "groupBy": [
                    {"name": "customer_order__clerk_on_order"},
                ],
                "where": [
                    {"sql": "year({{ TimeDimension('metric_time', 'DAY') }}) = 2023"},
                    {
                        "sql": "{{ Dimension('customer__customer_market_segment') }} ilike 'automobile'"
                    },
                ],
                "orderBy": [
                    {"metric": {"name": "total_profit"}, "descending": True},
                ],
                "limit": 1,
            }
        )
        .json()
        .replace("{", "{{")
        .replace("}", "}}"),
    },
]


MODELS = {
    "mistral-large": "Top-tier model that scores well on a variety of metrics. It features a context window of 32,000 tokens (about 24,000 words), allowing it to engage in complex reasoning across extended conversations. It is also the most compute-intensive of the models offered by Snowflake Cortex and thus the most costly to run.",
    "mistral-8x7b": "This model provides low latency and high quality results, while also supporting a context length of 32,000 tokens. It is ideal for many enterprise production use cases.",
    "llama2-70b-chat": "This is well-suited to complex, large-scale tasks that require a moderate amount of reasoning, like extracting data or helping you to write job descriptions.",
    "mistral-7b": "Ideal for your simplest summarization and classification tasks that require a smaller degree of customization. Its 32,000 token limit gives it the ability to process multiple pages of text.",
    "gemma-7b": "suitable for simple code and text completion tasks. It has a context window of 8,000 tokens but is surprisingly capable within that limit, and quite cost-effective.",
}


def set_question():
    previous_question = st.session_state.get("_question", None)
    st.session_state._question = st.session_state.question
    st.session_state.refresh = not previous_question == st.session_state._question


def is_none(item):
    return (
        item is None or
        item == 0 or
        item == ""
    )


def prepare_app():

    host = st.session_state.get("host")
    environment_id = st.session_state.get("environment_id")
    token = st.session_state.get("token")
    if any([is_none(i) for i in [host, environment_id]]):
        st.error("Host and environment ID are required inputs")
        st.stop()

    sql = f"select retrieve_sl_metadata('{host}', {environment_id}, '{token}')"
    with st.spinner("Retrieving metadata..."):
        response = session.sql(sql).collect()
        data = json.loads(response[0][0])
        try:
            metrics = data["data"]["metrics"]
        except TypeError:

            # `data` is None and there may be an error
            try:
                error = data["errors"][0]["message"]
                st.error(error)
            except (KeyError, TypeError):
                st.warning(
                    "No metrics returned.  Ensure your project has metrics defined "
                    "and a production job has been run successfully."
                )
        else:
            st.session_state.metric_dict = {m["name"]: m for m in metrics}
            st.session_state.dimension_dict = {
                dim["name"]: dim for metric in metrics for dim in metric["dimensions"]
            }
            st.session_state.metrics = ", ".join(list(st.session_state.metric_dict.keys()))
            st.session_state.dimensions = ", ".join(list(st.session_state.dimension_dict.keys()))
            for metric in st.session_state.metric_dict:
                st.session_state.metric_dict[metric]["dimensions"] = [
                    d["name"]
                    for d in st.session_state.metric_dict[metric]["dimensions"]
                ]
            if not st.session_state.metric_dict:
                # Query worked, but nothing returned
                st.warning(
                    "No Metrics returned!  Ensure your project has metrics defined "
                    "and a production job has been run successfully."
                )
            else:
                st.success("Success!  Start asking questions!")
                st.experimental_rerun()


parser = PydanticOutputParser(pydantic_object=Query)

prompt_example = PromptTemplate(
    template=EXAMPLE_PROMPT,
    input_variables=["metrics", "dimensions", "question", "result"],
)

prompt = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=prompt_example,
    prefix="""Given a question involving a user's data, transform it into a structured query object.
                It's important to remember that in the 'orderBy' field, only one of 'metric' or 'groupBy' should be set, not both.
                Here are some examples showing how to correctly and incorrectly structure a query based on a user's question.
                {format_instructions}
                Additionally, when adding items to the `where` field and the identifier contains multiple dunder (__) characters,
                you'll need to change how you specify the dimension.  An example of this is `customer_order__customer__customer_market_segment`.
                This needs to be represented as Dimension('customer__customer_market_segment').
                Also, any questions that begin with "Which..." should always include a limit of 1.
            """,
    suffix="Metrics: {metrics}\nDimensions: {dimensions}\nQuestion: {question}\nResult:\n",
    input_variables=["metrics", "dimensions", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Activate session
session = get_active_session()

st.markdown("# Explore the dbt Semantic Layer")

st.markdown(
    """
    Use this app to ask questions about your data based on the semantics defined in your dbt project.  This application uses Snowflake's new [Large Language
    Model Functions]((https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#label-cortex-llm-complete)) (Snowflake Cortex), specifically the `Complete` function,
    to turn your question into a semantic layer request.

    ### Want to learn more?
    - Get started with the [dbt Semantic Layer](https://docs.getdbt.com/docs/use-dbt-semantic-layer/quickstart-sl)
    - Understand how to [build your metrics](https://docs.getdbt.com/docs/build/build-metrics-intro)
    - View the [Semantic Layer API](https://docs.getdbt.com/docs/dbt-cloud-apis/sl-api-overview)
    ---
    
    To get started, input your semantic layer configuration in the box below.  After receiving confirmation, start asking questions of your data!
    """
)

# st.selectbox(
#     label="Select Model",
#     options=list(MODELS.keys()),
#     index=1,
#     key="llm_model",
# )
# st.caption(MODELS[st.session_state.llm_model])

with st.expander("Input Semantic Layer Config", expanded="metric_dict" not in st.session_state):
    st.text_input(
        label="Host",
        value=st.session_state.get("host", "semantic-layer.cloud.getdbt.com"),
        key="host",
        help="This is where your instance of dbt Cloud is located - see [here](https://docs.getdbt.com/docs/dbt-cloud-apis/sl-graphql#dbt-semantic-layer-graphql-api) for more info",
    )
    
    st.number_input(
        label="Environment ID",
        key="environment_id",
        value=st.session_state.get("environment_id", 218762),
        help="The ID of your environment designated as Production",
    )
    
    st.text_input(
        label="Token",
        key="token",
        type="password",
        value=st.session_state.get("token", ""),
        help="Service token created to query the semantic layer.  This is optional if your admin has already created the appropriate secret within your environment.",
    )

    submitted = st.button("Submit")
    if submitted:
        prepare_app()


st.markdown("## Ask Some Questions!")

question = st.text_input(
    label="",
    placeholder="e.g. What is total revenue?",
    key="question",
    on_change=set_question,
)


if question and st.session_state.get('refresh', False):

    prompt_text = prompt.format(
        metrics=st.session_state.metrics,
        dimensions=st.session_state.dimensions,
        question=question
    )

    st.toast("Turning question into request...", icon="🤖")
    sql = f"select snowflake.cortex.complete('mixtral-8x7b', $${prompt_text}$$::variant) as response"
    response = session.sql(sql).collect()
    response_str = response[0][0]
    response_str = response_str.replace("\\_", "_")
    try:
        query = parser.parse(response_str)
    except OutputParserException as e:
        st.error(e)
        st.stop()

    st.toast("Querying the semantic layer...", icon="💻")
    payload = {"query": query.gql, "variables": query.variables}
    sql = f"select submit_sl_request('{st.session_state.host}', {st.session_state.environment_id}, $${json.dumps(payload)}$$, '{st.session_state.token}')"
    response = session.sql(sql).collect()
    data = json.loads(response[0][0])
    try:
        df = to_arrow_table(data["arrowResult"])
    except TypeError:
        st.error(data.get('error', 'Unkown Error'))
        st.stop()
    df.columns = [col.lower() for col in df.columns]
    st.session_state.df = df
    st.session_state.data = data
    st.session_state.query = query
    st.toast("Query successful", icon="✅")


st.session_state.refresh = False
if 'data' in st.session_state and 'df' in st.session_state:
    tab1, tab2, tab3 = st.tabs(["Chart", "Data", "SQL"])
    data = st.session_state.data
    df = st.session_state.df
    with tab1:
        create_chart(df, st.session_state.query)
    with tab2:
        st.dataframe(df, use_container_width=True)
    with tab3:
        st.code(data["sql"], language="sql")

if "metric_dict" not in st.session_state and all([i in st.session_state for i in ["host", "environment_id"]]):
    prepare_app()    
    
