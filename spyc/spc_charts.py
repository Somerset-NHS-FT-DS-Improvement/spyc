import plotly.graph_objects as go
import plotly.io as pio


def filter_out_of_control(data):

    df_rules = data.filter(regex="^Rule ", axis=1)
    out_of_control = data[df_rules.sum(axis=1) > 0]
    return out_of_control


def add_control_lines(fig, data):

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["CL"],
            mode="lines",
            line=dict(color="#66a182", dash="dash"),
            name="Center Line",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["UCL"],
            mode="lines",
            line=dict(color="#d1495b", dash="dot"),
            name="Upper Control Limit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["LCL"],
            mode="lines",
            line=dict(color="#d1495b", dash="dot"),
            name="Lower Control Limit",
        )
    )


def add_process_lines(fig, data, y_col, line_name):
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[y_col],
            mode="lines",
            line=dict(color="#00789c", width=2),
            name=line_name,
        )
    )


def add_out_of_control_points(fig, signals_data, y_col):

    # Determine rule columns
    rule_columns = [col for col in signals_data.columns if col.startswith("Rule")]

    fig.add_trace(
        go.Scatter(
            x=signals_data.index,
            y=signals_data[y_col],
            mode="markers",
            marker=dict(
                size=12,
                color="rgba(209, 73, 91, 0.25)",
                line=dict(color="rgba(209, 73, 91, 0.6)", width=2),
            ),
            name="Out-of-Control",
            hoverinfo="text",
            hovertext=[
                "Out of Control<br>"
                f"Violations: {', '.join([rule for rule in rule_columns if row[rule] == 1])}"
                for _, row in signals_data.iterrows()
            ],
        )
    )


def plotly_chart(data, figure_title="SPC chart"):

    # Initialise base figure
    fig = go.Figure()

    # Add SPC elements
    out_of_control = filter_out_of_control(data)
    add_out_of_control_points(fig, out_of_control, "process")
    add_process_lines(fig, data, "process", line_name="Process")
    add_control_lines(fig, data)

    # Add process change vertical lines
    process_change_dates = data["period_name"].dropna().index.to_list()

    if process_change_dates:
        for date in process_change_dates:
            # Plotly used 'new text' label if None provided in annotation_text
            annotation_text = data.loc[date]["period_name"]

            fig.add_vline(
                x=date.timestamp() * 1000,
                line_dash="dash",
                line_color="#27374D",
                annotation_text=annotation_text,
                annotation_position="top",
                annotation_font=dict(size=14, color="black"),
                annotation_bgcolor="white",
            )

    # Update Layout
    fig.update_layout(
        title=figure_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="#dde6ed",
            range=[data.index.min(), data.index.max()],
        ),
        yaxis=dict(
            title="Measure",
            showgrid=True,
            gridcolor="#dde6ed",
        ),
    )

    return fig


def combine_figures(
    figures,
    fig_names,
    process_change_dates,
    process_change_dict,
    save_to_html=False,
    file_name=None,
):
    # Create a new figure for combining
    combined_fig = go.Figure()

    # Initialize a list to store the number of traces for each figure
    num_traces = []

    # Add traces from all figures
    for fig in figures:
        combined_fig.add_traces(fig.data)
        num_traces.append(len(fig.data))

    # Create toggle buttons for each figure
    buttons = []
    start_idx = 0
    for i, fig in enumerate(figures):
        num_fig_traces = num_traces[i]
        end_idx = start_idx + num_fig_traces

        # Set visibility for each figure's traces: visible for current figure, hidden for others
        visibility = [False] * len(combined_fig.data)
        visibility[start_idx:end_idx] = [True] * num_fig_traces

        # Add a button for toggling this figure
        buttons.append(
            dict(
                label=f"{fig_names[i]}",
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            fig.layout.title.text
                            if fig.layout.title
                            else f"Figure {i+1}"
                        )
                    },
                ],
            )
        )
        start_idx = end_idx

    # Update layout to preserve titles, axis ranges, and legends
    combined_fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=buttons,
                direction="right",
                x=0.5,
                xanchor="center",
                y=1.2,
                yanchor="top",
            )
        ],
        xaxis=figures[
            0
        ].layout.xaxis,  # Set the xaxis from the first figure (to preserve range, etc.)
        yaxis=figures[0].layout.yaxis,  # Set the yaxis from the first figure
        legend=figures[0].layout.legend,  # Set the legend from the first figure
        hovermode="x unified",  # Use unified hover mode for combined chart
    )

    # process_change_dates = data.index[data['period'].diff().fillna(0) != 0].tolist()
    if process_change_dates:
        for date in process_change_dates:
            # Plotly used 'new text' label if None provided in annotation_text
            annotation_text = (
                ""
                if not process_change_dict[date]["process_names"]
                else process_change_dict[date]["process_names"]
            )

            combined_fig.add_vline(
                x=date.timestamp() * 1000,
                line_dash="dash",
                line_color="#27374D",
                annotation_text=annotation_text,
                annotation_position="top",
                annotation_font=dict(size=14, color="black"),
                annotation_bgcolor="white",
            )

    # Carry over hoverinfo and hoverlabel from individual traces
    for i, fig in enumerate(figures):
        for trace_idx, trace in enumerate(fig.data):
            # Add hoverinfo and hoverlabel to ensure tooltips are displayed correctly
            combined_fig.data[sum(num_traces[:i]) + trace_idx].update(
                hoverinfo=trace.hoverinfo, hoverlabel=trace.hoverlabel
            )

    # Ensure initial visibility (only the first figure's traces are visible)
    for i in range(len(combined_fig.data)):
        combined_fig.data[i].visible = i < num_traces[0]

    # Optionally save the combined figure as an HTML file
    if save_to_html:
        pio.write_html(combined_fig, file=f"{file_name}.html", full_html=True)

    return combined_fig
