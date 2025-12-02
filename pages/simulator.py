import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from models import Measurement, MeasurementResults

st.set_page_config(page_title="Actuator Simulator", layout="wide")

st.title("Actuator Simulator")

with st.sidebar:
    st.header("Measurement Setup")
    with st.form("measurement_form"):
        collimator_size = st.number_input(
            "Collimator Size (mm)", min_value=1, max_value=200, value=16, step=1
        )
        exposure_time_min = st.number_input(
            "Exposure Time (min)", min_value=0.1, max_value=60.0, value=1.0, step=0.1
        )
        actuator_speed = st.number_input(
            "Actuator Speed (mm/s)", min_value=0.1, max_value=50.0, value=5.0, step=0.1
        )
        dose_rate = st.number_input(
            "Dose Rate (Gy/min)", min_value=0.001, max_value=20.0, value=3.405, step=0.001
        )

        submitted = st.form_submit_button("Save Inputs")
        if submitted:
            # st.session_state.measurement = Measurement(
            #     colimiator_size=collimator_size,
            #     exposure_time_s=int(exposure_time_min * 60),
            #     actuator_speed_mm_s=actuator_speed,
            #     dose_rate_gy_s=dose_rate / 60.0,
            # ).dict()
            st.sidebar.success("Measurement saved.")

st.divider()
max_travel_mm = 50.0
x_axis = np.linspace(0, max_travel_mm, 200)

if "actuator_position" not in st.session_state:
    st.session_state.actuator_position = 0.0

if "gradient_history" not in st.session_state:
    st.session_state.gradient_history = []

if "half_history_left" not in st.session_state:
    st.session_state.half_history_left = []

if "half_history_right" not in st.session_state:
    st.session_state.half_history_right = []

if "gradient_results" not in st.session_state:
    st.session_state.gradient_results = None

if "half_results" not in st.session_state:
    st.session_state.half_results = None

if "active_animation" not in st.session_state:
    st.session_state.active_animation = None

layout_col_pos, layout_col_plot = st.columns([1, 2], gap="large")
actuator_plot_placeholder = layout_col_pos.empty()
dose_plot_placeholder = layout_col_plot.empty()
current_position = st.session_state.actuator_position

measurement_data = st.session_state.get("measurement")
beam_center_mm = max_travel_mm / 2.0
fwhm_mm = (
    measurement_data["colimiator_size"]
    if measurement_data
    else collimator_size
)
dmax_gy_s = (
    measurement_data["dose_rate_gy_s"]
    if measurement_data
    else dose_rate / 60.0
)

def gaussian(position: float) -> float:
    normalized_fwhm = max(fwhm_mm, 1e-6)
    return float(
        dmax_gy_s * np.exp(-2.773 * ((position - beam_center_mm) / normalized_fwhm) ** 2)
    )

def gaussian_grad(position: float) -> float:
    normalized_fwhm = max(fwhm_mm, 1e-6)
    denom = normalized_fwhm**2
    return float(-2 * 2.773 * (position - beam_center_mm) / denom)

dose_profile = np.array([gaussian(x) for x in x_axis])
current_dose = gaussian(current_position)

half_value = 0.5 * dmax_gy_s
center_idx = int(np.argmax(dose_profile))
left_half_point = None
right_half_point = None

if center_idx > 0:
    left_vals = dose_profile[: center_idx + 1]
    left_pos = x_axis[: center_idx + 1]
    if left_vals.min() <= half_value <= left_vals.max():
        left_half_point = float(np.interp(half_value, left_vals, left_pos))

right_vals = dose_profile[center_idx:]
right_pos = x_axis[center_idx:]
if right_vals.size > 1:
    if right_vals.min() <= half_value <= right_vals.max():
        right_half_point = float(
            np.interp(half_value, right_vals[::-1], right_pos[::-1])
        )

st.session_state["dose_profile"] = MeasurementResults(
    x_axis_mm=list(x_axis),
    y_axis_gy=list(dose_profile),
    center_point_mm=beam_center_mm,
    left_half_point=left_half_point,
    right_half_point=right_half_point,
    fwhm_mm=float(fwhm_mm),
).dict()

learning_rate = 5
max_steps = 200
grad_tolerance = 1e-3
frame_delay = 0.05
binary_tolerance = 1e-2
position_tolerance = 0.05
binary_frame_delay = 0.1
scan_step = 0.5


def render_actuator_plot(position: float) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.zeros_like(x_axis),
            y=x_axis,
            mode="lines",
            name="Travel",
        )
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=0.5,
        y0=position,
        y1=position,
        line=dict(color="red", width=4),
    )
    fig.update_layout(
        yaxis_title="Position (mm)",
        title="Actuator Position Monitor",
        title_x=0.5,
        title_y=0.94,
        xaxis_visible=False,
        xaxis_showticklabels=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
    )
    actuator_plot_placeholder.plotly_chart(fig, use_container_width=True)


def render_dose_plot(
    gradient_history=None,
    left_history=None,
    right_history=None,
    half_results=None,
    actuator_position=None,
) -> None:
    gradient_history = (
        gradient_history
        if gradient_history is not None
        else st.session_state.get("gradient_history", [])
    )
    left_history = (
        left_history
        if left_history is not None
        else st.session_state.get("half_history_left", [])
    )
    right_history = (
        right_history
        if right_history is not None
        else st.session_state.get("half_history_right", [])
    )
    half_results = (
        half_results
        if half_results is not None
        else st.session_state.get("half_results")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=dose_profile,
            mode="lines",
            name="D(x)",
            line=dict(color="gray", dash="dash", width=2),
            opacity=0.7,
        )
    )
    position_marker = (
        st.session_state.actuator_position
        if actuator_position is None
        else actuator_position
    )
    fig.add_vline(
        x=position_marker,
        line=dict(color="red", dash="dash"),
        annotation_text="Actuator",
        annotation_position="top",
    )
    fig.add_hline(
        y=half_value,
        line=dict(color="gray", dash="dot"),
        annotation_text="Half dose",
        annotation_position="bottom right",
    )
    if gradient_history:
        fig.add_trace(
            go.Scatter(
                x=[pt[0] for pt in gradient_history],
                y=[pt[1] for pt in gradient_history],
                mode="markers+lines",
                name="Gradient Path",
                marker=dict(color="orange", size=8),
            )
        )
    if half_results:
        left = half_results.get("left")
        right = half_results.get("right")
        if left is not None:
            fig.add_vline(
                x=left,
                line=dict(color="green", dash="dot"),
                annotation_text="Left 50%",
                annotation_position="bottom",
            )
        if right is not None:
            fig.add_vline(
                x=right,
                line=dict(color="purple", dash="dot"),
                annotation_text="Right 50%",
                annotation_position="bottom",
            )
        if left is not None and right is not None:
            fig.add_trace(
                go.Scatter(
                    x=[left, right],
                    y=[half_value, half_value],
                    mode="markers",
                    name="Half-dose points",
                    marker=dict(color="gold", size=10, symbol="diamond"),
                )
            )
    fig.update_layout(
        xaxis_title="Position (mm)",
        yaxis_title="Dose Rate (Gy/s)",
        title="Dose Profile & Searches",
        title_x=0.5,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    dose_plot_placeholder.plotly_chart(fig, use_container_width=True)


def animate_move(
    target: float,
    gradient_history=None,
    left_history=None,
    right_history=None,
) -> None:
    target = float(np.clip(target, 0.0, max_travel_mm))
    current = st.session_state.actuator_position
    if abs(target - current) < 1e-6:
        return
    steps = max(int(abs(target - current) / 0.5), 1) + 1
    path = np.linspace(current, target, steps)
    for pos in path[1:]:
        st.session_state.actuator_position = float(np.clip(pos, 0.0, max_travel_mm))
        render_actuator_plot(st.session_state.actuator_position)
        render_dose_plot(
            gradient_history=gradient_history,
            left_history=left_history,
            right_history=right_history,
            actuator_position=st.session_state.actuator_position,
        )
        time.sleep(binary_frame_delay)


def run_gradient_animation():
    if st.session_state.active_animation:
        return
    st.session_state.active_animation = "gradient"
    history = []
    position = st.session_state.actuator_position
    for step in range(max_steps):
        dose = gaussian(position)
        history.append((position, dose))
        render_actuator_plot(position)
        render_dose_plot(gradient_history=history, actuator_position=position)
        grad = gaussian_grad(position)
        if abs(grad) < grad_tolerance:
            break
        position = float(
            np.clip(position + learning_rate * grad, 0.0, max_travel_mm)
        )
        time.sleep(frame_delay)
    st.session_state.actuator_position = position
    st.session_state.gradient_history = history
    st.session_state.gradient_results = {
        "center_mm": position,
        "max_dose": gaussian(position),
        "half_dose": half_value,
        "steps": history,
    }
    st.session_state.active_animation = None
    render_actuator_plot(position)
    render_dose_plot()


def run_half_search_animation():
    if st.session_state.active_animation or not st.session_state.gradient_results:
        return
    st.session_state.active_animation = "half"
    center_ref = st.session_state.gradient_results["center_mm"]
    animate_move(
        center_ref,
        gradient_history=st.session_state.gradient_history,
        left_history=st.session_state.half_history_left,
        right_history=st.session_state.half_history_right,
    )

    left_history: list[tuple[float, float]] = []
    right_history: list[tuple[float, float]] = []

    def scan_direction(direction: int, history: list[tuple[float, float]]):
        prev_pos = st.session_state.actuator_position
        prev_dose = gaussian(prev_pos)
        while True:
            candidate = prev_pos + direction * scan_step
            if candidate < 0.0 or candidate > max_travel_mm:
                candidate = np.clip(candidate, 0.0, max_travel_mm)
            animate_move(
                candidate,
                gradient_history=st.session_state.gradient_history,
                left_history=history if direction < 0 else (right_history if direction > 0 else None),
                right_history=right_history if direction > 0 else (history if direction > 0 else None),
            )
            dose = gaussian(candidate)
            history.append((candidate, dose))
            if dose <= half_value or candidate in (0.0, max_travel_mm):
                if abs(dose - half_value) < 1e-6 or dose == prev_dose:
                    half_pos = candidate
                else:
                    frac = (half_value - prev_dose) / (dose - prev_dose)
                    half_pos = prev_pos + frac * (candidate - prev_pos)
                    animate_move(
                        half_pos,
                        gradient_history=st.session_state.gradient_history,
                        left_history=left_history,
                        right_history=right_history,
                    )
                    history.append((half_pos, half_value))
                return half_pos
            prev_pos = candidate
            prev_dose = dose

    left_point = scan_direction(-1, left_history)
    st.session_state.half_results = {"left": left_point, "right": None, "target": half_value}
    render_dose_plot(half_results=st.session_state.half_results)
    right_point = scan_direction(1, right_history)

    st.session_state.actuator_position = center_ref
    st.session_state.half_history_left = left_history
    st.session_state.half_history_right = right_history
    st.session_state.half_results = {
        "left": left_point,
        "right": right_point,
        "target": half_value,
    }
    st.session_state.active_animation = None
    render_actuator_plot(st.session_state.actuator_position)
    render_dose_plot()


render_actuator_plot(current_position)
render_dose_plot()

st.divider()

controls_container = st.container()
status_container = st.container()

with controls_container:
    control_cols = st.columns(2)
    control_cols[0].button(
        "Run Gradient Ascent",
        on_click=run_gradient_animation,
        disabled=st.session_state.active_animation is not None,
        use_container_width=True,
    )
    control_cols[1].button(
        "Find Half-Dose Points",
        on_click=run_half_search_animation,
        disabled=(
            st.session_state.active_animation is not None
            or not st.session_state.gradient_results
        ),
        use_container_width=True,
    )

status_payload = {
    "actuator_position_mm": round(st.session_state.actuator_position, 3),
    "gradient": {
        "running": st.session_state.active_animation == "gradient",
        "steps_recorded": len(st.session_state.gradient_history),
        "result": st.session_state.gradient_results,
    },
    "half_search": {
        "running": st.session_state.active_animation == "half",
        "left_history_count": len(st.session_state.half_history_left),
        "right_history_count": len(st.session_state.half_history_right),
        "result": st.session_state.half_results,
    },
}

with status_container:
    st.json(status_payload, expanded=False)