import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from models import Measurement, MeasurementResults

st.set_page_config(page_title="Actuator Simulator", layout="wide")

st.title("Measurement Setup")

# Pre-populate with the values you supplied
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

    submitted = st.form_submit_button("Record Measurement")
    if submitted:
        st.session_state.measurement = Measurement(
            colimiator_size=collimator_size,
            exposure_time_s=int(exposure_time_min * 60),
            actuator_speed_mm_s=actuator_speed,
            dose_rate_gy_s=dose_rate / 60.0,
        ).dict()
        st.success("Measurement saved to session state.")

st.divider()
max_travel_mm = 50.0
x_axis = np.linspace(0, max_travel_mm, 200)

layout_col_pos, layout_col_plot = st.columns([1, 2], gap="large")

with layout_col_pos:
    if "actuator_position" not in st.session_state:
        st.session_state.actuator_position = 0.0

    current_position = st.session_state.actuator_position

    if "gradient_state" not in st.session_state:
        st.session_state.gradient_state = {
            "running": False,
            "position": current_position,
            "history": [],
            "step": 0,
        }

    if "half_search_state" not in st.session_state:
        st.session_state.half_search_state = None

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
        y0=current_position,
        y1=current_position,
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

    st.plotly_chart(fig, use_container_width=True)

def trigger_gradient_start():
    gradient_state = st.session_state.gradient_state
    current_position = st.session_state.actuator_position
    gradient_state["running"] = True
    gradient_state["position"] = current_position
    gradient_state["history"] = []
    gradient_state["step"] = 0
    st.session_state.half_search_state = None
    st.session_state.pop("gradient_results", None)
    st.session_state.pop("half_results", None)
    st.rerun()


def trigger_half_search_start():
    if st.session_state.gradient_state["running"]:
        return
    center_ref = st.session_state.actuator_position
    st.session_state.half_search_state = {
        "running": True,
        "side": "left",
        "center": center_ref,
        "left": {
            "low": 0.0,
            "high": center_ref,
            "history": [],
            "result": None,
            "position_sequence": [],
            "pending_mid": None,
        },
        "right": {
            "low": center_ref,
            "high": max_travel_mm,
            "history": [],
            "result": None,
            "position_sequence": [],
            "pending_mid": None,
        },
        "step": 0,
    }
    st.session_state.pop("half_results", None)
    st.rerun()

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
next_rerun_delay: float | None = None

with layout_col_plot:
    graph_placeholder = st.empty()

    dose_fig = go.Figure()
    dose_fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=dose_profile,
            mode="lines",
            name="D(x)",
            line=dict(color="royalblue"),
        )
    )
    dose_fig.add_vline(
        x=current_position,
        line=dict(color="red", dash="dash"),
        annotation_text="Actuator",
        annotation_position="top",
    )
    dose_fig.add_hline(
        y=half_value,
        line=dict(color="gray", dash="dot"),
        annotation_text="Half dose",
        annotation_position="bottom right",
    )

    learning_rate = 5
    max_steps = 200
    grad_tolerance = 1e-3
    frame_delay = 0.05
    binary_tolerance = 1e-2
    position_tolerance = 0.05
    binary_frame_delay = 0.4

    gradient_state = st.session_state.gradient_state
    half_state = st.session_state.half_search_state

    latest_grad = None
    latest_dose = None

    if gradient_state["running"]:
        current_iter_position = gradient_state["position"]
        st.session_state.actuator_position = current_iter_position
        latest_dose = gaussian(current_iter_position)
        latest_grad = gaussian_grad(current_iter_position)
        gradient_state["history"].append((current_iter_position, latest_dose))
        gradient_state["step"] += 1
    else:
        current_iter_position = (
            gradient_state["history"][-1][0]
            if gradient_state["history"]
            else gradient_state.get("position", current_position)
        )

    if gradient_state["history"]:
        dose_fig.add_trace(
            go.Scatter(
                x=[pt[0] for pt in gradient_state["history"]],
                y=[pt[1] for pt in gradient_state["history"]],
                mode="markers+lines",
                name="Gradient Path",
                marker=dict(color="orange", size=8),
            )
        )

    if gradient_state["running"] and latest_grad is not None:
        converged = abs(latest_grad) < grad_tolerance or gradient_state["step"] >= max_steps
        if converged:
            gradient_state["running"] = False
            center_mm = gradient_state["history"][-1][0]
            max_dose = gradient_state["history"][-1][1]
            st.session_state.actuator_position = center_mm
            st.session_state["gradient_results"] = {
                "center_mm": center_mm,
                "max_dose": max_dose,
                "half_dose": half_value,
                "left_half": left_half_point,
                "right_half": right_half_point,
                "steps": gradient_state["history"],
            }
        else:
            next_position = float(
                np.clip(
                    current_iter_position + learning_rate * latest_grad,
                    0.0,
                    max_travel_mm,
                )
            )
            gradient_state["position"] = next_position
            st.session_state.actuator_position = next_position
            next_rerun_delay = frame_delay

    half_state = st.session_state.half_search_state

    def append_history_trace(segment, label, color):
        if segment and segment["history"]:
            dose_fig.add_trace(
                go.Scatter(
                    x=[pt[0] for pt in segment["history"]],
                    y=[pt[1] for pt in segment["history"]],
                    mode="markers+lines",
                    name=label,
                    marker=dict(color=color, size=8),
                )
            )

    def ensure_sequence(start: float, end: float) -> list[float]:
        direction = 1 if end >= start else -1
        path = np.arange(start, end, direction * 0.5)
        if path.size == 0 or path[-1] != end:
            path = np.append(path, end)
        return path.tolist()

    def binary_search_step(segment, is_left: bool):
        if "position_sequence" in segment and segment["position_sequence"]:
            next_position = segment["position_sequence"].pop(0)
            st.session_state.actuator_position = float(np.clip(next_position, 0.0, max_travel_mm))
            return False, st.session_state.actuator_position, gaussian(st.session_state.actuator_position)

        if segment.get("pending_mid") is not None:
            mid = segment.pop("pending_mid")
            mid_dose = gaussian(mid)
            segment["history"].append((mid, mid_dose))

            completed = False
            if abs(mid_dose - half_value) < binary_tolerance or abs(segment["high"] - segment["low"]) < position_tolerance:
                segment["result"] = mid
                completed = True
                target = mid
            else:
                if is_left:
                    if mid_dose > half_value:
                        segment["high"] = mid
                    else:
                        segment["low"] = mid
                else:
                    if mid_dose > half_value:
                        segment["low"] = mid
                    else:
                        segment["high"] = mid
                target = (segment["low"] + segment["high"]) / 2.0

            if completed:
                return True, mid, mid_dose

            segment["pending_mid"] = None
            current_position = st.session_state.actuator_position
            segment["position_sequence"] = ensure_sequence(current_position, target)
            segment["pending_mid"] = target

            if segment["position_sequence"]:
                next_position = segment["position_sequence"].pop(0)
                st.session_state.actuator_position = float(np.clip(next_position, 0.0, max_travel_mm))
                return False, st.session_state.actuator_position, gaussian(st.session_state.actuator_position)
            else:
                return False, target, gaussian(target)

        low = segment["low"]
        high = segment["high"]
        mid = (low + high) / 2.0
        segment["pending_mid"] = mid
        current_position = st.session_state.actuator_position
        segment["position_sequence"] = ensure_sequence(current_position, mid)

        if segment["position_sequence"]:
            next_position = segment["position_sequence"].pop(0)
            st.session_state.actuator_position = float(np.clip(next_position, 0.0, max_travel_mm))
            return False, st.session_state.actuator_position, gaussian(st.session_state.actuator_position)

        return False, mid, gaussian(mid)

    if half_state and half_state.get("running"):
        current_side = half_state["side"]
        if current_side == "left":
            done, mid, mid_dose = binary_search_step(half_state["left"], True)
            if done:
                half_state["side"] = "right"
        else:
            done, mid, mid_dose = binary_search_step(half_state["right"], False)
            if done:
                half_state["running"] = False
                left_res = half_state["left"]["result"]
                right_res = half_state["right"]["result"]
                st.session_state["half_results"] = {
                    "left": left_res,
                    "right": right_res,
                    "target": half_value,
                }
        if half_state.get("running"):
            next_rerun_delay = binary_frame_delay

    if half_state:
        append_history_trace(half_state["left"], "Left Search", "green")
        append_history_trace(half_state["right"], "Right Search", "purple")

    half_results = st.session_state.get("half_results")
    if half_results:
        dose_fig.add_trace(
            go.Scatter(
                x=[half_results["left"], half_results["right"]],
                y=[half_value, half_value],
                mode="markers",
                name="Half-dose points",
                marker=dict(color="gold", size=10, symbol="diamond"),
            )
        )

    dose_fig.update_layout(
        xaxis_title="Position (mm)",
        yaxis_title="Dose Rate (Gy/s)",
        title="Dose Profile & Searches",
        title_x=0.5,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    graph_placeholder.plotly_chart(dose_fig, use_container_width=True)

st.divider()

controls_container = st.container()
status_container = st.container()

with controls_container:
    control_cols = st.columns(2)
    control_cols[0].button(
        "Run Gradient Ascent",
        on_click=trigger_gradient_start,
        disabled=gradient_state["running"],
        use_container_width=True,
    )
    control_cols[1].button(
        "Find Half-Dose Points",
        on_click=trigger_half_search_start,
        disabled=(
            gradient_state["running"]
            or (half_state is not None and half_state.get("running") is True)
        ),
        use_container_width=True,
    )

status_payload = {
    "actuator_position_mm": round(st.session_state.actuator_position, 3),
    "gradient": {
        "running": gradient_state["running"],
        "step": gradient_state["step"],
        "current_estimate_mm": round(current_iter_position, 3)
        if current_iter_position is not None
        else None,
        "current_dose_Gy_s": round(latest_dose, 4) if latest_dose is not None else None,
        "current_gradient": round(latest_grad, 6) if latest_grad is not None else None,
        "result": st.session_state.get("gradient_results"),
    },
    "half_search": {
        "running": half_state["running"] if half_state else False,
        "side": half_state["side"] if half_state else None,
        "left_bounds_mm": [
            round(half_state["left"]["low"], 3),
            round(half_state["left"]["high"], 3),
        ]
        if half_state
        else None,
        "right_bounds_mm": [
            round(half_state["right"]["low"], 3),
            round(half_state["right"]["high"], 3),
        ]
        if half_state
        else None,
        "result": st.session_state.get("half_results"),
    },
}

with status_container:
    st.json(status_payload, expanded=False)

if next_rerun_delay is not None:
    time.sleep(next_rerun_delay)
    st.rerun()