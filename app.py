import streamlit as st
from models import Instrumentation, MeasurementConditions
from datetime import date

# Page selector
if "page" not in st.session_state:
    st.session_state.page = 0

if st.session_state.page == 0:
    st.title("Instrumentation Form")

    with st.form("instr_form"):
        manafacturer = st.text_input("Manufacturer", value="Elekta")
        gk_model = st.text_input("GK Model", value="LGK ESPRIT")
        serial_number = st.text_input("Serial Number", value="8156")
        beam_energy = st.text_input("Beam Energy", value="60Co")
        phantom = st.text_input("Phantom", value="Spherical Solid Water")
        diameter_mm = st.number_input("Diameter (mm)", min_value=0, step=1, value=160)

        submitted = st.form_submit_button("Next")
        if submitted:
            st.session_state.instrumentation = {
                "manafacturer": manafacturer,
                "gk_model": gk_model,
                "serial_number": serial_number,
                "beam_energy": beam_energy,
                "phantom": phantom,
                "diameter_mm": diameter_mm,
            }
            st.session_state.page = 1
            st.rerun()

elif st.session_state.page == 1:
    st.title("Measurement Conditions Form")

    with st.form("meas_form"):
        temperature = st.number_input("Temperature (°C)", step=0.1, value=0.0)
        pressure = st.number_input("Pressure (mmHg)", step=0.1, value=0.0)
        electrometer_model = st.text_input("Electrometer Model", value="PTW Unidos")
        serial_number = st.text_input(
            "Electrometer Serial Number", key="em_serial_number", value="50092"
        )
        electr_corr_factor = st.number_input("Electrometer Corr. Factor (Pelec)", value=1.000, step=0.001, format="%.3f")
        chamber_model = st.text_input("Chamber Model", value="Exradin A-16")
        chamber_serial = st.text_input("Chamber Serial Number", value="XAA070958")
        calibration_factor = st.number_input("Calibration Factor ND,w60Co (cGy/nC)", value=4.468, step=0.001, format="%.3f")
        date_of_report = st.date_input("Date of Report", value=date(2024, 1, 24))

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.measurement_conditions = {
                "temperature": temperature,
                "pressure": pressure,
                "electrometer_model": electrometer_model,
                "serial_number": serial_number,
                "date_of_report": date_of_report,
            }
            st.session_state.page = 2
            st.rerun()

elif st.session_state.page == 2:
    st.title("Done")
    st.write("Instrumentation Data:")
    st.write(st.session_state.get("instrumentation", {}))
    st.write("Measurement Conditions:")
    st.write(st.session_state.get("measurement_conditions", {}))
