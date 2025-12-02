from datetime import date
from typing import Optional

from pydantic import BaseModel


class Instrumentation(BaseModel):
    manafacturer: str
    gk_model: str
    serial_number: str
    beam_energy: str
    phantom: str
    diameter_mm: int


class MeasurementConditions():
    temperature: int
    pressure: int
    electrometer_model: str
    serial_number: str
    date_of_report: date


class Measurement(BaseModel):
    colimiator_size: int
    exposure_time_s: int
    actuator_speed_mm_s: int
    dose_rate_gy_s: int

class MeasurementResults(BaseModel):
    x_axis_mm: list[float]
    y_axis_gy: list[float]
    center_point_mm: Optional[float] = None
    left_half_point: Optional[float] = None
    right_half_point: Optional[float] = None
    fwhm_mm: float


