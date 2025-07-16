from calibration.calibrate import calibrate
from calibration.dataset import CalibrationDataset


def test_calibration_reduces_ece():
    data = CalibrationDataset.load_sample()
    _, before, after = calibrate(data)
    assert after <= 0.8 * before
