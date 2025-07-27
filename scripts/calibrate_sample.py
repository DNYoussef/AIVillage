from calibration.calibrate import calibrate
from calibration.dataset import CalibrationDataset


def main() -> None:
    data = CalibrationDataset.load_sample()
    _, before, after = calibrate(data)
    print(f"ECE before: {before:.3f} -> after: {after:.3f}")


if __name__ == "__main__":
    main()
