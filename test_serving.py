from src.serving.predictor import (
    predict_failure
)


sample_machine = {
    "Type": 1,
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 100,
    "power_watts": 6951.5906,
    "temp_delta_K": 10.5,
    "tool_wear_torque_interaction": 4280,
    "tool_wear_rpm_interaction": 155100,
    "strain_index": 2.7576
}


result = predict_failure(
    sample_machine
)

print("\nPrediction Result\n")

print(result)