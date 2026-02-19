class ProbabilityCalibrator:

    def __init__(self, method: str = "identity", model=None):
        self.method = method
        self.model = model

    def transform(self, X):
        import numpy as np

        values = np.asarray(X, dtype=float)
        if self.method == "identity" or self.model is None:
            return values

        if hasattr(self.model, "predict"):
            return self.model.predict(values)

        raise ValueError(f"Unsupported calibrator method '{self.method}' without a compatible model.")

    def save_state(self, output_dir: str = "models"):
        import os
        import json
        import joblib

        os.makedirs(output_dir, exist_ok=True)
        state = {"method": self.method}
        if self.model is not None:
            model_path = os.path.join(output_dir, "calibrator_model.joblib")
            joblib.dump(self.model, model_path)
            state["model_filename"] = "calibrator_model.joblib"

        state_path = os.path.join(output_dir, "calibrator_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, input_dir: str = "models") -> "ProbabilityCalibrator":
        """Load calibrator from state directory (JSON config + optional joblib model)."""
        import os
        import json
        import joblib

        state_path = os.path.join(input_dir, "calibrator_state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Calibration state not found: {state_path}")

        with open(state_path, "r") as f:
            state = json.load(f)

        method = state.get("method", "identity")
        model = None
        model_filename = state.get("model_filename")
        if model_filename:
            model_path = os.path.join(input_dir, model_filename)
            model = joblib.load(model_path)

        return cls(method=method, model=model)
