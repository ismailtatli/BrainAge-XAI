"""Public result contract and non-diagnostic interpretation, without participant identity."""

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from brainage_xai.modeling.statistics import apply_bias, reference_position

AGE_WARNING = (
    "Age is outside the model's observed training/reference range; interpretation is extrapolative."
)
XAI_WORDING = (
    "Grad-CAM indicates regions receiving relatively greater attribution "
    "for the model's prediction."
)
RESEARCH_DISCLAIMER = (
    "BrainAge-XAI is a research prototype. BAG is not a diagnosis. The model was "
    "trained/evaluated on IXI; external clinical validity has not been established. "
    "Corrected brain age is a bias-adjusted model estimate, not a medical measurement."
)
LIMITATIONS = [
    "Corrected test BAG retains age association (r = -0.279); correction reduces but "
    "does not guarantee age-independence.",
    "Zero Grad-CAM maps and background/nonbrain attribution may occur; attribution is "
    "not causal or diagnostic.",
    "IXI cohort size, site/scanner confounding, 2.5D inputs and residual rigid alignment "
    "limit generalization; IOP contrast/visibility limitations remain.",
    "Reference context is validation-derived, not an age/site-specific clinical norm. "
    "No independent external cohort or clinical validation is available.",
    "The upload's T1 modality and population similarity cannot be established automatically.",
]


def validate_age(age):
    if (
        isinstance(age, bool)
        or not isinstance(age, (int, float))
        or not np.isfinite(age)
        or age < 0
    ):
        raise ValueError("Chronological age in years must be finite and nonnegative")
    return float(age)


def interpretation(bag):
    if round(bag, 2) == 0:
        return (
            "Corrected BAG rounds to 0.00 years at the displayed precision. "
            "Its numerical research-reference position defines no clinical normal range."
        )
    if bag > 0:
        return (
            "The model estimated an older-looking structural brain pattern relative "
            "to chronological age and the learned research reference."
        )
    if bag < 0:
        return (
            "The model estimated a younger-looking structural brain pattern relative "
            "to chronological age and the learned research reference."
        )
    raise ValueError("BAG must be finite")


def age_warnings(age, ranges):
    outside = [k for k, (lo, hi) in ranges.items() if not lo <= age <= hi]
    return [AGE_WARNING] if outside else []


def bag_values(age, prediction, config, analysis_id):
    f = pd.DataFrame(
        {
            "subject_id": [analysis_id],
            "split": ["inference"],
            "age": [age],
            "predicted_age": [prediction],
        }
    )
    corrected = apply_bias(f, config.correction).iloc[0]
    percentile = float(reference_position([corrected.corrected_BAG], config.reference)[0])
    lo, hi = config.reference["raw_residual_quantiles"]
    return {
        "raw_BAG": float(corrected.raw_BAG),
        "corrected_BAG": float(corrected.corrected_BAG),
        "corrected_brain_age": float(corrected.corrected_brain_age),
        "reference_percentile": percentile,
        "empirical_uncertainty": {
            "label": "empirical model uncertainty",
            "reference": "research-model reference",
            "quantity": "chronological-age reference interval from raw prediction",
            "lower_years": float(prediction - hi),
            "upper_years": float(prediction - lo),
            "validation_residual_quantiles": [lo, hi],
            "nominal_percentiles": [2.5, 97.5],
            "scope": "Marginal empirical reference interval; no conditional coverage guarantee "
            "or clinical confidence interpretation.",
        },
    }


@dataclass(frozen=True)
class InferenceResult:
    schema_version: int
    analysis_id: str
    timestamp_utc: str
    application_version: str
    chronological_age: float
    raw_predicted_brain_age: float
    raw_BAG: float
    corrected_brain_age: float
    corrected_BAG: float
    reference_percentile: float
    empirical_uncertainty: dict
    preprocessing_qc: dict
    registration_qc: dict
    provenance: dict
    observed_age_ranges_years: dict
    warnings: list
    xai: dict
    interpretation: str
    disclaimer: str = RESEARCH_DISCLAIMER

    def __post_init__(self):
        values = [
            self.chronological_age,
            self.raw_predicted_brain_age,
            self.raw_BAG,
            self.corrected_brain_age,
            self.corrected_BAG,
            self.reference_percentile,
        ]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite inference result")
        if not 0 <= self.reference_percentile <= 100:
            raise ValueError("Invalid reference percentile")
        if not np.isclose(self.raw_BAG, self.raw_predicted_brain_age - self.chronological_age):
            raise ValueError("Raw BAG sign mismatch")
        if not np.isclose(self.corrected_brain_age, self.chronological_age + self.corrected_BAG):
            raise ValueError("Corrected brain age mismatch")

    def as_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n"
