from hrdmc.workflows.dmc.rn_block_stationarity import (
    TRIANGULATED_PRECISION_WARNING,
    classify_blocked_case,
)


def test_missing_blocking_plateau_becomes_precision_warning_with_correlated_error():
    result = classify_blocked_case(
        hygiene_gate=True,
        rn_weight_status="RN_WEIGHT_GO",
        energy={"plateau_all": False, "blocked_zscore_gate": False},
        rms={"plateau_all": True, "blocked_zscore_gate": True},
        r2={"plateau_all": True, "blocked_zscore_gate": True},
        diagnostics=_diagnostics("GO"),
        stationarity_audit=_stationarity_audit("GO"),
        correlated_errors={
            "energy": _correlated_ok(),
            "rms": _correlated_ok(),
            "r2": _correlated_ok(),
        },
    )

    assert result == TRIANGULATED_PRECISION_WARNING


def test_rhat_failure_remains_hard_no_go():
    result = classify_blocked_case(
        hygiene_gate=True,
        rn_weight_status="RN_WEIGHT_GO",
        energy={"plateau_all": False, "blocked_zscore_gate": False},
        rms={"plateau_all": True, "blocked_zscore_gate": True},
        r2={"plateau_all": True, "blocked_zscore_gate": True},
        diagnostics=_diagnostics("NO_GO_RHAT"),
        stationarity_audit=_stationarity_audit("GO"),
        correlated_errors={
            "energy": _correlated_ok(),
            "rms": _correlated_ok(),
            "r2": _correlated_ok(),
        },
    )

    assert result == "NO_GO_STATIONARITY"


def test_coordinate_plateau_missing_without_correlated_error_is_mixed_warning():
    result = classify_blocked_case(
        hygiene_gate=True,
        rn_weight_status="RN_WEIGHT_GO",
        energy={"plateau_all": True, "blocked_zscore_gate": True},
        rms={"plateau_all": False, "blocked_zscore_gate": False},
        r2={"plateau_all": True, "blocked_zscore_gate": True},
        diagnostics=_diagnostics("GO"),
        stationarity_audit=_stationarity_audit("GO"),
        correlated_errors={
            "energy": _correlated_ok(),
            "rms": _correlated_unavailable(),
            "r2": _correlated_ok(),
        },
    )

    assert result == "MIXED_OBSERVABLE_WARNING"


def test_energy_plateau_missing_without_correlated_error_remains_no_go():
    result = classify_blocked_case(
        hygiene_gate=True,
        rn_weight_status="RN_WEIGHT_GO",
        energy={"plateau_all": False, "blocked_zscore_gate": False},
        rms={"plateau_all": True, "blocked_zscore_gate": True},
        r2={"plateau_all": True, "blocked_zscore_gate": True},
        diagnostics=_diagnostics("GO"),
        stationarity_audit=_stationarity_audit("GO"),
        correlated_errors={
            "energy": _correlated_unavailable(),
            "rms": _correlated_ok(),
            "r2": _correlated_ok(),
        },
    )

    assert result == "NO_GO_NO_BLOCKING_PLATEAU"


def _diagnostics(status):
    return {
        "energy": {"classification": status},
        "rms": {"classification": "GO"},
        "r2": {"classification": "GO"},
    }


def _stationarity_audit(reason):
    return {
        "energy": {"reason": reason},
        "rms": {"reason": "GO"},
        "r2": {"reason": "GO"},
    }


def _correlated_ok():
    return {
        "status": "TRIANGULATED_2_OF_3",
        "case_correlated_stderr": 0.01,
    }


def _correlated_unavailable():
    return {
        "status": "CORRELATED_ERROR_UNAVAILABLE",
        "case_correlated_stderr": float("nan"),
    }
