"""
Analysis Service — API Routes
================================
Endpoints for scenario switching, pipeline status, and attribution results.
"""

from flask import Blueprint, request, jsonify
from .detector import (
    get_detector_state,
    get_generator,
    get_scorer,
    analyze_payload,
)

analysis_bp = Blueprint("analysis", __name__)


# ---------------------------------------------------------------------------
# Legacy endpoint (kept for backward compat with Node-RED flows)
# ---------------------------------------------------------------------------

@analysis_bp.route("/analyze", methods=["POST"])
def analyze():
    """Analyze incoming sensor data (legacy endpoint)."""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    if isinstance(data, list):
        for item in data:
            analyze_payload(item)
    else:
        analyze_payload(data)

    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Service Status
# ---------------------------------------------------------------------------

@analysis_bp.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "running", "service": "LA-DT Analysis Engine"})


# ---------------------------------------------------------------------------
# Scenario Management
# ---------------------------------------------------------------------------

@analysis_bp.route("/scenario", methods=["GET"])
def get_scenario():
    """Get current active scenario."""
    gen = get_generator()
    return jsonify({
        "scenario": gen.scenario,
        "available": ["normal", "byzantine", "fdi"],
    })


@analysis_bp.route("/scenario", methods=["POST"])
def set_scenario():
    """
    Switch the active data generation scenario.

    Body: {"scenario": "normal" | "byzantine" | "fdi"}
    """
    data = request.json or {}
    scenario = data.get("scenario", "").lower()

    if scenario not in ("normal", "byzantine", "fdi"):
        return jsonify({
            "error": f"Unknown scenario '{scenario}'",
            "available": ["normal", "byzantine", "fdi"],
        }), 400

    gen = get_generator()
    gen.set_scenario(scenario)

    # Reset the scorer baselines when switching scenarios
    scorer = get_scorer()
    scorer.reset()

    # Update detector state
    state = get_detector_state()
    state.active_scenario = scenario

    return jsonify({
        "status": "ok",
        "scenario": scenario,
        "message": f"Switched to '{scenario}' scenario. Scorer baselines reset.",
    })


# ---------------------------------------------------------------------------
# Pipeline Status & Results
# ---------------------------------------------------------------------------

@analysis_bp.route("/pipeline/status", methods=["GET"])
def pipeline_status():
    """Get detector pipeline status."""
    state = get_detector_state()
    return jsonify(state.get_status())


@analysis_bp.route("/pipeline/results", methods=["GET"])
def pipeline_results():
    """Get all attribution results."""
    state = get_detector_state()
    results = state.get_results()

    # Optional filters
    verdict_filter = request.args.get("verdict")
    if verdict_filter:
        results = [r for r in results if r["verdict"] == verdict_filter]

    return jsonify({
        "count": len(results),
        "results": results,
    })


@analysis_bp.route("/pipeline/results/<result_id>", methods=["GET"])
def pipeline_result_detail(result_id):
    """Get a single attribution result by ID."""
    state = get_detector_state()
    result = state.get_result_by_id(result_id)

    if not result:
        return jsonify({"error": "Result not found"}), 404

    return jsonify(result)


@analysis_bp.route("/pipeline/scores", methods=["GET"])
def pipeline_scores():
    """Get recent anomaly scores (for debugging / visualization)."""
    state = get_detector_state()
    with state._lock:
        scores = list(state.recent_scores)
    return jsonify({"count": len(scores), "scores": scores})


@analysis_bp.route("/pipeline/baselines", methods=["GET"])
def pipeline_baselines():
    """Get current EWMA baselines for all sensors."""
    scorer = get_scorer()
    return jsonify(scorer.get_all_baselines_summary())


# ---------------------------------------------------------------------------
# Alerts (legacy compat)
# ---------------------------------------------------------------------------

@analysis_bp.route("/alerts", methods=["GET"])
def alerts():
    """Legacy alerts endpoint — returns attribution results formatted as alerts."""
    state = get_detector_state()
    results = state.get_results()
    legacy_alerts = []
    for r in results:
        legacy_alerts.append({
            "device_id": f"device-node_{r['trigger']['node_id']}",
            "alert_level": (
                "danger" if r["verdict"] in ("fdi", "byzantine") else "warning"
            ),
            "alert_type": r["attribution"]["verdict"],
            "alert_message": (
                f"Node {r['trigger']['node_id']} "
                f"({r['trigger']['sensor']}): "
                f"z={r['trigger']['z_score']:.2f}"
            ),
            "timestamp": r["timestamp"],
        })
    return jsonify(legacy_alerts)
