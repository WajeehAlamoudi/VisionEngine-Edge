from __future__ import annotations


def _build_payload(match, device_id: str, branch_id: str) -> dict:
    """
    Build the full alert webhook payload.

    Carries maximum context so the frontend has full flexibility:
    identity, spatial, temporal, camera, rule, and branch routing fields.
    The backend strips any fields it does not need — adding a field here
    never breaks the receiving endpoint.
    """
    det  = match.detection
    rule = match.rule

    return {
        # ── routing ───────────────────────────────────────────────────────────
        "branch_id":   branch_id,       # which branch → which WS channel to broadcast to
        "device_id":   device_id,       # which edge node sent this

        # ── rule ──────────────────────────────────────────────────────────────
        "rule_name":   rule.name,
        "severity":    rule.severity,   # critical | warning | info
        "message":     match.message,   # formatted rule message with placeholders resolved

        # ── detection identity ────────────────────────────────────────────────
        "track_id":    det.track_id,    # stable UUID string; None when tracker not active for this camera
        "class":       det.class_name,
        "confidence":  round(det.confidence, 4),
        "model_id":    det.model_id,

        # ── camera ────────────────────────────────────────────────────────────
        "camera_id":   det.camera_id,
        "camera_name": det.camera_name,

        # ── zone ──────────────────────────────────────────────────────────────
        "zone":        det.zone,

        # ── spatial — pixel coords (for bbox overlay on camera snapshot) ──────
        "bbox_x1":     int(det.bbox[0]),
        "bbox_y1":     int(det.bbox[1]),
        "bbox_x2":     int(det.bbox[2]),
        "bbox_y2":     int(det.bbox[3]),
        "anchor_x":    det.anchor_x,
        "anchor_y":    det.anchor_y,
        "frame_w":     det.frame_w,
        "frame_h":     det.frame_h,

        # ── spatial — normalized (for heatmap overlay, cross-camera comparison) ─
        "anchor_x_norm": det.anchor_x_norm,   # 0.0–1.0
        "anchor_y_norm": det.anchor_y_norm,   # 0.0–1.0

        # ── temporal ──────────────────────────────────────────────────────────
        "ts":          det.capture_ts,  # ISO 8601 UTC, grabbed at cap.read() time

        # ── attributes (optional second-stage model output) ───────────────────
        **({"attributes": det.attributes} if det.attributes else {}),
    }
