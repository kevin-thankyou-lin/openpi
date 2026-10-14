"""Pinned July SuP identity for the separate matched comparator branch."""
WEIGHTS = {0.0175: "93fedf6e3deb7e5a64cbbe613d58034c90bb01e23b60b57c66056468cd5a8214",
           0.02: "c1b7f7fc8d33f0adb98dd97e3add3d3e94538ea7dafb65e08b2647fa7a36cc64"}
SOURCE_HASHES = {
    "speedtuning_rl/sup/iql.py": "a63a60b5da6c5501317079760745c8ef67f0c3ae08eb3247cd4b65dcf8b44657",
    "speedtuning_rl/sup/models.py": "c116069e4d28c6704962ad2968dac169934737f69efe12be717d8ecd8ead949c",
    "speedtuning_rl/sup/actions.py": "fafb1187a2dc1cf901d6aacd18ed9167dd607d571310ac9f4273537ffb8c2ae2",
}

def expected_selector_metadata(route):
    route.verify()
    if route.method != "sup":
        raise ValueError("SuP selector requires a SuP route")
    return {"method": "sup_iql_scheduler", "protocol": "predict_k_v1",
            "scheduler_sha256": route.sup_checkpoint.sha256, "scheduler_bytes": 5586716,
            "epsilon": route.sup_epsilon, "k_values": [1, 2], "state_dim": 8,
            "action_dim": 7, "chunk_size": 10, "control_hz": 20,
            "action_protocol": "protocol_v2",
            "action_composition": "controller-space sum-position left-compose-rotation last-gripper",
            "source_sha256": SOURCE_HASHES}

def validate_selector_metadata(value, route):
    expected = expected_selector_metadata(route)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("SuP selector handshake differs from pinned learned checkpoint/config")
