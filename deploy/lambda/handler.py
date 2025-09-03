import json
import os
import time
import numpy as np
import tensorflow as tf
import joblib
from base64 import b64decode
import sys
sys.path.append('/var/task')
from models.dqn.agent_lw import LightweightDQN
from models.dqn.cooldown_manager import CooldownManager

# Globals for Lambda warm start
bigru_interpreter = None
bigru_scaler = None
dqn_agent = None
model_version = "1.0.0"

# Paths for embedded models
BIGRU_MODEL_PATH = "models/bigru/weights/model_quantized.tflite"
BIGRU_SCALER_PATH = "models/bigru/weights/scaler.pkl"
DQN_MODEL_WEIGHTS_PATH = "models/dqn/weights/dqn_model.h5"
DQN_CONFIG_PATH = "models/dqn/weights/model_config.json"

def load_bigru():
    global bigru_interpreter, bigru_scaler
    if bigru_interpreter is None:
        bigru_interpreter = tf.lite.Interpreter(model_path=BIGRU_MODEL_PATH)
        bigru_interpreter.allocate_tensors()
    if bigru_scaler is None:
        bigru_scaler = joblib.load(BIGRU_SCALER_PATH)
    return bigru_interpreter, bigru_scaler

def load_dqn():
    global dqn_agent
    if dqn_agent is None:
        # Load config JSON
        with open(DQN_CONFIG_PATH, "r") as f:
            config = json.load(f)
        state_size = config['state_size']
        action_size = config['action_size']
        cooldown_config = config['cooldown_config']
        # Instantiate agent and load weights
        dqn_agent = LightweightDQN(state_size, action_size, cooldown_config)
        dqn_agent.model.load_weights(DQN_MODEL_WEIGHTS_PATH)
    return dqn_agent

def load_models():
    """Load both models for Lambda cold start."""
    load_bigru()
    load_dqn()

def lambda_handler(event, context):
    """Lambda entry point. Handles /balance, /inject-model, and /version."""
    # Load models if not warm
    load_models()

    path = event.get("path", "")
    http_method = event.get("httpMethod", "POST")
    body = event.get("body", "{}")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = {}

    if path.endswith("/balance") and http_method == "POST":
        return handle_balance(body)
    elif path.endswith("/inject-model") and http_method == "POST":
        return handle_inject_model(body)
    elif path.endswith("/version") and http_method == "GET":
        return {
            "statusCode": 200,
            "body": json.dumps({"model_version": model_version}),
        }
    else:
        return {"statusCode": 404, "body": json.dumps({"error": "Invalid endpoint"})}

def handle_balance(body):
    """Core load balancing entrypoint."""
    global dqn_agent
    metrics = body.get("metrics", {})
    bigru_interpreter, bigru_scaler = load_bigru()
    dqn_agent = load_dqn()

    # Prepare state (use last 24 steps if available)
    state_seq = prepare_state_sequence(metrics, bigru_scaler)
    if state_seq is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid or missing metrics for prediction"}),
        }

    # BiGRU prediction (forecasting workload)
    prediction = predict_workload(state_seq)

    # Compose DQN state
    dqn_state = prepare_dqn_state(metrics, prediction)

    # DQN action (cooldown-aware)
    action = dqn_agent.act(dqn_state)

    # Enforce cooldown (agent handles masking; double-check)
    if hasattr(dqn_agent, "cooldown_manager") and not dqn_agent.cooldown_manager.is_action_valid(action):
        action = 0  # fallback: no_action

    routing = map_action_to_routing(action, metrics)
    cooldown_status = dqn_agent.cooldown_manager.get_status() if hasattr(dqn_agent, "cooldown_manager") else {}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "routing": routing,
            "predicted_load": prediction.tolist(),
            "dqn_action": int(action),
            "cooldown_status": cooldown_status,
        }),
    }

def prepare_state_sequence(metrics, scaler):
    """Prepare a (1, 24, 3) input for BiGRU from metrics buffer or incoming history."""
    # Lambda: expects body['metrics'] = [{...}, ...] or {...} with recent values
    if isinstance(metrics, list) and len(metrics) >= 24:
        # Use last 24 points
        mtx = []
        for m in metrics[-24:]:
            mtx.append([
                m.get("cpu_utilization", 0),
                m.get("memory_utilization", 0),
                m.get("net_out", 0),
            ])
        arr = np.array(mtx)
    elif isinstance(metrics, dict):
        # Single snapshot - can't forecast, fallback
        arr = np.zeros((24, 3))
        arr[-1, 0] = metrics.get("cpu_utilization", 0)
        arr[-1, 1] = metrics.get("memory_utilization", 0)
        arr[-1, 2] = metrics.get("net_out", 0)
    else:
        return None

    # Scale
    arr_scaled = scaler.transform(arr)
    arr_scaled = arr_scaled.reshape(1, 24, 3)
    return arr_scaled

def predict_workload(state_seq):
    """Run BiGRU model to get workload prediction (for Lambda)."""
    global bigru_interpreter
    input_index = bigru_interpreter.get_input_details()[0]["index"]
    output_index = bigru_interpreter.get_output_details()[0]["index"]
    bigru_interpreter.set_tensor(input_index, state_seq.astype(np.float32))
    bigru_interpreter.invoke()
    prediction = bigru_interpreter.get_tensor(output_index)
    return prediction[0]  # shape: (12,)

def prepare_dqn_state(metrics, prediction):
    """Assemble DQN agent state vector: [metrics + prediction]."""
    # Example metrics: cpu_utilization, memory_utilization, request_rate, response_time_p95, active_connections
    # (Scale/normalize as needed)
    state = np.array([
        metrics.get('cpu_utilization', 0) / 100,
        metrics.get('memory_utilization', 0) / 100,
        metrics.get('request_rate', 0) / 1000,
        metrics.get('response_time_p95', 0) / 1000,
        metrics.get('active_connections', 0) / 100,
    ])
    # Pad prediction to fixed size if needed
    pred_pad = np.zeros(8)
    pred_len = min(len(prediction), 8)
    pred_pad[:pred_len] = prediction[:pred_len]
    # Add temporal features (if needed)
    temporal_features = extract_temporal_features(metrics)
    dqn_state = np.concatenate([state, pred_pad, temporal_features])
    return dqn_state

def map_action_to_routing(action, metrics):
    """Map DQN action to routing decision."""
    mapping = ["no_action", "scale_up", "scale_down"]
    if isinstance(action, (np.integer, int)):
        action = int(action)
    return mapping[action % len(mapping)]

def extract_temporal_features(metrics):
    """Extract temporal features from metrics (stub for completeness)."""
    # These could be last N CPU/memory, or just zeros if not available
    # Here, return a vector of zeros for placeholder
    return np.zeros(8)

def handle_inject_model(body):
    """Hot model update for BiGRU or DQN; validates and replaces in-memory model."""
    global bigru_interpreter, dqn_agent, model_version, bigru_scaler
    try:
        model_type = body.get("model_type")
        version = body.get("version", "unknown")
        model_data_b64 = body.get("model_data", None)
        if not model_type or not model_data_b64:
            raise ValueError("Both model_type and model_data required")

        model_data = b64decode(model_data_b64)

        if model_type == "bigru":
            # Validate new TFLite model
            interpreter = tf.lite.Interpreter(model_content=model_data)
            interpreter.allocate_tensors()
            # Test inference (dummy input)
            input_shape = interpreter.get_input_details()[0]["shape"]
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], np.zeros(input_shape, dtype=np.float32))
            interpreter.invoke()
            # Swap loaded model
            bigru_interpreter = interpreter
            model_version = version
            # Optionally update scaler if provided
            scaler_data_b64 = body.get("scaler_data")
            if scaler_data_b64:
                bigru_scaler = joblib.loads(b64decode(scaler_data_b64))

        elif model_type == "dqn":
            # Expecting keras weights (.h5) as model_data, and config as JSON
            # Save weights to temp file and reload agent
            config_b64 = body.get("config_data")
            if config_b64:
                config = json.loads(b64decode(config_b64).decode())
            else:
                # Fallback: use existing config
                with open(DQN_CONFIG_PATH, "r") as f:
                    config = json.load(f)
            state_size = config['state_size']
            action_size = config['action_size']
            cooldown_config = config['cooldown_config']
            new_agent = LightweightDQN(state_size, action_size, cooldown_config)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
                f.write(model_data)
                f.flush()
                new_agent.model.load_weights(f.name)
            dqn_agent = new_agent
            model_version = version

        else:
            raise ValueError("Unsupported model_type")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"{model_type} model updated",
                "version": version,
            }),
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)}),
        }