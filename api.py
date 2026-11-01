from flask import Flask, jsonify, request

from visualization import parse_graph, search_graph


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/search")
def run_search():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        graph = parse_graph(payload.get("graph"))
        result = search_graph(
            graph,
            payload.get("start"),
            payload.get("goal"),
            payload.get("algorithm"),
            heuristic=payload.get("heuristic"),
            max_depth=payload.get("max_depth", 20),
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(result)
