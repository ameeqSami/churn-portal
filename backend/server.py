import os
import sys
from flask import Flask, jsonify, request, send_from_directory

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_service import ChurnModelService

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Initialize model service on startup
print("Initializing Churn Model Service...")
model_service = ChurnModelService()

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    try:
        metrics = model_service.get_summary_metrics()
        return jsonify({"status": "success", "data": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/customers", methods=["GET"])
def get_customers():
    try:
        query = request.args.get("q", default="", type=str)
        risk = request.args.get("risk", default="all", type=str)
        contract = request.args.get("contract", default="all", type=str)
        internet = request.args.get("internet", default="all", type=str)
        sort_by = request.args.get("sort_by", default="churn_prob", type=str)
        sort_dir = request.args.get("sort_dir", default="desc", type=str)
        page = request.args.get("page", default=1, type=int)
        per_page = request.args.get("per_page", default=25, type=int)

        result = model_service.search_customers(
            query=query,
            risk_tier=risk,
            contract=contract,
            internet=internet,
            sort_by=sort_by,
            sort_dir=sort_dir,
            page=page,
            per_page=per_page
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict_all", methods=["POST"])
def predict_all():
    try:
        metrics = model_service.predict_all()
        return jsonify({"status": "success", "data": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/reset_predictions", methods=["POST"])
def reset_predictions():
    try:
        metrics = model_service.reset_predictions()
        return jsonify({"status": "success", "data": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/customers/<int:test_index>/predict", methods=["POST"])
def predict_customer(test_index):
    try:
        customer = model_service.predict_customer(test_index)
        metrics = model_service.get_summary_metrics()
        return jsonify({"status": "success", "data": {"customer": customer, "metrics": metrics}})
    except IndexError:
        return jsonify({"status": "error", "message": "Customer index not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/customers/<int:test_index>/analyze", methods=["GET"])
def analyze_customer(test_index):
    try:
        analysis = model_service.analyze_customer(test_index)
        return jsonify({"status": "success", "data": analysis})
    except IndexError:
        return jsonify({"status": "error", "message": "Customer index not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/customers/<int:test_index>/simulate", methods=["POST"])
def simulate_customer(test_index):
    try:
        payload = request.get_json() or {}
        contract_override = payload.get("contract")
        payment_override = payload.get("payment_method")
        tech_support_override = payload.get("tech_support")
        tenure_override = payload.get("tenure")
        if tenure_override is not None:
            tenure_override = int(tenure_override)

        sim_result = model_service.simulate_what_if(
            test_index=test_index,
            contract_override=contract_override,
            payment_override=payment_override,
            tech_support_override=tech_support_override,
            tenure_months_override=tenure_override
        )
        return jsonify({"status": "success", "data": sim_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n=======================================================")
    print(f" Telco Churn Intelligence Portal Server")
    print(f" Model: DecisionTree from chrunPipe.ipynb")
    print(f" URL: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
