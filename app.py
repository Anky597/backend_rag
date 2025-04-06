# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
from gradio_client import Client
import logging
import os
import time

# --- Configuration ---
HF_GRADIO_API_SPACE = os.getenv("HF_GRADIO_API_SPACE", "Ankys/shl-recommender-api")
API_ENDPOINT = os.getenv("API_ENDPOINT", "/predict")
PORT = int(os.getenv("PORT", 8080))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Enable CORS ---
# This allows requests from any origin to all routes (*)
# For more restrictive settings in production, specify origins:
# origins = ["https://your-frontend-domain.com", "http://localhost:3000"] # Example
# CORS(app, origins=origins)
CORS(app) # Apply CORS to the Flask app instance

# --- Initialize Gradio Client ---
gradio_client = None
try:
    log.info(f"Initializing Gradio client for Space: {HF_GRADIO_API_SPACE}...")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        log.info("Using HF_TOKEN for Gradio client.")
        gradio_client = Client(HF_GRADIO_API_SPACE, hf_token=hf_token)
    else:
        log.info("No HF_TOKEN environment variable found, assuming public Gradio space.")
        gradio_client = Client(HF_GRADIO_API_SPACE)

    log.info("Gradio client initialized successfully.")

except Exception as e:
    log.critical(f"CRITICAL FAILURE: Could not initialize Gradio client for {HF_GRADIO_API_SPACE}. Error: {e}", exc_info=True)

# --- Flask Route Definitions ---
@app.route('/ask', methods=['POST'])
def ask_sync():
    """
    Synchronous endpoint to proxy requests to the Gradio API.
    Expects JSON body: {"user_question": "Your question here"}
    """
    endpoint_start_time = time.time()
    log.info(f"Received request for endpoint: {request.path}")

    if not gradio_client:
        log.error("Request received, but Gradio client is not available (initialization failed?).")
        return jsonify({"status": "error", "message": "Backend Gradio client not initialized or unavailable."}), 503 # Service Unavailable

    if not request.is_json:
        log.warning("Received non-JSON request.")
        return jsonify({"status": "error", "message": "Request body must be JSON."}), 400

    data = request.get_json()
    user_question = data.get('user_question')

    if user_question is None:
        log.warning("Received request without 'user_question' key.")
        return jsonify({"status": "error", "message": "Missing 'user_question' key in JSON body."}), 400

    log.info(f"Processing question: '{str(user_question)[:100]}...'")

    try:
        log.info(f"Calling Gradio API [{HF_GRADIO_API_SPACE}] endpoint: {API_ENDPOINT}")
        gradio_call_start_time = time.time()

        result = gradio_client.predict(
            user_question=user_question,
            api_name=API_ENDPOINT
        )

        gradio_call_duration = time.time() - gradio_call_start_time
        log.info(f"Gradio API call successful. Duration: {gradio_call_duration:.3f} seconds.")

        total_duration = time.time() - endpoint_start_time
        log.info(f"Request to {request.path} completed successfully. Total duration: {total_duration:.3f} seconds.")
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        total_duration = time.time() - endpoint_start_time
        log.error(f"Error calling Gradio API endpoint {API_ENDPOINT}: {e}", exc_info=True)
        log.info(f"Request to {request.path} failed. Total duration: {total_duration:.3f} seconds.")
        return jsonify({"status": "error", "message": f"Failed to get prediction from underlying Gradio API. Please check logs or try again later."}), 502 # Bad Gateway

@app.route('/health', methods=['GET'])
def health_check():
    """ Simple health check endpoint """
    log.debug("Health check requested.")
    if gradio_client:
        return jsonify({"status": "ok", "gradio_client_status": "initialized"}), 200
    else:
         return jsonify({"status": "error", "gradio_client_status": "not_initialized"}), 503

# No app.run() here for production deployment
