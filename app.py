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
# Render provides the PORT env var, but we don't directly use it in app.py when using Gunicorn
# Gunicorn will bind to the port specified in the Start Command or Procfile ($PORT)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__) # Use named logger is slightly better practice

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Enable CORS ---
CORS(app) # Apply CORS to the Flask app instance

# --- Initialize Gradio Client ---
gradio_client = None
try:
    log.info(f"Initializing Gradio client for Space: {HF_GRADIO_API_SPACE}...")
    # Check for HF_TOKEN in environment variables (set in Render dashboard if needed)
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
    # Ensure gradio_client remains None if initialization fails
    gradio_client = None

# --- Flask Route Definitions ---
@app.route('/ask', methods=['POST'])
def ask_sync():
    """
    Synchronous endpoint to proxy requests to the Gradio API (POST method).
    Expects JSON body: {"user_question": "Your question here"}
    """
    endpoint_start_time = time.time()
    log.info(f"Received POST request for endpoint: {request.path}")

    if not gradio_client:
        log.error("Request received, but Gradio client is not available (initialization failed?).")
        return jsonify({"status": "error", "message": "Backend Gradio client not initialized or unavailable."}), 503 # Service Unavailable

    if not request.is_json:
        log.warning("Received non-JSON POST request.")
        return jsonify({"status": "error", "message": "Request body must be JSON for POST requests."}), 400

    data = request.get_json()
    user_question = data.get('user_question')

    if user_question is None:
        log.warning("Received POST request without 'user_question' key.")
        return jsonify({"status": "error", "message": "Missing 'user_question' key in JSON body for POST requests."}), 400

    # Call the shared processing function
    return _process_question(user_question, endpoint_start_time)

@app.route('/ask', methods=['GET'])
def ask_get():
    """
    Synchronous endpoint to proxy requests to the Gradio API (GET method).
    Expects 'user_question' as a query parameter.
    """
    endpoint_start_time = time.time()
    log.info(f"Received GET request for endpoint: {request.path}")

    if not gradio_client:
        log.error("Request received, but Gradio client is not available (initialization failed?).")
        return jsonify({"status": "error", "message": "Backend Gradio client not initialized or unavailable."}), 503 # Service Unavailable

    # Get 'user_question' from query parameters for GET requests
    user_question = request.args.get('user_question')

    if user_question is None:
        log.warning("Received GET request without 'user_question' query parameter.")
        return jsonify({"status": "error", "message": "Missing 'user_question' query parameter for GET requests."}), 400

    # Call the shared processing function
    return _process_question(user_question, endpoint_start_time)

def _process_question(user_question, endpoint_start_time):
    """
    Helper function to process the user question and call the Gradio API.
    Called by both GET and POST /ask endpoints.
    """
    log.info(f"Processing question: '{str(user_question)[:100]}...'")

    # Double-check client status here just before calling predict
    if not gradio_client:
        log.error("Processing aborted: Gradio client is not available.")
        return jsonify({"status": "error", "message": "Backend Gradio client not initialized or unavailable."}), 503

    try:
        log.info(f"Calling Gradio API [{HF_GRADIO_API_SPACE}] endpoint: {API_ENDPOINT}")
        gradio_call_start_time = time.time()

        # Call the underlying Gradio API
        result = gradio_client.predict(
            user_question=user_question, # Make sure this matches the expected argument name in your Gradio function
            api_name=API_ENDPOINT
        )

        gradio_call_duration = time.time() - gradio_call_start_time
        log.info(f"Gradio API call successful. Duration: {gradio_call_duration:.3f} seconds.")

        total_duration = time.time() - endpoint_start_time
        log.info(f"Request to /ask completed successfully. Total duration: {total_duration:.3f} seconds.")
        # Return the successful result
        return jsonify({"status": "success", "result": result})

    except Exception as e:
        total_duration = time.time() - endpoint_start_time
        log.error(f"Error calling Gradio API endpoint {API_ENDPOINT}: {e}", exc_info=True)
        log.info(f"Request to /ask failed. Total duration: {total_duration:.3f} seconds.")
        # Return an error indicating failure during the Gradio API call
        return jsonify({"status": "error", "message": f"Failed to get prediction from underlying Gradio API. Please check logs or try again later."}), 502 # Bad Gateway

@app.route('/health', methods=['GET'])
def health_check():
    """ Simple health check endpoint """
    log.debug("Health check requested.") # Use debug for potentially frequent calls
    if gradio_client:
        # Basic check: client object exists
        return jsonify({"status": "ok", "gradio_client_status": "initialized"}), 200
    else:
        # Client initialization failed or hasn't happened
        return jsonify({"status": "error", "gradio_client_status": "not_initialized"}), 503


# REMOVED this block - Gunicorn will run the app object directly
# if __name__ == '__main__':
#     # This block is only for running locally with `python app.py`
#     # Set debug=False to better simulate production environment if running locally
#     print(f"--- Starting Flask development server on http://0.0.0.0:{PORT} ---")
#     print("--- This mode is NOT for production deployment on Render ---")
#     app.run(host='0.0.0.0', port=PORT, debug=False)
