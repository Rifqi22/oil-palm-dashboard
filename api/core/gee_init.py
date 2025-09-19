import ee
import os
import json
import tempfile
from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

print("EE_KEY_JSON:", os.environ.get("EE_KEY_JSON")[:50], "...")  # debug


EE_ACCOUNT = "oil-palm-gee@gee-oil-palm-dashboard.iam.gserviceaccount.com"

def init_gee():
    key_json_str = os.environ.get("EE_KEY_JSON")
    if not key_json_str:
        raise ValueError("Environment variable EE_KEY_JSON is not set!")

    key_json_str = key_json_str.strip().strip("'").strip('"')

    # Parse JSON
    key_dict = json.loads(key_json_str)

    # Write the JSON string to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        f.write(json.dumps(key_dict))
        f.flush()
        key_file_path = f.name

    credentials = ee.ServiceAccountCredentials(
        EE_ACCOUNT,
        key_file=key_file_path
    )

    ee.Initialize(credentials)
    print("Earth Engine initialized in serverless mode")
