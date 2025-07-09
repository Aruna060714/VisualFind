from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os
import time
from urllib.parse import urlparse
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from opensearchpy import OpenSearch
from dotenv import load_dotenv
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
load_dotenv()
BONSAI_URL = os.getenv("BONSAI_URL")
BONSAI_USER = os.getenv("BONSAI_USER")
BONSAI_PASS = os.getenv("BONSAI_PASS")
parsed = urlparse(BONSAI_URL)
client = OpenSearch(
    hosts=[{"host": parsed.hostname, "port": parsed.port}],
    http_auth=(BONSAI_USER, BONSAI_PASS),
    use_ssl=True
)
model = MobileNet(weights="imagenet")
def detect_type(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    raw_label = decode_predictions(preds, top=1)[0][0][1].lower()
    print(f"🔍 Predicted type: {raw_label}")
    return raw_label
def search_similar_products(product_type, top_k=5):
    query = {
        "size": top_k,
        "query": {
            "match": {
                "type": product_type
            }
        }
    }
    response = client.search(index="products_ai", body=query)
    hits = response["hits"]["hits"]
    if not hits:
        print(f" No matches found for type: {product_type}")
    return [hit["_source"] for hit in hits]
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['image']
    if file:
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_type = detect_type(filepath)
        results = search_similar_products(predicted_type)
        return render_template("results.html", results=results, category=predicted_type)
    return "No file uploaded."
if __name__ == "__main__":
    app.run(debug=True)