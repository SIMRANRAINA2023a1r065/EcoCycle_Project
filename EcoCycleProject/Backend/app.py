import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "ecocycle_secret_crypto_key_jammu"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load custom trained .h5 model file
MODEL_PATH = 'mobilenetv2_ecocycle.h5'

if os.path.exists(MODEL_PATH):
    print(f"Success: Loading custom trained model layers from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"CRITICAL ERROR: Place your '{MODEL_PATH}' file in the project folder!")

CATEGORIES = ["Recycle", "Repair", "Reuse"]

# Volatile In-Memory User Storage with Pre-Seeded Team Accounts
USER_DB = {
    "Mahi Sharma": "mahi123",
    "Payal Sharma": "payal123",
    "Simran Raina": "simran123"
}

# Session tracker for scanned item analytics
if 'analytics' not in globals():
    ANALYTICS_TRACKER = {"Recycle": 0, "Reuse": 0, "Repair": 0}

def get_general_pillar_statement(category_prediction):
    if category_prediction == "Recycle":
        return {
            "title": "Recycle Category Verified",
            "reason": "The model's dense layer activation identified raw material characteristics matching recyclable targets. Processing this item prevents valuable polymers or metals from entering regional landfills.",
            "action": "Route to Authorized Secondary Processing Node (Gangyal Industrial Area)",
            "value": "Market Scrap Value Applied"
        }
    elif category_prediction == "Reuse":
        return {
            "title": "Reuse Category Verified",
            "reason": "The model's dense layer activation identified high structural permanence and material stability. The physical form factor is intact, making it an ideal asset to extend its lifecycle without energy-intensive melting.",
            "action": "Upload Asset to the Live EcoCycle Peer-to-Peer Marketplace Hub",
            "value": "User Listed / Negotiable Price"
        }
    elif category_prediction == "Repair":
        return {
            "title": "Repair Category Verified",
            "reason": "The model's dense layer activation detected a complex material assembly with localized structural wear. The lifecycle can be fully restored through component adjustment or mending.",
            "action": "Connect with Certified Local Service Artisans (Gandhi Nagar / Janipur Hubs)",
            "value": "Estimated Local Repair Cost Applied"
        }
    return {
        "title": "Unclassified Material Asset",
        "reason": "Tensor probability margins fall below the optimized confidence thresholds for the core EcoCycle pillars.",
        "action": "Manual Sort Required",
        "value": "—"
    }

JAMMU_MARKETPLACE = [
    {"id": 1, "item": "Premium Glass Containers / Jars", "category": "Reuse", "price": "₹45 per unit", "shop": "Verma Glass House & Scrap Traders", "location": "Digiana, Jammu", "pincode": "180010"},
    {"id": 2, "item": "Industrial Grade PET Plastic Bottles", "category": "Recycle", "price": "₹12 per kg", "shop": "Trikuta Waste Management", "location": "Gangyal Industrial Area, Jammu", "pincode": "180010"},
    {"id": 3, "item": "Electronic & Furniture Salvage Hub", "category": "Repair", "price": "Fix: ₹150 - ₹500", "shop": "Janipur Repair Hub", "location": "Main Bazaar, Janipur, Jammu", "pincode": "180007"}
]

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/api/register', methods=['POST'])
def handle_registration():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({"error": "Fields cannot be empty!"}), 400
    if username in USER_DB:
        return jsonify({"error": "User already exists! Try logging in."}), 400
        
    USER_DB[username] = password
    return jsonify({"success": "Registration successful! You can now log in."})

@app.route('/api/login', methods=['POST'])
def handle_login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if username in USER_DB and USER_DB[username] == password:
        session['user'] = username
        return jsonify({"success": True, "redirect": url_for('dashboard_page', username=username)})
    
    return jsonify({"error": "Invalid username or matching password credentials!"}), 401

@app.route('/dashboard')
def dashboard_page():
    images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('dashboard.html', database_images=images)

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    return jsonify(ANALYTICS_TRACKER)

@app.route('/api/marketplace', methods=['GET'])
def get_marketplace(): 
    return jsonify({"items": JAMMU_MARKETPLACE})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "Empty filename"}), 400

    mode = request.form.get('mode', 'single')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None: return jsonify({"error": "Invalid image format"}), 400

    img_h, img_w, _ = img.shape
    detected_objects = []

    def run_custom_inference(image_matrix):
        resized = cv2.resize(image_matrix, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        normalized_inputs = (img_array / 127.5) - 1.0
        
        predictions = model.predict(normalized_inputs)
        best_index = int(np.argmax(predictions[0]))
        return CATEGORIES[best_index], float(predictions[0][best_index])

    if mode == 'single':
        pred_class, confidence = run_custom_inference(img)
        ANALYTICS_TRACKER[pred_class] += 1
        statement = get_general_pillar_statement(pred_class)
        
        detected_objects.append({
            "label": statement["title"],
            "confidence": f"{round(confidence * 100, 1)}%",
            "reason": statement["reason"],
            "action": statement["action"],
            "est_value": statement["value"],
            "box": None
        })
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = img_h * img_w
        valid_contours = [c for c in contours if (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]) > (img_area * 0.015)]

        if len(valid_contours) == 0:
            pred_class, confidence = run_custom_inference(img)
            ANALYTICS_TRACKER[pred_class] += 1
            statement = get_general_pillar_statement(pred_class)
            detected_objects.append({
                "label": f"Primary Focus: {statement['title']}",
                "confidence": f"{round(confidence * 100, 1)}%",
                "reason": statement["reason"],
                "action": statement["action"],
                "est_value": statement["value"],
                "box": {"x": 0, "y": 0, "w": img_w, "h": img_h}
            })
        else:
            for i, ctr in enumerate(valid_contours):
                x, y, w, h = cv2.boundingRect(ctr)
                crop = img[y:y+h, x:x+w]
                if crop.shape[0] < 15 or crop.shape[1] < 15: continue
                
                pred_class, confidence = run_custom_inference(crop)
                ANALYTICS_TRACKER[pred_class] += 1
                statement = get_general_pillar_statement(pred_class)

                detected_objects.append({
                    "label": f"Object #{i+1}: {statement['title']}",
                    "confidence": f"{round(confidence * 100, 1)}%",
                    "reason": statement["reason"],
                    "action": statement["action"],
                    "est_value": statement["value"],
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                })

    return jsonify({
        "objects": detected_objects, 
        "filename": file.filename,
        "width": int(img_w),
        "height": int(img_h)
    })

@app.route('/benefits')
def benefits_page():
    return render_template('benefits.html')

@app.route('/logout')
def handle_logout():
    session.clear()
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)