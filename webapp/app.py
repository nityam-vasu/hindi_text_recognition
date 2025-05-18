import os
import json
import shutil
import atexit
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import re

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_ROOT = "essential/models"
LOG_DIR = 'static/logs'
LOG_FILE = os.path.join(LOG_DIR, 'predictions.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default model names for each recognition type
DEFAULT_MODEL_MAPPING = {
    'hwt': 'HWT_recognition_model.pth',
    'text': 'Text_recognition_model.pth',
    # 'hcr': 'HCR_recognition_model.pt'
}

# Character list paths based on recognition type
CHARLIST_MAPPING = {
    'hwt': 'essential/charlist.txt',
    'text': 'essential/charlist.txt',
    # 'hcr': 'essential/References.npy'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_most_recent_model(recognition_type):
    """
    Find the most recent model file for the given recognition type based on the date in the filename.
    If no dated model is found, fall back to the default model.
    """
    base_model_name = DEFAULT_MODEL_MAPPING[recognition_type].rsplit('.', 1)[0]  # e.g., 'HWT_recognition_model'
    model_dir = MODEL_ROOT
    date_pattern = re.compile(rf"{base_model_name}_(\d{{8}})\.pth$")  # e.g., HWT_recognition_model_20250518.pth
    
    today = datetime.strptime("20250518", "%Y%m%d").date()  # Today's date: May 18, 2025
    most_recent_model = DEFAULT_MODEL_MAPPING[recognition_type]  # Default fallback
    most_recent_date = None

    # Scan the model directory for matching files
    for filename in os.listdir(model_dir):
        match = date_pattern.match(filename)
        if match:
            date_str = match.group(1)  # Extract the date (e.g., 20250518)
            try:
                model_date = datetime.strptime(date_str, "%Y%m%d").date()
                # Calculate the difference from today
                if most_recent_date is None or (today - model_date).days < (today - most_recent_date).days:
                    most_recent_date = model_date
                    most_recent_model = filename
            except ValueError:
                continue

    print(f"Selected model for {recognition_type}: {most_recent_model}")
    return most_recent_model

def load_log():
    log = []
    if not os.path.exists(LOG_FILE):
        print(f"Log file does not exist at {LOG_FILE}, creating an empty one")
        # Create an empty log file if it doesn't exist
        try:
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error creating log file: {str(e)}")
        return log
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            try:
                log = json.load(f)
                print(f"Loaded log entries: {log}")
            except json.JSONDecodeError:
                print("Log file is empty or invalid JSON, resetting to empty list")
                return []
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return []
    return log

def save_log(model, image_path, prediction, is_correct="Not Set"):
    """Saves a new log entry."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = {
        'model': model,
        'image_path': image_path,
        'prediction': prediction,
        'is_correct': is_correct,
        'timestamp': timestamp
    }
    print(f"Saving log entry: {row}")
    log = load_log()
    print(f"Current log before append: {log}")
    log.append(row)
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=4)
        print(f"Successfully wrote log entry to {LOG_FILE}")
    except Exception as e:
        print(f"Error saving log entry: {str(e)}")
        raise  # Re-raise the exception to catch it in the calling function

def update_log(image_path, is_correct):
    """Updates the is_correct field of an existing log entry."""
    log = load_log()
    updated = False
    for entry in log:
        if 'image_path' not in entry:
            print(f"Warning: Log entry missing 'image_path' in update_log: {entry}")
            continue
        if entry['image_path'] == image_path:
            entry['is_correct'] = str(is_correct).lower()
            updated = True
            print(f"Updated log entry for {image_path}: is_correct = {entry['is_correct']}")
            break
    if updated:
        try:
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(log, f, indent=4)
            print(f"Successfully wrote updated log to {LOG_FILE}")
        except Exception as e:
            print(f"Error updating log entry: {str(e)}")
    else:
        print(f"Warning: No log entry found for image_path '{image_path}' in update_log")

@app.route('/')
def index():
    recognition_type = request.args.get('recognition_type', 'hwt')
    log = load_log()
    return render_template('index.html', prediction_pairs=[], log=log, recognition_type=recognition_type)

@app.route('/predict', methods=['POST'])
def predict_route():
    log = load_log()
    prediction_pairs = []
    error = None
    recognition_type = request.form.get('recognition_type', 'hwt')

    if 'image' not in request.files:
        return render_template('index.html', error="No file part", log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    files = request.files.getlist('image')

    if not files or not any(file.filename for file in files):
        return render_template('index.html', error="No selected file", log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    if not recognition_type:
        return render_template('index.html', error="No recognition type selected", log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    if recognition_type not in DEFAULT_MODEL_MAPPING:
        return render_template('index.html', error="Invalid recognition type", log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    # Find the most recent model for the recognition type
    model_name = find_most_recent_model(recognition_type)

    if recognition_type == 'hwt':
        import recognition_hwt as recognition
        charlist_file = CHARLIST_MAPPING['hwt']
        char_to_idx, idx_to_char = recognition.load_charlist(charlist_file)
        num_classes = len(char_to_idx) + 1
        model_path = os.path.join(MODEL_ROOT, model_name)
        model = recognition.load_model(model_path, num_classes, DEVICE)
        transform = recognition.get_transform()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)

                try:
                    prediction = recognition.predict(image_path, model, idx_to_char, transform, DEVICE)
                    prediction_pairs.append((prediction, image_path))
                    save_log(model_name, image_path, prediction)
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    return render_template('index.html', error=error, log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    if recognition_type == 'text':
        import recognition_text as recognition
        charlist_file = CHARLIST_MAPPING['text']
        char_to_idx, idx_to_char = recognition.load_charlist(charlist_file)
        num_classes = len(char_to_idx) + 1
        model_path = os.path.join(MODEL_ROOT, model_name)
        model = recognition.load_model(model_path, num_classes, DEVICE)
        transform = recognition.get_transform()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)

                try:
                    prediction = recognition.predict(image_path, model, idx_to_char, transform, DEVICE)
                    prediction_pairs.append((prediction, image_path))
                    save_log(model_name, image_path, prediction)
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    return render_template('index.html', error=error, log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    if recognition_type == 'hcr':
        import recognition_hcr as recognition
        ref_file = CHARLIST_MAPPING['hcr']
        ref = recognition.load_reference(ref_file)
        num_classes = len(ref)
        model_path = os.path.join(MODEL_ROOT, model_name)
        model = recognition.load_model(model_path, num_classes, DEVICE)
        transform = recognition.get_transform()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)

                try:
                    prediction = recognition.predict(image_path, model, transform, DEVICE, ref)
                    prediction_pairs.append((prediction, image_path))
                    save_log(model_name, image_path, prediction)
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    return render_template('index.html', error=error, log=log, prediction_pairs=prediction_pairs, recognition_type=recognition_type)

    return render_template('index.html', prediction_pairs=prediction_pairs, log=load_log(), recognition_type=recognition_type, error=error)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    image_path = request.form.get('image_path')
    is_correct = request.form.get('is_correct', 'off') == 'on'
    prediction_pairs = request.form.get('prediction_pairs', [])
    recognition_type = request.form.get('recognition_type', 'hwt')

    # Debug: Print form data
    print(f"Received form data: image_path={image_path}, is_correct={is_correct}, recognition_type={recognition_type}")

    # Update the log with the feedback
    if image_path:
        update_log(image_path, is_correct)

    # Parse prediction_pairs and remove the submitted image
    if prediction_pairs:
        prediction_pairs = eval(prediction_pairs) if isinstance(prediction_pairs, str) else prediction_pairs
        prediction_pairs = [(pred, path) for pred, path in prediction_pairs if path != image_path]

    log = load_log()
    return render_template('index.html', prediction_pairs=prediction_pairs, log=log, recognition_type=recognition_type)

@app.route('/clear_log', methods=['POST'])
def clear_log_route():
    recognition_type = request.form.get('recognition_type', 'hwt')
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
    except FileNotFoundError:
        print(f"Log file not found at {LOG_FILE}, nothing to clear")
    except Exception as e:
        print(f"Error clearing log file: {e}")
    return render_template('index.html', prediction_pairs=[], log=load_log(), recognition_type=recognition_type)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/version')
def version():
    return render_template('version.html')

def cleanup_upload_folder():
    """Removes all files from the upload folder."""
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print(f"Cleaned up upload folder: {folder}")

atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)