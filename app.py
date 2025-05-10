from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.nn.functional import softmax
from PIL import Image
import os
import datetime

app = Flask(__name__)

# ========== Configuration ==========
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
history = []

# ========== Device Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Model Once ==========
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Real vs Counterfeit
model.load_state_dict(torch.load('resnet18_counterfeit.pth', map_location=device))
model.to(device)
model.eval()

# ========== Define Transform ==========
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== Routes ==========

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess Image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        label = 'Real Money' if predicted.item() == 1 else 'Counterfeit Money'
        confidence_score = confidence.item() * 100
        timestamp = datetime.datetime.now().strftime('%I:%M %p %m/%d/%Y')

        history.insert(0, {
            'filename': filename,
            'label': label,
            'confidence': round(confidence_score, 2),
            'timestamp': timestamp
        })

        return render_template('index.html', filename=filename, label=label, confidence=confidence_score)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history_page():
    return render_template('history.html', history=history)

@app.route('/about')
def about():
    return render_template('about.html')

# ========== Run Server ==========
if __name__ == '__main__':
    app.run(debug=True)


