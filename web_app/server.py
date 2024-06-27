import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve index.html from the 'web_app' folder
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Serve static files (CSS, JS, images, etc.) from 'web_app/static' folder
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'web_app', 'static'), filename)

# Load the model and define transformations
model_path = os.path.join(app.root_path, 'Chess_Pieces.pt')
labels = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
model = models.vgg16_bn()
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=len(labels), bias=True)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Endpoint for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                img = Image.open(file)
                img = transform(img).unsqueeze(0)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(img)
                    _, predicted = torch.max(outputs, 1)
                    class_idx = predicted.item()
                    accuracy = float(torch.softmax(outputs, dim=1)[0][predicted.item()] * 100)
                    
                    result = {
                        'class': labels[class_idx],
                        'accuracy': accuracy
                    }
                    return jsonify(result)
            except Exception as e:
                return str(e), 500
        return 'File not received', 400

if __name__ == '__main__':
    app.run(port=3000, debug=True)
