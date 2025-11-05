from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from model import AlexNet  # Your custom model

app = Flask(__name__)

# ✅ Allow your Vercel frontend and local testing
CORS(app, resources={
    r"/*": {"origins": [
        "https://minor-cyan.vercel.app",
        "http://localhost:3000"
    ]}
})

# ✅ Lazy-load model to avoid high startup memory
loaded_model = None

def get_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = AlexNet()
        loaded_model.load_state_dict(torch.load("AlexNet.pt", map_location="cpu"))
        loaded_model.eval()
        print("✅ Model loaded successfully!")
    return loaded_model

# ✅ Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

labels = ['NORMAL', 'PNEUMONIA']

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Backend is running and ready for predictions!"})

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        model = get_model()

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()

        confidence = round(probabilities[prediction].item() * 100, 2)

        return jsonify({
            "prediction": labels[prediction],
            "confidence": f"{confidence}%",
            "probabilities": {
                labels[0]: f"{round(probabilities[0].item() * 100, 2)}%",
                labels[1]: f"{round(probabilities[1].item() * 100, 2)}%"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Use dynamic port (important for deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 = default for Hugging Face
    app.run(host="0.0.0.0", port=port)
