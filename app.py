from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from model import AlexNet  # your custom model

app = Flask(__name__)

# ✅ Allow all origins temporarily (for testing)
# Once it's working, you can limit it again to your frontend domain.
CORS(app)

# ✅ Lazy-load model to avoid high startup memory usage
loaded_model = None

def get_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = AlexNet()
        model_path = "AlexNet.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        loaded_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        loaded_model.eval()
        print("✅ Model loaded successfully and ready for predictions!")
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
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

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
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ✅ Render dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns dynamic ports
    app.run(host="0.0.0.0", port=port)
