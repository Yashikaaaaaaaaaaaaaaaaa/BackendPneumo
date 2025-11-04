from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
from model import AlexNet  # Your model class

app = Flask(__name__)

# ✅ Allow both localhost (for testing) and your deployed Vercel frontend
CORS(app, resources={
    r"/*": {"origins": [
        "https://minor-cyan.vercel.app",
        "http://localhost:3000"
    ]}
})

# ✅ Load model
loaded_model = AlexNet()
loaded_model.load_state_dict(torch.load("./results/AlexNet.pt", map_location="cpu"))
loaded_model.eval()
print("✅ Model loaded successfully and ready for predictions!")

# ✅ Preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

labels = ['NORMAL', 'PNEUMONIA']

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Backend is running and ready for predictions!"})

# ✅ Prediction route with confidence
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = loaded_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]  # Get confidence values
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
