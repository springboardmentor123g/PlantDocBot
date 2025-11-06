import { useState } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [imageResult, setImageResult] = useState("");
  const [textInput, setTextInput] = useState("");
  const [textResult, setTextResult] = useState("");

  const BASE_URL = "http://localhost:8000";

  const api = axios.create({
    baseURL: BASE_URL,
  });

  // ✅ TEXT Prediction
  const handleTextPrediction = async () => {
    if (textInput.trim() === "") {
      setTextResult("⚠️ Please enter symptoms!");
      return;
    }

    setTextResult("🔍 Analyzing...");
    try {
      const response = await api.post(
        "/text-prediction",
        { input: textInput },
        { headers: { "Content-Type": "application/json" } }
      );

      const data = response.data;
      setTextResult(
        `✅ ${data.predicted_disease}\nConfidence: ${data.confidence}\n💡 Advice: ${data.recommendation}`
      );
    } catch (error) {
      console.error("Error:", error);
      setTextResult("❌ Error predicting from text.");
    }
  };

  // ✅ IMAGE Upload Handler
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
    }
  };

  // ✅ IMAGE Prediction
  const predictImage = async () => {
    if (!image) {
      setImageResult("⚠️ Please upload an image first!");
      return;
    }

    setImageResult("🔍 Analyzing...");
    try {
      const fileInput = document.getElementById("fileInput");
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      const response = await api.post("/image-prediction", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;
      setImageResult(
        `✅ ${data.predicted_class}\nConfidence: ${data.confidence}\n💡 Advice: ${data.recommendation}`
      );
    } catch (error) {
      console.error(error);
      setImageResult("❌ Error predicting image.");
    }
  };

  return (
    <div className="app">
      <div className="background"></div>

      <motion.div
        className="container"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1 }}
      >
        <header>
          <h1>🌿 AI Plant Disease Detection</h1>
          <p>Instant analysis using image or text</p>
        </header>

        <div className="sections">
          {/* IMAGE SECTION */}
          <motion.div
            className="card"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 200 }}
          >
            <h2>🖼️ Image Detection</h2>
            <div
              className="upload-box"
              onClick={() => document.getElementById("fileInput").click()}
            >
              <input
                id="fileInput"
                type="file"
                accept="image/*"
                hidden
                onChange={handleImageUpload}
              />
              {image ? (
                <img src={image} alt="Preview" className="preview" />
              ) : (
                <p>📸 Click to upload</p>
              )}
            </div>
            <button className="btn" onClick={predictImage}>
              Predict
            </button>
            <motion.pre
              key={imageResult}
              className="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {imageResult}
            </motion.pre>
          </motion.div>

          {/* TEXT SECTION */}
          <motion.div
            className="card"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 200 }}
          >
            <h2>📝 Text Detection</h2>
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Describe symptoms..."
            ></textarea>
            <button className="btn" onClick={handleTextPrediction}>
              Predict
            </button>
            <motion.pre
              key={textResult}
              className="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {textResult}
            </motion.pre>
          </motion.div>
        </div>

        <footer>
          <br /><br />
          © 2025 Dhanush | AI Disease Detection 🌱
        </footer>
      </motion.div>
    </div>
  );
}

export default App;
