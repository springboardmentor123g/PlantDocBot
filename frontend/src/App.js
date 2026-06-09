import React, { useState } from "react";
import ImageUpload from "./components/ImageUpload"; // TextUpload ka import hata diya ✂️
import "./App.css";

export default function App() {
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [recommendation, setRecommendation] = useState("");

  return (
    <div className="app-container">
      <h1 className="title">Plant Disease Detector</h1>

      <div className="upload-section">
        {/* Sirf ImageUpload rakha hai, text box gayab! */}
        <ImageUpload
          setPrediction={setPrediction}
          setConfidence={setConfidence}
          setRecommendation={setRecommendation}
        />
      </div>

      <div className="result-box">
        <h3>Prediction Result</h3>
        <p><b>Disease:</b> {prediction || "No prediction yet"}</p>
        <p><b>Confidence:</b> {confidence ? `${confidence}%` : "0%"}</p>
        <p><b>Recommendation:</b> {recommendation || "Upload an image to see recommendations."}</p>
      </div>
    </div>
  );
}