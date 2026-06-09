import React, { useState } from "react";

export default function ImageUpload({ setPrediction, setConfidence, setRecommendation }) {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setImage(URL.createObjectURL(selected));
      setFile(selected);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      
      let res = await fetch("https://plantdocbot-1-wo2l.onrender.com/predict_image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Server response mein error hai");
      }

      let data = await res.json();
      setPrediction(data.predicted_class);
      setConfidence(data.confidence);
      setRecommendation(data.recommendation);
    } catch (error) {
      console.error("Prediction fail ho gayi:", error);
      setPrediction("Error connecting to server");
      setConfidence("0");
      setRecommendation("Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>Upload Leaf Image</h3>

      <input type="file" accept="image/*" onChange={handleFile} />

      {image && (
        <img src={image} alt="preview" className="preview-img" />
      )}

      <button className="btn" disabled={!file || loading} onClick={handlePredict}>
        {loading ? "Predicting..." : "Predict Disease"}
      </button>
    </div>
  );
}