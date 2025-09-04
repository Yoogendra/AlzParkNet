# 🧬 AlzParkNet: Protein Sequence Classification with CNN

AlzParkNet is a **deep learning model** built with TensorFlow/Keras to classify protein sequences into four categories:

- 🧠 Parkinson's Disease
- 🧩 Alzheimer's Disease
- 🔬 Other Neurodegenerative Disorders
- ✅ Healthy

---

## 📂 Project Structure

protein-cnn/
│── alzparknet_cnn.py  
│── final_seq_data.csv  
│── requirements.txt  
│── README.md

---

## ⚙️ Installation & Setup

```bash
# 1. Clone Repository
git clone https://github.com/yourusername/AlzParkNet.git
cd AlzParkNet

# 2. Create Virtual Environment (Recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. (macOS M1/M2 only) Install TensorFlow with Metal support
pip install tensorflow-macos==2.12.0 tensorflow-metal==0.8.0

# 5. (Optional, for Jupyter Notebook users) Link environment kernel
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"


⸻

🚀 Usage

# Train the model
python alzparknet_cnn.py

# During run, you’ll be prompted to enter a protein sequence
Enter protein sequence: ACDEFGHIKLMNPQRSTVWY
Prediction: Parkinson's Disease

The script will also generate:
	•	✅ Confusion Matrix
	•	📈 Training vs Validation Accuracy curve
	•	📉 Training vs Validation Loss curve
	•	📊 F1-Score per class (bar chart)
	•	🧪 Classification Report
	•	🔍 Confidence vs Accuracy scatter plot

⸻

🏗️ Model Architecture
	•	Input: Encoded protein sequence (200 max length)
	•	Layers:
	•	Embedding layer
	•	Conv1D + MaxPooling + Dropout
	•	Conv1D + MaxPooling
	•	Conv1D + GlobalMaxPooling + Dropout
	•	Dense + Dropout
	•	Output Dense (Softmax, 4 classes)
	•	Optimizer: Adam (lr = 0.0005)
	•	Loss: Categorical Crossentropy
	•	Metrics: Accuracy, F1-score

⸻

🛠️ Future Work
	•	🔄 Add Transformer-based protein models (ProtBERT, ESM)
	•	🧬 Experiment with LSTM/GRU sequence encoders
	•	🧪 Data augmentation & larger datasets
	•	☁️ Deploy as a web app (Streamlit/Flask)

⸻

📜 License

MIT License — free to use, modify, and distribute with attribution.
```
