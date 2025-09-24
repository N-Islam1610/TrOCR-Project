# TrOCR-Project

# TrOCR Fine-Tuning for Curved Text Recognition
🔍 Project Overview

This project fine-tunes Microsoft’s TrOCR (Transformer-based OCR) model on the SCUT-CTW1500 dataset for recognizing curved text in natural images. TrOCR combines a Vision Transformer (ViT) encoder with a Transformer-based decoder, enabling end-to-end scene text recognition.

The notebook includes:

Dataset preparation and preprocessing.

Model loading and fine-tuning using HuggingFace’s transformers.

Evaluation using Character Error Rate (CER).

Saving and reloading the trained model for inference.

# 🚀 Features

End-to-end training pipeline for OCR on curved text.

Custom dataset class (CustomOCRDataset) for handling image-text pairs.

Data collator for batching OCR data.

Evaluation metric: Character Error Rate (CER).

HuggingFace Seq2SeqTrainer integration for training.

# 📂 Dataset

Dataset Used: SCUT-CTW1500

Format:

Images containing curved text.

Labels provided in .txt files.

Data is split into training and validation sets using train_test_split.

⚙️ Installation & Requirements

Install the required dependencies:

pip install torch torchvision transformers datasets editdistance scikit-learn Pillow

# 🧑‍💻 Usage
1. Clone Repository & Prepare Data
git clone https://github.com/your-username/trocr-curved-text.git
cd trocr-curved-text


Download and unzip the SCUT dataset into work/scut_data/.

2. Run Training Notebook

Open the Jupyter notebook and execute cells to:

Load and preprocess data.

Fine-tune TrOCR.

Save the model.

3. Evaluate Model

The notebook computes Character Error Rate (CER):

from editdistance import eval as editdistance_eval

# 📊 Results

Evaluation Metric: Character Error Rate (CER).

Lower CER indicates better recognition performance.

The fine-tuned model achieves strong performance on curved text samples (results vary depending on training epochs, batch size, and learning rate).

# 🔮 Future Improvements

Train with larger batch sizes on GPUs/TPUs for faster convergence.

Experiment with TrOCR large models.

Apply data augmentation (rotation, noise, distortion).

Extend to multi-language OCR.
