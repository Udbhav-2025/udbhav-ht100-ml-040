---
title: Mental Health Risk Assessment
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Mental Health Risk Assessment

A multi-modal AI system for mental health risk assessment using text, images, and object detection.

## Features

- **Multi-Modal Analysis**: Combines text sentiment, visual emotion recognition, and harmful object detection
- **Adaptive Processing**: Works with text-only, image-only, or combined inputs
- **89.6% F1-Score**: Trained on 52,000+ clinical scenarios
- **Clinical Support Tool**: Supplementary screening tool for healthcare professionals

## Model Architecture

- **Text Analysis**: Fine-tuned BERT (768-dim embeddings)
- **Image Analysis**: EfficientNet-B0 (1280-dim) + Emotion Detection
- **Object Detection**: 18 harmful objects with risk scoring
- **Fusion Model**: Neural network combining all modalities

## Disclaimer

This tool is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is in crisis, please contact emergency services or call 988 (Suicide & Crisis Lifeline).
