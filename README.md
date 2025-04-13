#LLM Compression via Knowledge Distillation and Quantization 
##Overview
This project demonstrates how to compress large language models (LLMs) using a combination of Knowledge Distillation and Quantization, reducing model size and computational cost while maintaining (or even improving) performance. A phishing website classifier is built and optimized through these techniques using PyTorch and Hugging Face Transformers.

The goal is to make powerful LLMs accessible for resource-constrained environments like laptops or mobile devices, ensuring greater efficiency, lower financial and environmental costs, and on-device privacy.

##Tools & Frameworks Used
Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets & Hub

Google Colab (GPU support)

Scikit-learn (for evaluation metrics)

Bitsandbytes (for model quantization)

##Steps Followed
###1. Dataset Preparation
A binary classification dataset containing URLs and phishing labels is loaded from Hugging Face.

The dataset is split into training, testing, and validation subsets.

Text is tokenized and converted into PyTorch tensors.

###2. Teacher Model Loading
A large BERT-based model fine-tuned for phishing classification is loaded as the teacher.

The model is deployed to GPU for inference.

###3. Student Model Creation
A smaller version of DistilBERT is created by reducing attention heads and layers.

A classification head is attached to enable phishing detection.

###4. Knowledge Distillation
The student model learns from both the soft targets (logits) of the teacher and the true labels.

A custom loss function combines KL divergence (distillation loss) and cross-entropy (hard loss).

###5. Training & Evaluation
Both models are evaluated using accuracy, precision, recall, and F1 score.

The student model eventually outperforms the teacher on all metrics.

###6. Quantization
The student model is quantized to 4-bit precision using bitsandbytes.

Quantization configuration includes:

nf4 (normal float) precision

Brain floating point 16 compute

Double quantization

Post-quantization, the model size drops significantly (from 211MB to ~62MB) with improved performance.

##Advantages of the Project
7x reduction in model size without sacrificing accuracy

Faster inference and lower memory usage

Enables on-device execution for privacy and accessibility

Demonstrates practical use of multi-step model compression in real-world NLP applications

Student model outperforms teacher due to reduced overfitting and redundant complexity
