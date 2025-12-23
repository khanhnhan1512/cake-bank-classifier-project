Data Link:

https://drive.google.com/file/d/1Y_6TfutXPg42HwMCZBlrjSZSxy0vEpC2/view

Section 1: Building a Liveness Classifier (40 points)

In face verification services such as eKYC and access control, ensuring that a presented face is from a live person rather than a spoofed artifact (e.g., printed photos, digital screens, masks, or deepfake videos) is a critical challenge. This problem, known as liveness detection, aims to distinguish between real and fake facial inputs to enhance security and prevent fraud.

Task

Given a dataset containing images labeled as "normal" or "spoof," develop an AI model to classify them accurately.

A liveness detection model processes an image and outputs a liveness score in the range [0,1], where 0 represents a real (live) face and 1 indicates a spoofed (fake) face.

Requirements:

Load and preprocess the dataset. Choose an appropriate model architecture. Train and evaluate the classifier (on test set). Provide code and rationale for your choices.

--

Section 2: Report on Model Performance (30 points)

Task: Write a brief report covering the following aspects:

Metrics chosen for evaluation (e.g., accuracy, precision, recall, F1-score, ROC-AUC).

Justification of the chosen evaluation approach.

Potential limitations of the model and areas for improvement.

Propose alternative techniques that might improve the classifier.

--

Section 3: Designing a Robust Fraud Detection System (30 points)

Task:

Liveness detection is just one component of a broader fraud detection system. If given the opportunity to design a robust fraud detection system, propose a solution considering multiple layers of defense.

Guidelines:

Outline a multi-layered approach incorporating liveness detection.

Consider additional security measures (e.g., behavioral analysis, biometric verification, anomaly detection).

Discuss potential challenges and how to mitigate them. Present a high-level architecture diagram (if applicable).

--

Submission Guidelines: Provide code in a Jupyter Notebook or a Python script. Submit a PDF report for Sections 2 & 3. Ensure code is well-documented and readable. Include necessary dependencies and instructions for running the code.
