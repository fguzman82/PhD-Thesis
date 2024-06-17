# PhD Thesis: Methodologies for Concept Identification and Recovery in Convolutional Neural Networks

## Overview
This repository contains the code and documentation related to my PhD thesis titled "Methodologies for Concept Identification and Recovery in Convolutional Neural Networks (CNNs)." The work focuses on enhancing the interpretability and reliability of CNNs through the development of novel methodologies for concept attribution and recovery.

## Motivation
The rapid advancement of CNN architectures, such as AlexNet, VGG16, GoogLeNet, ResNet50, and Florence, has led to impressive performance in various image classification tasks. However, these models often act as "black boxes," making their decision-making processes opaque and raising concerns about their reliability and fairness.

## Objectives
1. **Concept Attribution**: Develop techniques to identify and attribute concepts within CNNs, improving the quality and robustness of explanations.
2. **Concept Recovery**: Design methodologies to recover original concepts from CNNs affected by adversarial attacks or noise, thereby enhancing prediction reliability.

## Methodologies
### Concept Identification
- **Techniques**: Implement and evaluate various attribution methods, such as LIME, SHAP, Grad-CAM, and Meaningful Perturbation (MP).
- **Fidelity and Robustness**: Assess the fidelity and robustness of these techniques using metrics like Intersection over Union (IoU) and Structural Similarity Index Measure (SSIM).

### Concept Recovery
- **Adversarial Robustness**: Utilize adversarial training techniques to improve the model's resilience against adversarial samples.
- **Optimization Algorithms**: Develop algorithms that generate masks to identify and recover key attributes, restoring the model's original predictions.

## Implementation
- **Dataset**: Experiments are conducted on standard datasets such as ImageNet.
- **CNN Architectures**: Various CNN models, including ResNeXt and FixResNeXt-101, are used to validate the methodologies.
- **Metrics**: Evaluation metrics include IoU, SSIM, Pearson HOG, and deletion scores.

## Results
The proposed methodologies demonstrate significant improvements in both fidelity and robustness compared to baseline models. The developed techniques provide high-quality explanations and effective recovery of original predictions, enhancing the transparency and reliability of CNN-based systems.

## Conclusion
This work contributes to the field of AI interpretability by offering robust and reliable methodologies for concept identification and recovery in CNNs. Future research directions include exploring concept-based transfer learning and network compression guided by interpretability.

## Repository Structure
- `code/`: Contains the implementation of the methodologies.
- `data/`: Includes sample datasets used for testing and evaluation.
- `results/`: Stores the results and evaluation metrics of the experiments.
- `docs/`: Documentation and additional resources related to the thesis.

## References


## Contact
For any inquiries or collaborations, please contact Fabio Guzmán at fabioandres.guzman@gmail.com

---

Feel free to explore the repository and contribute to this ongoing research.
