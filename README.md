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

1. “AI Now Institute.” https://ainowinstitute.org/research.html (accessed May 03, 2019).
2. L. F. Barrett, R. Adolphs, S. Marsella, A. M. Martinez, and S. D. Pollak, “Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements,” Psychol. Sci. Public Interes., vol. 20, no. 1, pp. 1–68, Jul. 2019.
3. K. Crawford, “Halt the use of facial-recognition technology until it is regulated,” Nature, vol. 572, no. 7771, p. 565, Aug. 2019.
4. S. Emerson, “San Francisco Bans Facial Recognition Use by Police and the Government,” Vice Magazine, 2019. www.vice.com/en/article/wjvxxb/san-francisco-bans-facial-recognition-use-by-police-and-the-government.
5. C. Haskins, “Oakland Becomes Third U.S. City to Ban Facial Recognition,” Vice Magazine, 2019. www.vice.com/en/article/zmpaex/oakland-becomes-third-us-city-to-ban-facial-recognition-xz.
6. K. Lannan, “Somerville Bans Government Use Of Facial Recognition Tech.” www.wbur.org/bostonomix/2019/06/28/somerville-bans-government-use-of-facial-recognition-tech.
7. L. Kelion, “MPs call for halt to police’s use of live facial recognition,” BBC Newspaper, 2019. www.bbc.com/news/technology-49030595.
8. A. Kak, “REGULATING BIOMETRICS Global Approaches and Urgent Questions,” in AI Now Institute, 2020, pp. 1–112.
9. R. Cellan-Jones, “Voice technology firm under fire,” BBC News. http://news.bbc.co.uk/2/hi/technology/8163511.stm.
10. C. Newton, “Facebook is shutting down M, its personal assistant service that combined humans and AI,” The Verge Newspaper, 2018. www.theverge.com/2018/1/8/16856654/facebook-m-shutdown-bots-ai.
11. J. Sadowski, “Potemkin AI,” Real Life Magazine, 2018. https://reallifemag.com/potemkin-ai/.
12. A. Davies, “Self-Driving Cars Have a Secret Weapon: Remote Control,” Wired Magazine, 2018. www.wired.com/story/phantom-teleops/.
13. “Kiwibot.” www.kiwibot.com/ (accessed Jul. 10, 2019).
14. G. Ogbonyenitan, “Kiwibots are not fully Autonomous and are controlled by operators in Colombia,” Techidence Magazine, 2019. www.techidence.com/kiwibots-are-not-fully-autonomous-and-are-controlled-by-operators-in-colombia/.
15. D. Byler, “China’s hi-tech war on its Muslim minority,” The Guardian Newspaper, 2019. www.theguardian.com/news/2019/apr/11/china-hi-tech-war-on-muslim-minority-xinjiang-uighurs-surveillance-face-recognition.
16. “Michigan unemployment agency made 20,000 false fraud accusations – report,” The Guardian Newspaper, 2016. www.theguardian.com/us-news/2016/dec/18/michigan-unemployment-agency-fraud-accusations.
17. S. Masís, Interpretable Machine Learning with Python. Packit Publishing Ltd., 2021.
18. “ACM Conference on Fairness, Accountability, and Transparency (ACM FAccT).” https://facctconference.org/.
19. A. Abdul, J. Vermeulen, D. Wang, B. Y. Lim, and M. Kankanhalli, “Trends and Trajectories for Explainable, Accountable and Intelligible Systems: An HCI Research Agenda,” in Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, 2018, pp. 1–18, [Online]. Available: https://doi.org/10.1145/3173574.3174156.
20. J. M. Alonso, C. Castiello, and C. Mencar, “A Bibliometric Analysis of the Explainable Artificial Intelligence Research Field,” in International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems, 2018, pp. 3–15.
21. C. Rudin, “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead,” Nov. 2018, Accessed: Dec. 09, 2021. [Online]. Available: https://arxiv.org/abs/1811.10154.
22. C. M. Bishop, “Pattern Recognition and Machine Learning,” Springer, 2006.
23. D. Cohn, L. Atlas, and R. Ladner, “Improving Generalization with Active Learning,” Machine Learning, vol. 15, no. 2, pp. 201–221, 1994.
24. B. Settles, “Active Learning,” Synthesis Lectures on Artificial Intelligence and Machine Learning, vol. 4, no. 1, pp. 1–103, 2010.
25. M. C. L. de Ma. Oliveira, J. E. da Silva, R. C. A. Carvalho, and E. S. da Silva, “A Survey of Ensemble Learning,” J. Comput. Sci. Eng., vol. 11, no. 1, pp. 1–15, 2017.
26. T. G. Dietterich, “Ensemble Learning,” in Multiple Classifier Systems, Springer, 2000, pp. 1–15.
27. T. G. Dietterich, “Machine Learning Research: Four Current Directions,” AI Mag., vol. 18, no. 4, pp. 97–136, 1997.
28. K. R. Gabriel, “The Biplot Graphic Display of Matrices with Applications to Principal Component Analysis,” Biometrika, vol. 58, no. 3, pp. 453–467, 1971.
29. L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
30. T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785–794.
31. J. H. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine,” Ann. Stat., vol. 29, no. 5, pp. 1189–1232, 2001.
32. L. Breiman, “Stochastic Games, Partially Observable Markov Decision Processes and Machine Learning,” J. Am. Stat. Assoc., vol. 92, no. 440, pp. 598–603, 1997.
33. T. G. Dietterich, “Ensemble Methods in Machine Learning,” in Multiple Classifier Systems, Springer, 2000, pp. 1–15.
34. J. Wang, J. He, and Y. Li, “Analysis of the Performance of Decision Trees and Their Ensemble Methods,” Neural Process. Lett., vol. 46, no. 2, pp. 945–962, 2017.
35. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning Representations by Back-propagating Errors,” Nature, vol. 323, no. 6088, pp. 533–536, 1986.
36. F. K. Došilović, M. Brčić, and N. Hlupić, “Explainable Artificial Intelligence: A Survey,” in 2018 41st International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO), 2018, pp. 210–215.
37. A. Carrington, P. Fieguth, and H. Chen, “Measures of Model Interpretability for Model Selection,” in Machine Learning and Knowledge Extraction, 2018, pp. 329–349.
38. L. H. Gilpin, D. Bau, B. Z. Yuan, A. Bajwa, M. Specter, and L. Kagal, “Explaining Explanations: An Overview of Interpretability of Machine Learning,” May 2018, Accessed: May 01, 2021. [Online]. Available: http://arxiv.org/abs/1806.00069.
39. T. Miller, “Explanation in Artificial Intelligence: Insights from the Social Sciences,” Artif. Intell., vol. 267, pp. 1–38, 2019.
40. I. T. Jolliffe, “Principal Component Analysis,” Springer Series in Statistics, 2002.
41. A. G. S. P. Li, “Feature Selection for Machine Learning: An Overview,” in 2018 International Conference on Artificial Intelligence and Advanced Manufacturing, 2018, pp. 137–142.
42. Y. LeCun, Y. Bengio, and G. Hinton, “Deep Learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015.
43. Y. Freund and R. E. Schapire, “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting,” in European Conference on Computational Learning Theory, Springer, 1995, pp. 23–37.
44. T. G. Dietterich, “Ensemble Methods in Machine Learning,” in Multiple Classifier Systems, Springer, 2000, pp. 1–15.
45. R. Caruana and A. Niculescu-Mizil, “An Empirical Comparison of Supervised Learning Algorithms,” in Proceedings of the 23rd International Conference on Machine Learning, 2006, pp. 161–168.
46. D. J. Cohn, L. A. Brewer, and D. J. Beck, “Machine Learning in the Biological Sciences: A Survey,” Artif. Intell. Rev., vol. 18, no. 2, pp. 101–142, 2002.
47. L. Rokach, “Ensemble-Based Classifiers,” Artif. Intell. Rev., vol. 33, no. 1, pp. 1–39, 2010.
48. L. K. Hansen and P. Salamon, “Neural Network Ensembles,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 12, no. 10, pp. 993–1001, 1990.
49. R. E. Schapire, “The Strength of Weak Learnability,” Mach. Learn., vol. 5, no. 2, pp. 197–227, 1990.
50. V. Vapnik, “Statistical Learning Theory,” Wiley, 1998.
51. J. D. Viviano, B. Simpson, F. Dutil, Y. Bengio, and J. P. Cohen, “Saliency is a Possible Red Herring When Diagnosing Poor Generalization,” Int. Conf. Learn. Represent., 2021, Accessed: Aug. 19, 2021. [Online]. Available: https://arxiv.org/abs/1910.00199.
52. J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, “Confounding variables can degrade generalization performance of radiological deep learning models,” PLoS Med., vol. 15, no. 11, Jul. 2018, doi: 10.1371/journal.pmed.1002683.
53. J. G. Moreno-Torres, T. Raeder, R. Alaiz-Rodríguez, N. V. Chawla, and F. Herrera, “A unifying view on dataset shift in classification,” Pattern Recognit., vol. 45, no. 1, pp. 521–530, Jan. 2012, doi: 10.1016/j.patcog.2011.06.019.
54. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.
55. R. Geirhos et al., “Shortcut learning in deep neural networks,” Nat. Mach. Intell., vol. 2, no. 11, pp. 665–673, Apr. 2020, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/2004.07780.
56. K. L. Hermann and A. K. Lampinen, “What shapes feature representations? Exploring datasets, architectures, and training,” in Advances in Neural Information Processing Systems, Jun. 2020, vol. 2020-Decem, Accessed: Sep. 06, 2021. [Online]. Available: https://arxiv.org/abs/2006.12433.
57. G. Parascandolo, A. Neitz, A. Orvieto, L. Gresele, and B. Schölkopf, “Learning explanations that are hard to vary,” Sep. 2020, Accessed: Sep. 06, 2021. [Online]. Available: https://arxiv.org/abs/2009.00329.
58. C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding deep learning (still) requires rethinking generalization,” Commun. ACM, vol. 64, no. 3, pp. 107–115, Nov. 2021, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1611.03530.
59. G. Hinton, “Stacked Capsule Autoencoders,” Keynotes AAAI, 2020.
60. C. Szegedy et al., “Intriguing properties of neural networks,” Dec. 2014, Accessed: Sep. 06, 2021. [Online]. Available: https://arxiv.org/abs/1312.6199.
61. I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” Dec. 2015, Accessed: May 01, 2021. [Online]. Available: https://arxiv.org/abs/1412.6572.
62. F. Lambert, “Understanding the fatal tesla accident on autopilot and the nhtsa probe,” Electrek, 2016.
63. C. Olah, A. Mordvintsev, and L. Schubert, “Feature visualization,” Distill, 2017.
64. A. Mahendran and A. Vedaldi, “Visualizing Deep Convolutional Neural Networks Using Natural Pre-images,” Int. J. Comput. Vis., vol. 120, no. 3, pp. 233–255, Dec. 2016, doi: 10.1007/s11263-016-0911-8.
65. W. Samek and K.-R. Müller, “Towards Explainable Artificial Intelligence,” Sep. 2019, doi: 10.1007/978-3-030-28954-6_1.
66. P. Rajpurkar et al., “CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,” Nov. 2017, Accessed: Sep. 27, 2021. [Online]. Available: https://arxiv.org/abs/1711.05225.
67. J. Wu et al., “Expert identification of visual primitives used by CNNs during mammogram classification,” in Medical Imaging 2018: Computer-Aided Diagnosis, Houston, Texas, USA, 10-15 February 2018, 2018, vol. 10575, p. 105752T, doi: 10.1117/12.2293890.
68. J. Wu et al., “DeepMiner: Discovering Interpretable Representations for Mammogram Classification and Explanation,” May 2018, Accessed: Sep. 27, 2021. [Online]. Available: http://arxiv.org/abs/1805.12323.
69. V. Srinivasan, S. Lapuschkin, C. Hellge, K.-R. Müller, and W. Samek, “Interpretable human action recognition in compressed domain,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 1692–1696, doi: 10.1109/ICASSP.2017.7952445.
70. S. Lapuschkin, A. Binder, G. Montavon, K.-R. Müller, and W. Samek, “Analyzing Classifiers: Fisher Vectors and Deep Neural Networks,” 2016, pp. 2912–2920, doi: 10.1109/CVPR.2016.318.
71. S. Lapuschkin, A. Binder, K.-R. Müller, and W. Samek, “Understanding and Comparing Deep Neural Networks for Age and Gender Classification,” Aug. 2017, Accessed: Sep. 27, 2021. [Online]. Available: http://arxiv.org/abs/1708.07689.
72. K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks: Visualising image classification models and saliency maps,” Dec. 2014, Accessed: Sep. 13, 2021. [Online]. Available: https://arxiv.org/abs/1312.6034.
73. A. Mahendran and A. Vedaldi, “Salient Deconvolutional Networks,” ECCV, pp. 120–135, 2016.
74. J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” in Advances in Neural Information Processing Systems, Oct. 2018, vol. 2018-Decem, pp. 9505–9515, Accessed: Oct. 12, 2021. [Online]. Available: http://arxiv.org/abs/1810.03292.
75. N. Bansal, C. Agarwal, and A. Nguyen, “SAM: The Sensitivity of Attribution Methods to Hyperparameters,” Mar. 2020, Accessed: Sep. 27, 2021. [Online]. Available: http://arxiv.org/abs/2003.08754.
76. P. J. Kindermans et al., “The (Un)reliability of Saliency Methods,” in Lecture Notes in Computer Science, vol. 11700 LNCS, 2019, pp. 267–280.
77. A. Ghorbani, A. Abid, and J. Zou, “Interpretation of Neural Networks is Fragile,” Oct. 2017, Accessed: Oct. 12, 2021. [Online]. Available: http://arxiv.org/abs/1710.10547.
78. M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in Computer Vision - ECCV, 2014, vol. 8689 LNCS, no. PART 1, pp. 818–833, doi: 10.1007/978-3-319-10590-1_53.
79. J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller, “Striving for simplicity: The all convolutional net,” Dec. 2015, Accessed: Oct. 11, 2021. [Online]. Available: https://arxiv.org/abs/1412.6806.
80. D. Smilkov, N. Thorat, B. Kim, F. Viégas, and M. Wattenberg, “SmoothGrad: removing noise by adding noise,” Jun. 2017, Accessed: Sep. 28, 2021. [Online]. Available: https://arxiv.org/abs/1706.03825.
81. J. Zhang, S. A. Bargal, Z. Lin, J. Brandt, X. Shen, and S. Sclaroff, “Top-Down Neural Attention by Excitation Backprop,” Int. J. Comput. Vis., vol. 126, no. 10, pp. 1084–1102, Aug. 2016, doi: 10.1007/s11263-017-1059-x.
82. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” Oct. 2016, doi: 10.1007/s11263-019-01228-7.
83. M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should i trust you?’ Explaining the predictions of any classifier,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Feb. 2016, vol. 13-17-Augu, pp. 1135–1144, doi: 10.1145/2939672.2939778.
84. M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in 34th International Conference on Machine Learning, ICML 2017, Mar. 2017, vol. 7, pp. 5109–5118, Accessed: Oct. 12, 2021. [Online]. Available: http://arxiv.org/abs/1703.01365.
85. S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, and W. Samek, “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation,” PLoS One, vol. 10, p. e0130140, 2015, doi: 10.1371/journal.pone.0130140.
86. A. Kurakin, I. J. Goodfellow, and S. Bengio, “Adversarial examples in the physical world,” Jul. 2017, doi: 10.1201/9781351251389-8.
87. S. M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Nov. 2016, vol. 2016-Decem, pp. 2574–2582, doi: 10.1109/CVPR.2016.282.
88. M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, “Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition,” in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, 2016, pp. 1528–1540, doi: 10.1145/2976749.2978392.
89. K. Eykholt et al., “Robust Physical-World Attacks on Deep Learning Models,” CVPR, Jul. 2018, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1707.08945.
90. A. Shrikumar, P. Greenside, and A. Kundaje, “Learning important features through propagating activation differences,” in 34th International Conference on Machine Learning, ICML 2017, May 2017, vol. 7, pp. 4844–4866, Accessed: Oct. 12, 2021. [Online]. Available: https://arxiv.org/abs/1605.01713.
91. G. Montavon, S. Lapuschkin, A. Binder, W. Samek, and K.-R. Müller, “Explaining nonlinear classification decisions with deep Taylor decomposition,” Pattern Recognit., vol. 65, pp. 211–222, Dec. 2017, doi: 10.1016/j.patcog.2016.11.008.
92. S. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” May 2017, Accessed: Dec. 10, 2021. [Online]. Available: https://arxiv.org/abs/1705.07874.
93. P. Dabkowski and Y. Gal, “Real time image saliency for black box classifiers,” in Advances in Neural Information Processing Systems, May 2017, vol. 2017-Decem, pp. 6968–6977, Accessed: Oct. 15, 2021. [Online]. Available: http://arxiv.org/abs/1705.07857.
94. B. Goodman and S. Flaxman, “European Union regulations on algorithmic decision-making and a &quot;right to explanation&quot;,” Jun. 2016, doi: 10.1609/aimag.v38i3.2741.
95. B. Hutchinson and M. Mitchell, “50 Years of Test (Un)fairness: Lessons for machine learning,” in FAT 2019 - Proceedings of the 2019 Conference on Fairness, Accountability, and Transparency, Nov. 2019, pp. 49–58, Accessed: May 13, 2021. [Online]. Available: https://arxiv.org/abs/1811.10104.
96. C. Szegedy et al., “Going deeper with convolutions,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Sep. 2015, vol. 07-12-June, pp. 1–9, doi: 10.1109/CVPR.2015.7298594.
97. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Dec. 2016, vol. 2016-Decem, pp. 770–778, doi: 10.1109/CVPR.2016.90.
98. K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” Sep. 2015, Accessed: Oct. 22, 2021. [Online]. Available: http://arxiv.org/abs/1409.1556.
99. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” in Advances in Neural Information Processing Systems, 2012, vol. 25, [Online]. Available: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.
100. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2010, pp. 248–255, doi: 10.1109/cvpr.2009.5206848.
101. D. H. HUBEL and T. N. WIESEL, “Receptive fields of single neurones in the cat’s striate cortex,” J. Physiol., vol. 148, no. 3, pp. 574–591, Oct. 1959, [Online]. Available: https://pubmed.ncbi.nlm.nih.gov/14403679.
102. A. Nguyen, A. Dosovitskiy, J. Yosinski, T. Brox, and J. Clune, “Synthesizing the preferred inputs for neurons in neural networks via deep generator networks,” in Advances in Neural Information Processing Systems, May 2016, pp. 3395–3403, Accessed: Sep. 06, 2021. [Online]. Available: https://arxiv.org/abs/1605.09304.
103. J. Yosinski, J. Clune, A. Nguyen, T. Fuchs, and H. Lipson, “Understanding Neural Networks Through Deep Visualization,” ICML Conf., 2015, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1506.06579.
104. A. Mordvintsev, C. Olah, and M. Tyka, “Inceptionism: Going Deeper into Neural Networks,” Google Res. Blog, 2015, [Online]. Available: https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html.
105. A. Nguyen, J. Yosinski, and J. Clune, “Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks,” ICML Conf., Feb. 2016, Accessed: Sep. 06, 2021. [Online]. Available: https://arxiv.org/abs/1602.03616.
106. D. Erhan, Y. Bengio, A. Courville, and P. Vincent, “Visualizing Higher-Layer Features of a Deep Network,” ICML, 2009.
107. A. Nguyen, “AI Neuroscience: Visualizing and Understanding Deep Neural Networks,” 2017.
108. D. Wei, B. Zhou, A. Torrabla, and W. Freeman, “Understanding Intra-Class Knowledge Inside CNN,” arXiv Prepr., Jul. 2015, Accessed: Sep. 06, 2021. [Online]. Available: https://arxiv.org/abs/1507.02379.
109. A. Nguyen, J. Yosinski, and J. Clune, “Deep neural networks are easily fooled: High confidence predictions for unrecognizable images,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Dec. 2015, vol. 07-12-June, pp. 427–436, doi: 10.1109/CVPR.2015.7298640.
110. A. Nguyen, J. Clune, Y. Bengio, A. Dosovitskiy, and J. Yosinski, “Plug and play generative networks: Conditional iterative generation of images in latent space,” Nov. 2017, Accessed: Sep. 08, 2021. [Online]. Available: http://arxiv.org/abs/1612.00005.
111. C. Olah et al., “The Building Blocks of Interpretability,” Distill, vol. 3, 2018.
112. A. Dosovitskiy and T. Brox, “Generating images with perceptual similarity metrics based on deep networks,” in Advances in Neural Information Processing Systems, Feb. 2016, pp. 658–666, Accessed: Sep. 13, 2021. [Online]. Available: https://arxiv.org/abs/1602.02644.
113. I. Goodfellow et al., “Generative adversarial networks,” Adv. Neural Inf. Process. Syst., vol. 27, Jun. 2014, Accessed: Sep. 13, 2021. [Online]. Available: http://arxiv.org/abs/1406.2661.
114. C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, and S. Carter, “Zoom In: An Introduction to Circuits,” Distill, 2020, doi: 10.23915/distill.00024.001.
115. D. Bau, B. Zhou, A. Khosla, A. Oliva, and A. Torralba, “Network dissection: Quantifying interpretability of deep visual representations,” in Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Apr. 2017, vol. 2017-Janua, pp. 3319–3327, doi: 10.1109/CVPR.2017.354.
116. B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning Deep Features for Discriminative Localization,” Dec. 2015, Accessed: Sep. 28, 2021. [Online]. Available: https://arxiv.org/abs/1512.04150.
117. V. Dumoulin and F. Visin, “A guide to convolution arithmetic for deep learning,” arXiv, Mar. 2016, Accessed: Oct. 19, 2021. [Online]. Available: https://arxiv.org/abs/1603.07285.
118. S. Shi, Y. Du, and W. Fan, “An Extension of LIME with Improvement of Interpretability and Fidelity,” arXiv, Apr. 2020, Accessed: Oct. 20, 2021. [Online]. Available: http://arxiv.org/abs/2004.12277.
119. S. M. Lundberg, “SHAP Library.” https://github.com/slundberg/shap.
120. N. Kokhlikyan et al., “Captum: A unified and generic model interpretability library for PyTorch,” arXiv, Sep. 2020, Accessed: Oct. 20, 2021. [Online]. Available: https://arxiv.org/abs/2009.07896.
121. V. Arya et al., “AI Explainability 360: Impact and Design,” arXiv, Sep. 2021, Accessed: Oct. 21, 2021. [Online]. Available: https://arxiv.org/abs/2109.12151.
122. R. Luss et al., “Leveraging Latent Features for Local Explanations,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, May 2021, pp. 1139–1149, doi: 10.1145/3447548.3467265.
123. A. Kumar, P. Sattigeri, and A. Balakrishnan, “Variational inference of disentangled latent concepts from unlabeled observations,” Nov. 2018, Accessed: Oct. 21, 2021. [Online]. Available: http://arxiv.org/abs/1711.00848.
124. C. H. Chang, E. Creager, A. Goldenberg, and D. Duvenaud, “Explaining image classifiers by counterfactual generation,” Jul. 2019, Accessed: Oct. 14, 2021. [Online]. Available: http://arxiv.org/abs/1807.08024.
125. S. M. Lundberg et al., “Explainable AI for Trees: From Local Explanations to Global Understanding,” May 2019, Accessed: Dec. 10, 2021. [Online]. Available: http://arxiv.org/abs/1905.04610.
126. J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang, “Generative Image Inpainting with Contextual Attention,” CVPR, Jan. 2018, doi: 10.48550/arxiv.1801.07892.
127. P. Rodríguez, “Total Variation Regularization Algorithms for Images Corrupted with Different Noise Models: A Review,” J. Electr. Comput. Eng., vol. 2013, p. 217021, 2013, doi: 10.1155/2013/217021.
128. V. V. Estrela, H. A. Magalhaes, and O. Saotome, “Total Variation Applications in Computer Vision,” CVPR, Mar. 2016, doi: 10.4018/978-1-4666-8654-0.ch002.
129. A. Khan, A. Sohail, U. Zahoora, and A. S. Qureshi, “A survey of the recent architectures of deep convolutional neural networks,” Artif. Intell. Rev., vol. 53, no. 8, pp. 5455–5516, 2020, doi: 10.1007/s10462-020-09825-6.
130. O. Russakovsky et al., “ImageNet Large Scale Visual Recognition Challenge,” Sep. 2014, doi: 10.48550/arxiv.1409.0575.
131. T. Y. Lin et al., “Microsoft COCO: Common objects in context,” in Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), May 2014, vol. 8693 LNCS, no. PART 5, pp. 740–755, doi: 10.1007/978-3-319-10602-1_48.
132. M. Du, N. Liu, Q. Song, and X. Hu, “Towards explanation of DNN-based prediction with guided feature inversion,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Mar. 2018, pp. 1358–1367, doi: 10.1145/3219819.3220099.
133. N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), 2005, vol. 1, pp. 886–893 vol. 1, doi: 10.1109/CVPR.2005.177.
134. W. Kirch, Ed., “Pearson’s Correlation Coefficient BT  - Encyclopedia of Public Health,” Dordrecht: Springer Netherlands, 2008, pp. 1090–1091.
135. H. F. Tipton and H. F. Krause, “Understanding SSIM,” in Information Security Management Handbook, 2021, pp. 313–330.
136. R. Geirhos, C. Michaelis, F. A. Wichmann, P. Rubisch, M. Bethge, and W. Brendel, “Imagenet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness,” Nov. 2019, doi: 10.48550/arxiv.1811.12231.
137. A. Mądry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” Jun. 2018, Accessed: May 01, 2021. [Online]. Available: https://arxiv.org/abs/1706.06083.
138. L. Engstrom, A. Ilyas, H. Salman, S. Santurkar, and D. Tsipras, “Robustness (Python Library).” 2019, [Online]. Available: https://github.com/MadryLab/robustness.

## Contact
For any inquiries or collaborations, please contact Fabio Guzmán at fabioandres.guzman@gmail.com

---

Feel free to explore the repository and contribute to this ongoing research.
