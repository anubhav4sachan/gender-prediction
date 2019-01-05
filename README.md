# Gender Prediction from Handwriting.

The demographic categorization from the handwriting is an interesting field of research. 
This project determines/predicts the gender of the volunteer writer from its handwriting.

"Gender Prediction from Handwriting" is a machine learning project as a part of research internship under Dr. Dushyant Kumar Singh, Assistant Professor in department of Computer Science and Engineering at `Motilal Nehru National Institute of Technology Allahabad, India.`

### What does the project do?

This ML project employs one of the challenges of The Twelfth International Conference on Document Analysis and Recognition (ICDAR) to be held in Washington, DC on Kaggle.com to predict if a handwritten document has been produced by a male or female writer.

The dataset provided by Kaggle competition is the subset of Qatar University Writer Identification (QUWI) Offline Dataset. Since the competition is closed, and to evaluate the performance of the algorithms, we only use the training set which consists of 282 writers for which the genders are provided.

##### The file `tuning.py` containing the function "svmTuner" is responsible for selecting the best parameters using the Grid Search for the Support Vector Machine Classifier used in the `main.py`.

##### To run the program, you need to run the file `main.py`.