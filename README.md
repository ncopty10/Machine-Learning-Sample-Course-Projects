# Machine-Learning-Sample-Course-Projects

## K‑Means Clustering Algorithm

K‑Means is a classic unsupervised learning method used to group similar data points into clusters. Its purpose is to divide a dataset into K clusters so that points within the same group are more alike than those in different groups. Each cluster is represented by a centroid, which is the average of all points in that cluster.

The algorithm works by using Euclidean distance to measure similarity between data points and centroids. In each iteration, every point is assigned to the nearest centroid, and then the centroids are updated as the mean of the points assigned to them. This process continues until the centroids stop moving significantly or a set number of iterations is reached.

In this implementation, the algorithm is written in Python using only NumPy for calculations. The function accepts a dataset array and a user‑specified number of clusters 
𝜅, and returns both the final centroids and the cluster labels for each data point. Centroids are initialized by randomly selecting data points, which is a common strategy to start the K‑Means process.

## Comparison Between Binary Linear Classifier and Support Vector Machine

This assignment focused on empirically comparing the performance of a basic binary linear classifier with a Support Vector Machine (SVM) using the Iris dataset. The two models were evaluated based on metrics such as accuracy, decision boundaries, margins, support vectors, and how well they adapt to different training sizes.

A binary linear classifier separates data points into two categories using a linear decision boundary. In this case, it used sepal measurements from the Iris dataset. Logistic Regression, the model used here, estimates the probability of a sample belonging to one of the two classes using the sigmoid activation and is trained by optimizing a likelihood-based loss.

In contrast, a Support Vector Machine identifies the best separating line (hyperplane) that maximizes the margin between classes. The points closest to this boundary are called support vectors and have a direct influence on the placement of the boundary. SVMs are capable of handling both linear and non-linear data, depending on the kernel chosen, and are valued for their robustness and generalization ability.

Implementation details include using scikit-learn’s Logistic Regression and Linear SVM modules. The Logistic Regression model utilized the sigmoid function and cross-entropy loss, while the SVM used hinge loss and optimized the margin. Both models’ decision boundaries were visualized to illustrate how they classify the data.

The analysis showed clear differences in behavior between the two methods, emphasizing how the choice of classifier and dataset configuration can affect separation boundaries and prediction accuracy. Understanding these differences is important when selecting models for practical machine learning tasks.

## Deep Learning with Convolutional Neural Networks

This project covers multiple stages involving the design, training, evaluation, and refinement of Convolutional Neural Networks (CNNs).

The first part involves examining dataset splits for training, validation, and testing, and determining how many iterations occur in 30 epochs. A custom convolution filter and linear layer are implemented and verified against library equivalents. The distinction between training and validation sets is also clarified.

The second part focuses on training the CNN and configuring its behavior. A stochastic gradient descent optimizer is completed and batch sampling is implemented for both training and validation data. Training progress is tracked through loss and accuracy metrics, accompanied by visualizations. Hyperparameters are defined, and the importance of deferring test set evaluation until after tuning is explained. Four hyperparameters are selected to build two alternative models (M1 and M2), and their best validation accuracies and corresponding epochs are reported. A final model is chosen based on validation performance and its test set accuracy is recorded.

In the third part, this workflow is applied to a new dataset. The data is imported and partitioned, a base model is created, and hyperparameters are tuned to improve validation accuracy by 5–10%. The effects of the hyperparameter adjustments are discussed, and the final model’s accuracy on the test set is reported.

The final part involves exploratory hyperparameter tuning to achieve further validation improvements and comparison of the final test performance relative to the base model.

## Sentiment Analysis Using Gated Recurrent Neural Networks

This task focuses on using a Gated Recurrent Unit (GRU)‑based neural network to classify movie reviews from the IMDB dataset as either positive or negative. Sentiment analysis involves identifying the emotional tone in text, and the IMDB dataset provides 50,000 labeled reviews for this purpose.

Before training, text must be converted into numerical form. Two common methods are one‑hot encoding, which represents each word as a binary vector, and word embeddings, which map words into dense vector spaces where similar words appear closer together.

The model architecture begins with an embedding layer that transforms word indices into dense vectors. These embeddings are then fed into a GRU, a type of recurrent network that uses update and reset gates to capture patterns and dependencies in sequences. Outputs from the GRU are passed through fully connected layers with ReLU activations to produce sentiment predictions.

Training uses two optimizers (Stochastic Gradient Descent (SGD) and Adam) to compare their effects on model performance. Early stopping with a patience of five epochs is applied to prevent overfitting by stopping training when validation loss stops improving.

The assignment emphasizes training with the Adam optimizer, implementing early stopping, and understanding how these techniques influence the model’s accuracy and generalization.
