import sklearn.base
from sklearn.metrics import roc_auc_score

class jFitnessFunction:
    def __init__(self, model, num_feat, alpha, beta):
        """
        Initialization function
        Parameters:
            - model: Machine learning model object
            - num_feat: Number of features
            - alpha: Weight coefficient for error rate
            - beta: Weight coefficient for number of features
        """
        if not isinstance(model, sklearn.base.BaseEstimator):
            raise ValueError("Invalid model object. Model must be an instance of a scikit-learn estimator.")
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.num_feat = num_feat
        
    def __call__(self, x_train, x_test, y_train, y_test):
      """
      Call function, returns the fitness value
      Parameters:
          - x_train: Training dataset features
          - x_test: Testing dataset features
          - y_train: Training dataset labels
          - y_test: Testing dataset labels
      Returns:
          - cost: Fitness value
      """
      if x_train.size == 0 or x_test.size == 0:
          return 1
      if x_train is None:
          raise ValueError("Training dataset features cannot be None or empty.")
      if y_train is None or y_train.size == 0:
          raise ValueError("Training dataset labels cannot be None or empty.")
      if x_test is None:
          raise ValueError("Testing dataset features cannot be None or empty.")
      if y_test is None or y_test.size == 0:
          raise ValueError("Testing dataset labels cannot be None or empty.")

      self.model.fit(x_train, y_train)  # Fit the model
      error = 1 - roc_auc_score(y_test, self.model.predict_proba(x_test)[:, 1])  # Calculate AUC
      num_feat = len(x_train[0])  # Number of features
      cost = self.alpha * error + self.beta * (num_feat / self.num_feat)  # Calculate fitness value

      return cost