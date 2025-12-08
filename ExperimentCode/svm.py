"""
svm.py — Skeleton for Part II - Implementation of a Linear SVM with Hinge Loss optimization via gradient descent 

Do NOT use sklearn's SVC/LinearSVC or any external SVM solvers.
Allowed: numpy, matplotlib (for plots in your notebook), and sklearn.datasets ONLY for data generation/usage.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _check_X_y(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize the input data (X, y) for SVM training or evaluation.

    This helper ensures that:
      - X is a 2D NumPy array of floats.
      - y is a 1D NumPy array of binary labels in {-1, +1}.
      - If y contains {0, 1}, it is automatically remapped to {-1, +1}.
      - Shapes of X and y are consistent (same number of samples).

    The function is called internally at the start of each computation step
    (`fit`, `predict`, `score`, etc.) to prevent shape/type errors and maintain
    consistent conventions across the implementation.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
        Must be convertible to a NumPy array of dtype float64.
    y : np.ndarray
        Target label vector of shape (n_samples,).
        Must contain only values in {-1, +1} (or {0, 1}, which will be remapped).

    Returns
    -------
    X_clean : np.ndarray of shape (n_samples, n_features)
        Cleaned, float64-typed feature matrix.
    y_clean : np.ndarray of shape (n_samples,)
        Cleaned binary labels in {-1, +1}, float64-typed.

    Raises
    ------
    ValueError
        If:
        - X is not a 2D array,
        - y is not 1D or mismatched in length with X,
        - y contains labels other than {-1, +1, 0, 1}.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y size mismatch: {X.shape[0]} vs {y.shape[0]}")

    uniq = np.unique(y)
    if set(uniq.tolist()) == {0.0, 1.0}:
        y = np.where(y == 0.0, -1.0, 1.0)

    allowed = {-1.0, 1.0}
    if not set(np.unique(y).tolist()).issubset(allowed):
        raise ValueError(f"y must be in {{-1,+1}}; got {np.unique(y)}")

    return X, y

# -----------------------------------------------------------------------------
# This function plots 2D data and the SVM decision boundary and margins. It can be helpful in your project
# -----------------------------------------------------------------------------
def plot_linear_svm_2d(clf, X, y):
    import matplotlib.pyplot as plt
    w, b = clf.w, clf.b
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', alpha=0.7)

    # set bounds from data
    pad = 0.5
    ax.set_xlim(X[:,0].min()-pad, X[:,0].max()+pad)
    ax.set_ylim(X[:,1].min()-pad, X[:,1].max()+pad)

    # Clip and draw the lines a*x + b*y + c = 0 with (a,b,c)=(w0,w1,b)
    def clip_and_plot(a, bcoef, c, style, label=None):
        X0, X1 = ax.get_xlim(); Y0, Y1 = ax.get_ylim()
        pts = []
        if abs(bcoef) > 1e-12:
            yL = -(a*X0 + c)/bcoef; yR = -(a*X1 + c)/bcoef
            if Y0 <= yL <= Y1: pts.append((X0, yL))
            if Y0 <= yR <= Y1: pts.append((X1, yR))
        if abs(a) > 1e-12:
            xB = -(bcoef*Y0 + c)/a; xT = -(bcoef*Y1 + c)/a
            if X0 <= xB <= X1: pts.append((xB, Y0))
            if X0 <= xT <= X1: pts.append((xT, Y1))
        if len(pts) >= 2:
            (xA,yA),(xB,yB) = pts[0], next(p for p in pts[1:] if (abs(p[0]-pts[0][0])+abs(p[1]-pts[0][1]))>1e-9)
            ax.plot([xA,xB],[yA,yB], style, label=label)

    a, bcoef, c = w[0], w[1], b
    clip_and_plot(a,bcoef,c, 'k-', label='decision')
    clip_and_plot(a,bcoef,c-1, 'k--')
    clip_and_plot(a,bcoef,c+1, 'k--')
    ax.legend() 
    ax.set_title("Linear SVM (+/-1 margins)")
    ax.set_xlabel("X1") 
    ax.set_ylabel("X2")
    plt.show()
    
# -----------------------------------------------------------------------------
# Data class to store hyperparameters and other handy config parameters
# -----------------------------------------------------------------------------
@dataclass
class LinearSVMConfig:
    C: float = 1.0                
    learning_rate: float = 1e-3
    n_epochs: int = 1000
    batch_size: int = 64
    shuffle: bool = True
    random_state: Optional[int] = 0
    verbose: bool = False


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class LinearSVM:
    """Linear SVM with hinge loss optimized by (mini-batch) gradient descent.

    Objective:
        L(w,b) = 0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i (w^T x_i + b))
    """

    def __init__(self, config: Optional[LinearSVMConfig] = None):
        self.config = config or LinearSVMConfig()
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history_: Dict[str, list] = {"loss": []}
        self._rng = np.random.RandomState(self.config.random_state) if self.config.random_state is not None else np.random

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        """
        Train the Linear SVM using mini-batch stochastic gradient descent (SGD)
        on the hinge loss with L2 regularization.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Binary labels in {-1, +1}.

        Returns
        -------
        self : LinearSVM
            Trained model instance (for chaining).
        """
        X, y = _check_X_y(X, y)
        n_samples, n_features = X.shape

        # TODO: initialize parameters
        self.w = self._rng.randn(n_features) * 0.001 
        self.b = 0.0

        for epoch in range(self.config.n_epochs):
            # Shuffling indices
            idx = np.arange(n_samples)
            if self.config.shuffle:
                self._rng.shuffle(idx)
            Xs, ys = X[idx], y[idx]

            bs = max(1, int(self.config.batch_size))
            epoch_losses = []

            for start in range(0, n_samples, bs):
                end = start + bs
                Xb = Xs[start:end]
                yb = ys[start:end]

                # --- TODO: compute gradients dw, db for this batch ---
                dw, db = self._batch_gradients(Xb, yb)

                # --- TODO: update parameters ---
                self.w -= self.config.learning_rate * dw
                self.b -= self.config.learning_rate * db

                # Track batch loss
                epoch_losses.append(self._loss(Xb, yb))

            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float(self._loss(X, y))
            self.history_["loss"].append(mean_epoch_loss)

            if self.config.verbose and (epoch % max(1, self.config.n_epochs // 10) == 0):
                print(f"[epoch {epoch:4d}] loss={mean_epoch_loss:.6f}")

        return self

        # ---------------------------------------------------------------------
    # Core of the SVM optimization
    # ---------------------------------------------------------------------
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the value of the hinge loss function on a given batch.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, n_features)
            Feature matrix for current batch.
        y : np.ndarray of shape (batch_size,)
            Corresponding binary labels {-1, +1}.

        Returns
        -------
        loss : float
            Scalar objective value for the batch.
        """
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        X, y = _check_X_y(X, y)

        # TODO: implement hinge loss correctly and return float. COMMENT YOUR CODE!!
        #Computing scores yi = (wT*xi +b)
        samples = X.shape[0]
        scores = np.zeros(samples)

        for i in range(samples):
            xi = X[i, :]

            weighted_sum = np.dot(xi, self.w)

            scores[i] = weighted_sum + self.b

        functional_margin = y * scores
        
        hinge_loss_components = np.maximum(0, 1 - functional_margin)
        
        data_loss = self.config.C * np.sum(hinge_loss_components)
        
        regularization_term = 0.5 * np.dot(self.w, self.w)
        
        total_loss = regularization_term + data_loss

        return float(total_loss)
        
    def _batch_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of the SVM loss with respect to weights (w) and bias (b)
        for one mini-batch.

        Parameters
        ----------
        X : np.ndarray of shape (batch_size, n_features)
            Feature matrix for the current batch.
        y : np.ndarray of shape (batch_size,)
            Corresponding binary labels {-1, +1}.

        Returns
        -------
        dw : np.ndarray of shape (n_features,)
            Gradient with respect to weights.
        db : float
            Gradient with respect to bias.
        """
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        X, y = _check_X_y(X, y)

        # TODO: implement and return (dw, db) COMMENT YOUR CODE!!
        n_batch = X.shape[0]
        C = self.config.C

        # 1. Compute the functional margin: y_i * (w^T x_i + b)
        scores = X @ self.w + self.b
        functional_margin = y * scores
        
        # 2. Determine which samples violate the margin (margin < 1)
        # These are the "misclassified" or "support vector" samples that contribute to the loss.
        # The gradient is non-zero only for these samples.
        is_margin_violation = functional_margin < 1
        
        # 3. Compute the gradient of the loss with respect to the weights (dw)
        # Gradient is: dw = w + C * (sum over violations) (-y_i * x_i)
        
        # Start with the regularization gradient: dw = w
        dw = self.w.copy()
        
        # Calculate the gradient term for the data loss: -C * y_i * x_i
        # Mask X and y to include only the samples that violate the margin (i.e., where loss > 0)
        X_violations = X[is_margin_violation]
        y_violations = y[is_margin_violation]
        
        # The sum part: -(y_violations * X_violations)
        # Note: y_violations must be broadcasted correctly (multiplied row-wise)
        gradient_sum = -np.dot(y_violations, X_violations) 
        
        # Add the data loss gradient term to dw, normalized by batch size for SGD
        dw += C * (gradient_sum / n_batch)

        # 4. Compute the gradient of the loss with respect to the bias (db)
        # Gradient is: db = C * (sum over violations) (-y_i)
        
        # The sum part: -sum(y_violations)
        gradient_sum_b = -np.sum(y_violations)
        
        # Calculate db, normalized by batch size for SGD
        db = C * (gradient_sum_b / n_batch)
        
        return dw, float(db)
        
        
    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the raw (unnormalized) decision scores for input samples.

        The decision function is:
            f(x) = wᵀx + b

        These scores indicate how far each point is from the separating hyperplane:
          - f(x) > 0 → predicted class +1
          - f(x) < 0 → predicted class -1
          - |f(x)| is proportional to the distance from the margin boundary.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for which to compute decision scores.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Signed distances (unnormalized) to the decision boundary.
        """
        if self.w is None:
            raise RuntimeError("Model not fitted.")

        #TODO COMMENT YOUR CODE!!
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input samples based on the decision function.

        Uses the sign of the decision function:
            ŷ = sign(f(x)) = sign(wᵀx + b)

        A positive score yields label +1; a negative score yields label -1.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels in {-1, +1}.
        """
        
        #TODO COMMENT YOUR CODE!!
        scores = self.decision_function(X)
        
        # The sign function: returns -1 for negative, 1 for positive, 0 for zero
        # In this context, 0 is typically treated as +1 or -1; np.sign does this well.
        # np.sign(0) returns 0. For safety and standard practice, convert 0 scores to +1.
        y_pred = np.sign(scores)
        
        # Correctly map any zero scores (on the boundary) to +1, if they occur.
        y_pred[y_pred == 0] = 1.0
        
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the mean classification accuracy on the given test data and labels.

        Accuracy is defined as the fraction of correctly predicted labels:
            accuracy = (1 / n) * Σ 𝟙[ŷᵢ == yᵢ]

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for evaluation.
        y : np.ndarray of shape (n_samples,)
            True binary labels in {-1, +1}.

        Returns
        -------
        accuracy : float
            Proportion of correctly classified samples in [0, 1].
        """
        Xc, yc = _check_X_y(X, y) #Validates data format
        
        #TODO COMMENT YOUR CODE!!
        y_pred = self.predict(Xc)
        
        # Compute the number of correct predictions (where y_pred == yc)
        n_correct = np.sum(y_pred == yc)
        
        # Accuracy is the number of correct predictions divided by the total number of samples
        n_samples = Xc.shape[0]
        accuracy = n_correct / n_samples
        
        return float(accuracy)