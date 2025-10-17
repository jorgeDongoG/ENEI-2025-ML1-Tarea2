## Assignment: Logistic Regression and Multiclass Extensions

**Deadline:** Monday, October 13th, 2025, 23:59

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ucimlrepo`.

---

### Part A. Binary Logistic Regression from Scratch

1. **Dataset**
   Use the **Heart Disease dataset** from the UCI repository. You can do this by running:
```python
!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id = 45)
X = heart_disease.data.features # features
Y = heart_disease.data.targets # heart disease diagnosis

```
  Originally, the Y variable is an integer with varying values. Recode it to be either 0 (when the original value is 0) or 1 (otherwise)

   * Task: predict whether a patient has heart disease.
   * Standardize numeric features, one-hot encode categorical ones.
   * Split into 70% train / 30% test.

3. **Model Derivation and Implementation**

   * Implement gradient descent to maximize the log-likelihood (or equivalently, minimize the negative log-likelihood).
   * Show convergence plots for at least two learning rates.

4. **Evaluation**

   * Compute accuracy, precision, recall, F1 score in the test set.
   * Compare with `sklearn.linear_model.LogisticRegression`.

---

### Part B. Multiclass Logistic Regression via One-vs-All (OvA)

4. **Dataset**
   Use the **Wine dataset** (`from sklearn.datasets import load_wine`).

   * There are 3 wine cultivars (classes) with 13 chemical features.
   * Standardize all features.

5. **OvA Implementation**

   * Build **three binary classifiers**, each distinguishing one class vs. all others.
   * Use your binary logistic regression optimizer from Part A.
   * For prediction:

     * Compute probabilities from each classifier.
     * Assign each observation to the class with the highest predicted probability.
   * Report confusion matrix and accuracy.

6. **Comparison**

   * Fit `LogisticRegression(multi_class="ovr")` from sklearn.
   * Compare coefficients and accuracy to your own implementation.

---

### Part C. Multinomial (Softmax) Logistic Regression from Scratch

7. **Theory**

   * Derive the gradient of the log-likelihood function for muticlass classification (check the [notebook for session 4](https://colab.research.google.com/drive/1QKPnTQ_CtqY_4IZHr_dUAzR3nfj8bLbW?usp=sharing))

8. **Implementation**

   * Implement gradient descent updating all class weight vectors simultaneously.
   * Include a `softmax` function with numerical stability (`z -= np.max(z, axis=1, keepdims=True)` before exponentiation).
   * Monitor log-likelihood convergence.

9. **Evaluation**

   * Use the same Wine dataset.
   * Compute accuracy, per-class precision/recall, and confusion matrix.
   * Compare to `LogisticRegression(multi_class="multinomial", solver="lbfgs")`.

---

### Deliverables

You must fork the [original repository](https://github.com/RodrigoGrijalba/ENEI-2025-ML1-Tarea2), and turn in a link to your group's repository with:

* A Jupyter notebook (in the `src` folder) with:

  * Binary logistic regression (from scratch and sklearn)
  * OvA and multinomial implementations
  * Gradient derivations
  * Convergence and comparison plots
* A short (~600 words) write-up explaining:

The binary model (with a learning rate of 5e-4) and the multinomial model (with a learning rate of 1e-1) both converge at a similar rate. The One-vs-All (OvA) approach can be understood simply as an extension of the binary classification framework, applied independently across multiple classes.
The use of the softmax function helps prevent the exponential terms from growing excessively fast, which could otherwise lead to numerical overflow. By normalizing the output values into a probability distribution, softmax ensures that the resulting computations remain stable and within manageable numerical limits.
According to our estimates, no significant divergence was observed. Overall, both models demonstrate a high degree of robustness, suggesting that their performance remains consistent and reliable across different specifications or estimation conditions.

