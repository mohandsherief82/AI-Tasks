# Report:

## 1. Eigen-decompositions:
   * Eigen decompositions are a type of special math operations performed on square matrices.
   * They are used to transform a system of linear transformations in nth dimension graph.
   * Eigen decompositions:
     * Eigenvectors: it represents a vector in the transformed system that didn't change its direction and only changed in the magnitude ,where it is a non-zero vector, when they are multiplied by a matrix.
     * Eigenvalues: it represents the scalar quantity that is associated with the change of magnitude of the eigenvector after the multiplication with a square matrix.


## 2. **P**rincipal **C**omponent **A**nalysis(PCA):
  * It is a feature engineering technique that simplify complex datasets by *transforming* it into a smaller sets of new features that have the same covariance and variation before the transform.
  * It represents a similar system to a linear system with n dimensions where it describes a transform on a system retaining the relation between its n features creating the same system as before with a small change.

## 3. PCA and Eigen-decompositions:
  * As PCA, represents a system of linear transforms, they have a strong relation with eigenvalues and eigenvectors.
  * The eigen-pairs can be used to retain the variation of the data as they will represent them on a eigenvector that is stretched by an eigenvalue, making a transform of the original system with the same relation between some of its features, allowing for some features that doesn't have a strong covariance to be ignored in the process simplifing the dataset.