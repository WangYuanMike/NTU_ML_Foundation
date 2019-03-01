# SVM summary

### [SVM primal problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/201_handout.pdf)
#### What
- A linear binary classification model whic can find the "fattest" hyperplane as deciding boundary
#### Why
- Simple but powerful model with good explainability
- Better generalization performance than other linear model because of large margin
- Fast to train (using LIBLINEAER, LIBSVM, or sklearn.SVC)
- Light model with sparse solution (number of support vectors are usually very small) -> easy for deployment and predict
#### How
- Maximize the distance of the data instance that is closest to the hyperplane 
  - get the unit normal vector of deciding hyperplane (**w**)
  - get the difference vector (**v**) between the data instance and its intersection with the hyperplane
  - computer the inner product between **w** and **v** to get the margin
  - maximize the margin

TODO: formular image for the original svm primal problem

- Solve an equivalent quardratic programming problem (plenty of algorithms and packages to solve QP problem)

![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_primal.png)
#### When and Where
- Training data are linear seperable (but you do not know this info before training)
- So the heuristic to use this model are: 
  - number of instances << number of features, e.g. bioinformatics
  - both number of instances and features are large, e.g. document classification
#### Cons:
- Cannot handle case when data are not linear separable -> use SVM dual problem and kernel function to solve this issue

### [SVM dual problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/202_handout.pdf)
