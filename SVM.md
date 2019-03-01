# SVM summary

### [SVM primal problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/201_handout.pdf)
#### What
- A linear binary classification model whic can find the "fattest" hyperplane as decision boundary
#### Why
- Simple but powerful model with good explainability
- Better generalization performance than other linear model because of large margin (large margin -> limited number of hyperplanes -> smaller VC dimension -> good resistance to overfitting)
- Fast to train (using LIBLINEAER, LIBSVM, or sklearn.SVC)
- Light model with sparse solution (number of support vectors are usually very small, and they are the only relevant data instances for computing weight and bias of hyperplane) -> easy for deployment and predict
#### How
- Maximize the distance of the data instance that is closest to the hyperplane 
  - get the unit normal vector of deciding hyperplane (**w**)
  - get the difference vector (**v**) between the data instance and its intersection with the hyperplane
  - computer the inner product between **w** and **v** to get the margin
  - maximize the margin

TODO: formular image for the original svm primal problem

- Solve an equivalent quardratic programming problem (plenty of algorithms and packages to solve QP problem)

TODO: formular image for the svm primal QP problem

![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_primal.png)
#### When and Where
- Training data are linear seperable (but you do not know this info before training)
- So the heuristic to use this model are: 
  - number of instances << number of features, e.g. bioinformatics
  - both number of instances and features are large, e.g. document classification
#### Cons:
- Cannot handle case when data are not linear separable -> use SVM dual problem and kernel function to solve this issue

### [SVM dual problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/202_handout.pdf)
### What
- Lagrange dual problem of the SVM primal QR problem
### Why
- When training data are not linear separable, one solution is to map the original feature to higher dimension
- In a higher dimensional space (could even be an infinite dimensional space), the data could be linear separable, and due to large margin, the VC dimension of the model is still limited. Therefore, we can get a sophisticated decision boundary with not so many hyperthesis to choose.
- But the complexity of solving primal QP problem depends on the number of features, which means higher dimensional feature would cause much higher complexity to solve the problem, sometimes even impossible, e.g. infinite dimension
- The solution to the above complexity issue is to use Lagrange dual problem, because it transforms variables to optimize from weight and bias to the Lagrange multipliers of data instances -> from very high (or inifite) dimension to rather low (or finite) dimension
### How
