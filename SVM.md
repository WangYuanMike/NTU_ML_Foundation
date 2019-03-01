# SVM summary

## [SVM primal problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/201_handout.pdf)
### What
- Support Vector Machine(SVM) is a linear binary classification model whic can find the "fattest" hyperplane as decision boundary
- Support Vector is the data instance which lie on the margin, i.e. the one which has the minimum distance to the hyperplane
### Why
- Simple but powerful model with good explainability
- Better generalization performance than other linear model because of large margin (large margin -> limited number of hyperplanes -> smaller VC dimension -> good resistance to overfitting)
- Fast to train (using LIBLINEAER, LIBSVM, or sklearn.SVC)
- Linear model -> easy to deploy and predict
### How
- Maximize the distance of the data instance that is closest to the hyperplane 
  - get the unit normal vector of deciding hyperplane (**w**)
  - get the difference vector (**v**) between the data instance and its intersection with the hyperplane
  - computer the inner product between **w** and **v** to get the margin
  - maximize the margin

TODO: add image

- Solve an equivalent quardratic programming problem (plenty of algorithms and packages to solve QP problem)

TODO: add image

![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_primal.png)
### When and Where
- Training data are linear seperable (but you do not know this info before training)
- So the heuristic to use this model are: 
  - number of instances << number of features, e.g. bioinformatics
  - both number of instances and features are large, e.g. document classification
### Cons:
- Cannot handle case when data are not linear separable -> use SVM dual problem and kernel function to solve this issue
- This model is usually called **hard margin SVM**, and the above heuristic use cases are usually more suitable for [soft margin SVM](#soft-margin-svm)

## [SVM dual problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/202_handout.pdf)
### What
- Lagrange dual problem of the SVM primal QP problem
### Why
- When training data are not linear separable, one solution is to map the original feature to higher dimensional space
- In a higher dimensional space (could even be an infinite dimensional space), the data could be linear separable, and due to large margin, the VC dimension of the model is still limited. Therefore, we can get a sophisticated decision boundary with not so many hyperthesis to choose.
- But the complexity of solving primal QP problem depends on the number of features, which means higher dimensional feature would cause much higher complexity to solve the problem, sometimes even impossible, e.g. infinite dimension
- The solution is to solve its Lagrange dual problem, because it transforms variables to optimize from weight and bias to the Lagrange multipliers of data instances, i.e. from very high (or inifite) dimension to rather low (or finite) dimension
### How
- Use function phi to map current feature vector into higher dimensional space

TODO: add image

- Create the Lagrange function and transform the primal problem into an equivalent **min-max** optimization problem without constraints

TODO: add image

- Reason for the equivalence: For those weight and bias which generates a positive constaint (those does not fulfill the constraints), the maximization will make the inner function exploded to positive infinite， which in turn will make these weight and bias eliminated by the outer minimization step.

TODO: add image

- When below conditions are met, the optimal of the primal min-max problem is equal to the one of the dual max-min problem.
  - primal problem is convex
  - constraints are linear
  - primal problem is feasible (linear separable in the mapped high dimensional space)

TODO: add image

- Take gradient of Lagrange function with respect to weight, bias, and alpha, set these gradient to zero (condition for optimal solution) to get some relations between variables and some new constraints, and plug them back into the dual max-min problem to get a simple maximization QP problem

TODO: add image

- Use QP package to solve the problem and get the optimal alpha

TODO: add image

- Use **KKT optimality condition** (the sufficient and necessary condition for optimality) to compute weight and bias:
  - primal feasible (not relevant with computing weight and bias)
  - dual feasible (not relevant with computing weight and bias)
  - dual-inner optimal -> **only support vectors' alpha are bigger than 0** -> only need support vector to compute weight
  - primal-inner optimal (also called **complementary slackness**) -> use any support vector to compute bias or take average of support vectors to avoid numerical errors
  
TODO: add image

### When and Where
- no practical use case, see details in Cons below

### Cons
- This model successfully transforms number of to-be-optimized variables from number of features (could be infinite in the mapped feature space) into number of data instances. But it still does not completely solve the complexity issue of handling high dimensional features, because there is an inner product between two high dimensional instances needs to be computed in the dual problem, which leads to the next chapter **kernel SVM**

## [Kernel SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/203_handout.pdf)
## What
## Why
## How

## [Soft margin SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/204_handout.pdf)
