# Support Vector Machine(SVM)

## Table of Contents
[SVM primal problem](#svm-primal-problem)  
[SVM dual problem](#svm-dual-problem)  
[Kernel SVM](#kernel-svm)  
[Soft margin SVM](#soft-margin-svm)  
[Probabilistic SVM](#probabilistic-svm)  
[Support Vector Regression](#support-vector-regression)  
[Summary](#summary)  
[Appendix LIBSVM](#appendix-libsvm)  
   
## [SVM primal problem](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/201_handout.pdf)
### What
- **Support Vector Machine** is a linear binary classification model which can find the "fattest" hyperplane as decision boundary
- **Support Vector** is the data instance which lie on the margin, i.e. the one which has the minimum distance to the hyperplane
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
- Create the Lagrange function and transform the primal problem into an equivalent **min-max** optimization problem without constraints (actually Lagrange multipliers alpha >= 0 become the new constraints of the Lagrange functions)  
TODO: add image  
- Reason for the equivalence: 
   - **For those weight and bias which generates a positive constraint (those do not fulfill the constraints)**, the maximization will make the inner function exploded to positive infinite， which in turn will make these weight and bias eliminated by the outer minimization step
   - **For those weight and bias which generate a constraint equals zero**, alpha can be any value which >= 0(these corresponding data instances are the **Support Vector**), and only the minimization objective in the original problem will be kept
   - **For those weight and bias which generate a negative constraint**, the maximization will make alpha equals 0, which in turn will also keep the minimization objective of the original problem  
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
- This is a theoretical middle step for more advanced model. No practical use case, see details in Cons below
### Cons
- This model successfully transforms the number of to-be-optimized variables from number of features (could be infinite in the mapped feature space) into number of data instances. But it still does not completely solve the complexity issue of handling high dimensional features, because there is an inner product between two high dimensional instances needs to be computed in the dual problem, which leads to the next chapter **kernel SVM**
## [Kernel SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/203_handout.pdf)
### What
- Kernel function is a shortcut to compute inner product of high dimensional vectors by computing a function of the two corresponding low dimensional vectors, i.e. original feature vectors
### Why
- With kernel function, SVM dual problem with high dimensional feature space could be solved within the time complexity of original features' lower dimensional space
### How
- The common kernel functions are below:
   - linear kernel   TODO: add image
   - polynomial kernel  TODO: add image
   - rbf (gaussian) kernel TODO: add image
- Necessary and sufficient condition for valid kernel functions needs is **Mercer's condition** -> K matrix needs to be postive semi definite, but some kernel function which does not fulfill Mercer's condition may also work in practice
### When and Where
- Equipped with kernel function, SVM dual problem is the first practical **hard margin SVM**
- The linear kernel does not map feature vector at all, so it is valid for those scenarios mentioned in [SVM primal problem](#svm-primal-problem), and it should be tried first, but normally with **soft margin SVM**
- According to the [LIBSVM practical guide](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf), one should try rbf kernel after linear model, because it has just one hyperparameter (gamma) to tune and it maps feature into infinite dimensional space for linear separation
- The polynomial kernel should be tried with lower degree so as to resist overfitting
### Cons
- Linear kernel is safe but restricted, esp. with hard margin SVM
- Rbf kernel is mysterious, slow, and maybe too powerful
- Polynomial kernel has 3 hyperparameters to tune, and numerically difficult for high degree
- Hard margin SVM is still not good enough, because it cannot tolerate any data instance violating the margin, causing one to use an unnecessary high dimensional space for separating instances linearly, which in turn may generate a too sophisticated (easy to overfit) decision boundary. The solution is the next chapter **Soft margin SVM**.
## [Soft margin SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/204_handout.pdf)
### What
A kernel SVM which can tolerate some data instances violating the margin (even violating the decision boundary)
### Why
The reason has been described in the Cons of kernel SVM above
### How
- Add Xi (violation distance, which must be >= 0) of each data instance to the minimization objective in SVM primal problem, and adjust the constraints accordingly
- A hyperparameter **C** is added to the violation part in the minimization objective
   - A smaller C means the model can tolerate more violation
   - The default value of C is usually set to 1
- Soft margin SVM can be viewed as an error measurement(**max(0, violation distance)**) plus L2 regularization (the original SVM minimization objective) .  
TODO: Add image 
### When and Where
- Soft margin SVM is the first practical SVM model which could handle most of the classfication problem effectively and efficiently. In LIBSVM and some other packages, it is usually called **C-SVM**
### Cons
- Soft margin SVM can only be used in binary classification. 
- **For multi-class classification problem**, one needs to apply **ovo(one versus one) or ova(one versus all)** approach with Soft margin SVM
- **For soft binary classificiation problem (when user needs to know a classification probability)**, one needs to use **Probabilistic SVM** or **Kernel Logistic Regression**
- **For regression problem**, one needs to use **Support Vector Regression** or **Kernel Ridge Regression**
## [Probabilistic SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/205_handout.pdf)
### What
### Why
### How
### When and Where
### Cons
## [Support Vector Regression](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/206_handout.pdf)
### What
### Why
### How
### When and Where
### Cons
## [Summary](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/206_handout.pdf)
TODO: add image
## [Appendix LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
### Paper and Guide
[LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)  
[LIBLINEAR: A Library for Large Linear Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf)  
[A Practical Guide to Support Vector Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)  
