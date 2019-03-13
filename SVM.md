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
- Simple but powerful model with stable performance for mid-size data set
- Better generalization performance than other linear model because of large margin (large margin -> limited number of hyperplanes -> smaller VC dimension -> good resistance to overfitting)
- Fast to train (using LIBLINEAER, LIBSVM, or sklearn.SVC)
- Linear model -> easy to deploy and predict
- Strictly convex problem -> easy to find the unique optimal numerically. Comparing with SVM, neural networks is full of local optimals
### How
- Let the decision boundary hyperplane be **<w, x> + b = 0** 
- Compute the distance between a data instance and the hyperplane
  - **w** is the normal vector of the hyperplane 
  - **v** = **x** - **x'** (the difference vector between x and its intersection x' with the hyperplane)
  - distance = |<w, v>| / ||w||  
             = |<w, (x - x')>| / ||w||   
             = |<w, x> - <w, x'>| / ||w||  
             = |<w, x> + b - (<w, x'> + b)| / ||w||  
             = |<w, x> + b| / ||w||  
             = y(<w, x> + b) / ||w||  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_distance.png)
- Our goal is to get a hyperplane with largest margin, and we can convert it into a optimization problem with constraints  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_hard_margin_initial_problem.png)
- Let the margin goal be 1 (by scaling w and b, margin goal can be adjusted to 1 from any other constant)  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_hard_margin_one.png)
- The optimization problem can be solved easier under a looser constraint, and its optimal weight and bias still meets the original constraints (this can be justified through  contradiction)  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_hard_margin_loose_constraint.png)
- Finally, solve an equivalent quadratic programming problem with existing packages  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_hard_margin_qp.png)
### When and Where
- Training data are linear separable (but you do not know this info before training)
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
- In a higher dimensional space (could even be an infinite dimensional space), the data could be linear separable, and due to large margin, the VC dimension of the model is still limited. Therefore, we can get a sophisticated decision boundary with not so many hypothesis to choose.
- But the complexity of solving the primal QP problem depends on the number of features, which means higher dimensional feature would cause much higher complexity to solve the problem, sometimes even impossible, e.g. infinite dimension
- The solution is to solve its Lagrange dual problem, because it transforms to-be-optimized variables from weight and bias to the Lagrange multipliers of data instances, i.e. from very high (or inifite) dimension to rather low (or finite) dimension
### How
- Use function phi to map current feature vector into higher dimensional space  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_phi_mapping.png) 
- Create the Lagrange function and transform the primal problem into an equivalent **min-max** optimization problem without constraints (actually Lagrange multipliers alpha >= 0 become the new constraints of the Lagrange functions)   
- Reason for the equivalence: 
   - **For those weight and bias which generate a positive constraint (those do not fulfill the constraints)**, the maximization will make the inner function explode to positive infinite，which in turn will make these weight and bias eliminated by the outer minimization step
   - **For those weight and bias which generate a constraint equals zero**, alpha can be any value which is >= 0(these corresponding data instances are the **Support Vector**), and only the original minimization objective will be kept
   - **For those weight and bias which generate a negative constraint**, the maximization will make alpha equals 0, which in turn will also keep the original minimization objective
   - Both the zero and the negative case form the **complementary slackness** condition in **KKT optimality condtion** below, i.e. either alpha or the constraint (or both) needs to be zero 
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_dual.png)
- When below conditions are met, the optimal of the primal min-max problem is equal to the one of the dual max-min problem.
  - primal problem is convex
  - constraints are linear
  - primal problem is feasible (linear separable in the mapped high dimensional space)   
- Take gradient of Lagrange function with respect to weight, bias, and alpha, set these gradient to zero (condition for optimal solution) to get some relations between variables and some new constraints, and plug them back into the dual max-min problem to get a simple maximization QP problem  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_dual_gradient_b.png)
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_dual_gradient_w.png)
- Use QP package to solve the problem and get the optimal alpha  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_qp_solver.png)
- Use **KKT optimality condition** (the sufficient and necessary condition for optimality) to compute weight and bias:
  - primal feasible (not relevant with computing weight and bias)
  - dual feasible (not relevant with computing weight and bias)
  - dual-inner optimal -> **only support vectors' alpha are bigger than 0** -> only need support vector to compute weight
  - primal-inner optimal (also called **complementary slackness**) -> use any support vector to compute bias or take average on support vectors to avoid numerical errors  
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_kkt.png) 
### When and Where
- This is a theoretical middle step for more advanced model. No practical use case for it, see details in Cons below
### Cons
- This model successfully transforms the number of to-be-optimized variables from number of features (could be infinite in the mapped feature space) into number of data instances (could be large, but definitely finite). But it still does not completely solve the complexity issue of handling high dimensional features, because there is an inner product between two high dimensional feature vectors needs to be computed in the dual problem, which leads to the next section **kernel SVM**
## [Kernel SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/203_handout.pdf)
### What
- Kernel function is a shortcut to compute inner product of high dimensional vectors by computing a function of the two corresponding original feature vectors
### Why
- With kernel function, SVM dual problem with high dimensional feature space could be solved within the time complexity of original features' lower dimensional space
### How
- The common kernel functions are shown below:
   - linear kernel   ![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_linear_kernel.png)
   - polynomial kernel  ![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_polynomial_kernel.png)
   - rbf (gaussian) kernel ![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_rbf_kernel.png)
- The necessary and sufficient condition for valid kernel functions is **Mercer's condition** -> K matrix needs to be postive semi definite, but some kernel function which does not meet Mercer's condition may also work in practice
### When and Where
- Equipped with kernel function, SVM dual problem is the first practical **hard margin SVM**
- The linear kernel does not map feature vector at all, so it is valid for those scenarios mentioned in [SVM primal problem](#svm-primal-problem), and it should be tried first, but normally with **soft margin SVM** shown in the next section
- According to the [LIBSVM practical guide](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf), one should try rbf kernel after trying out linear model, because it has just one hyperparameter (gamma) to tune and it maps feature into infinite dimensional space for linear separation
- The polynomial kernel should be tried with lower degree to resist overfitting
### Cons
- Linear kernel is safe but restricted, esp. with hard margin SVM
- Rbf kernel is mysterious, slow, and maybe too powerful
- Polynomial kernel has 3 hyperparameters to tune, and numerically difficult for high degree
- Hard margin SVM is still not good enough, because it cannot tolerate any data instance violating the margin, causing one to use an unnecessary high dimensional space for separating instances linearly, which in turn may generate a too sophisticated (easy to overfit) decision boundary. The solution to mitigate the issue is the next section **Soft margin SVM**.
## [Soft margin SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/204_handout.pdf)
### What
A kernel SVM which can tolerate some data instances violating the margin (even violating the decision boundary)
### Why
The reason has been described in the Cons of kernel SVM above
### How
- Add Xi (violation distance, which must be >= 0) of each data instance to the minimization objective in SVM primal problem, and adjust the constraints accordingly
- A hyperparameter **C** is added as the coefficient of the violation part in the minimization objective
   - A smaller C means the model can tolerate more violation
   - The default value of C is usually set to 1
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_soft_margin_primal.png)
- Similar as hard margin SVM, transform the soft margin SVM primal problem to dual problem and simplify it
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_soft_margin_dual.png)
- These two documents provide the clue of solving this QR problem:
   - [Replace alpha with beta to simplify the QR problem from 2N variables to N variables](https://www.ics.uci.edu/~welling/classnotes/papers_class/SVregression.pdf)
   - [Transform a convex QR problem containing absolute value into a standard convex QR problem](https://scicomp.stackexchange.com/questions/28035/reformulate-a-strictly-convex-qp-problem-containing-absolute-value-term)
- After training, there are 3 types of data instances:
   - **alpha == 0 -> non-support vectors**, i.e. data instances which are classified correctly by the decision boundary and do not lie on the margin. These data instances are not useful in prediction, thus can be abandoned after training
   - **alpha == C -> bounded support vectors**, i.e. data instances which violates the margin. They are useful in computing the weight-feature inner product for prediction when kernel function is used, therefore must be kept after training
   - **0 < alpha < C -> free support vectors**, i.e. data instances which are classified correclty and lie on the margin. User must choose any one of these data instances to compute the bias when kernel function is used.
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_alpha.png)
- When kernel function is used, weight cannot be computed directly (because the feature mapping is done implicitly, and the mapped space could even be infinite dimensional space), therefore the prediction is computed with support vectors (both bounded and free ones) and bias is computed by any free support vectors
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_dual_predict.png)
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_dual_solve_b.png)
- Soft margin SVM can be viewed as a linear model error measurement **max(0, violation distance)** plus L2 regularization (the original SVM minimization objective) 
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_soft_margin_l2_regularized.png)
### When and Where
- Soft margin SVM is the first practical SVM model which could handle the classification problem effectively and efficiently. In LIBSVM and some other packages, it is usually called **C-SVC(C-Support Vector Classification)**
### Cons
- Soft margin SVM can only be used in binary classification. 
- **For multi-class classification problem**, one needs to apply **ovo(one versus one) or ova(one versus all)** approach with Soft margin SVM
- **For soft binary classification problem**(when user needs to know a classification probability), one needs to use **Probabilistic SVM** or **Kernel Logistic Regression**
- **For regression problem**, one needs to use **Support Vector Regression** or **Kernel Ridge Regression**
## [Probabilistic SVM](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/205_handout.pdf)
### What
- A soft binary SVM-based classifier which could predict class probability instead of a hard 0/1 classification  
### Why
- This model combines both flavor of SVM and logistic regression. Comparing with **Kernel Logistic Regression**, the solution of Probabilistic SVM is often sparse.
### How
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/probabilistic_svm.png)
### When and Where
- This model is used when one needs to know the class probability and wants to use kernel SVM to get a sparse solution
### Cons
- Probabilistic SVM is to estimate logistic regression in Z space, not to perform exact logistic regression in Z space
### Kernel Logistic Regression
- **Representer Theorem**: For any L2-regularized linear model, optimal weight can be represented by a linear combination of feature vectors
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/representer_theorem.png)
- According to Representer Theorem, by replacing weight with linear combination of feature vectors in the objective function of the L2 regularized logistic regression, one can develop a kernel version logistic regression
- The cons of this model is the solution is dense, therefore it is not as practical as Probabilistic SVM
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/kernel_logistic_regression.png)
## [Support Vector Regression](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/206_handout.pdf)
### What
- A L2-regularized linear regression model with **Tube error function(|<w, z> - y| - epsilon >= Xi, Xi >= 0)**, which provides the benefits similar as soft margin SVM 
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svr_tube.png)
### Why
- Comparing with Kernel Ridge Regression, this model has similar benefits of SVM, e.g. kernel function, sparse solution, good tolerance of outlier, convex optimization, and so on
### How
- Similar as SVM, Support Vector Regression is solved through dual problem and quadratic programming 
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svr_dual.png)
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svr_qr_solver.png)
- **Support Vectors of SVR** are those data instances which are on or out of the tube
- Weight is computed by just support vectors
- Bias can be computed by the support vectors on the tube, i.e. free support vectors, using formula b = y - <w, z> - epsilon. If there is no free support vectors, b can only be estimated by all data instances. See more detail in the [Support Vector Regression tutorial](https://alex.smola.org/papers/2004/SmoSch04.pdf) 
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svr_sparsity.png)
### When and Where
- When vanilla L2-regularized linear regression could not get a good performance, SVR is worth to try
### Cons
- [Support Vector Regression tutorial](https://alex.smola.org/papers/2004/SmoSch04.pdf) documents some area for SVR to improve, e.g. number of support vectors needs to be reduced in the context of big data
### Kernel Ridge Regression (Least-Square SVM)
- Similar as Kernel Logistic Regression, one can get a kernel version L2-regularized linear regression based on Representer Theorem too
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/kernel_ridge_regression_problem.png)
- Like linear regression, Kernel Ridge Regression can also be solved analytically, but the training time is O(N^3) (N is the number of training data instances) and the predict time is O(N), which would be pretty hard for big data.   
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/kernel_ridge_regression_solver.png)
## [Summary](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/206_handout.pdf)
- Horizontally there are 3 types of linear models:
    - Binary classifier
    - Soft binary classifier (giving probability)
    - Regression model
- Vertically, models are evolved from vanilla linear models to kernel and further to SVM
- Linear model (w/o kernel function) should always be tried first, and only if they are not good enough, SVM and kernel model should then be tried out
![alt_text](https://github.com/WangYuanMike/NTU_ML_Foundation/blob/master/SVM/svm_summary.png)
## [Appendix LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
### SVM models implemented in LIBSVM
- **C-SVC**: soft margin SVM
- **nu-SVC**: soft margin SVM with a parameter nu to control training error and number of support vectors
- **Distribution Estimation(one-class SVM)**: training data all belong to one class, and this model use SVM to train a decision boundary through origin so as to estimate distribution, i.e. whether a data belongs to this class
- **epsilon-SVR**: support vector regression
- **nu-SVR**: support vector regression with parameter nu to control training error and number of support vectors
### Unbalanced data
- Using different penalty parameters C- and C+ for two classes is implemented in LIBSVM
### Probability Estimate
- Probability Estimate is different from Distribution Estimate
- Probability Estimate is to estimate the probability of being one class or the other, which is an implementation of Probabilistic SVM
- Distribution Estimate is to build a function that takes the value +1 in a small region capturing most of the training data instances (all in one class, and denote the class with +1), and -1 elsewhere. This is an unsupervised learning approach which is normally used for outlier detection. See more detail in [this paper](https://papers.nips.cc/paper/1723-support-vector-method-for-novelty-detection.pdf)
### Multi-class classification
- Use ovo(one versus one) approach to elect champion among multiple classes
### Proposed procedure
- Read the [Practical Guide](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) to get start
### Paper
[LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)  
[LIBLINEAR: A Library for Large Linear Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf)  
