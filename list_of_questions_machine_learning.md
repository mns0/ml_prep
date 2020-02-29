# Machine Learning - List of questions

## Learning Theory

1. Describe bias and variance with examples.
	* Bias error is the result of erroneous assumptions by the machine learning algorithm. High bias will cause the algorithm to miss the relationships between features in data and target outputs. (Underfitting)
	* Variance is the extent of noise in the data. Fitting to noise in the data will cause the model to overfit. 
	* Formally the variance-bias decomposition is, 
	$$
	MSE(x_0) = E_T[f(x_0) - \hat{y}_0]^2
	= E_T[\hat{y}_0 - E_T[\hat{y}_0]]^2 + E_T[E_T[\hat{y}_0] - f(x_0)]^2
	= Var_T(\hat{y}_0) + Bias^2(\hat{y}_0)
	$$
	
	Where the bias is the exception of the distance the prediction is from the actual result. 



1. What is Empirical Risk Minimization? 
	* ERM is ...
1. What is Union bound and Hoeffding's inequality?
	* Union bound:
	    * Let $$F_1$$ and $$F_2$$ be independent events. Suppose that $$P[F_1] \geq 1 - p_1$$ and $$P[F_2] \geq 1 - p_2$$. Then  $$P[F_2] \geq 1 - p_2$$, due to independence.
	* Hoeffding's inequality:
		*  Let $$Z_i,...,Z_n$$ be IRV (Independent random variables), s.t. $$0\leq Z_i  \ leq 1 $$,

		$$
		P[\mid   \frac{1}{n} \sum_{i=1}^{n} Z_i-E[Z]  \mid~\leq~\delta~] =  2e^{-2n\epsilon^2} 
		$$


	Provides an upper bound on a binomial probability, given $$\epsilon $$ and $$n$$. The trade-off is certain distributions may converge much faster than the binomial distribution 
	[Great Resource on Concentration Bounds](https://people.cs.umass.edu/~domke/courses/sml2010/10theory.pdf)

1. Write the formulae for training error and generalization error. Point out the differences.
	* Generalization error, or test error (out-of-sample error), is prediction error over an independent test sample.

	$$
	Err_T = E[L(Y,\hat{f}(X))\mid T]
	$$
	
	where $$X$$ and $$Y$$ are drawn randomly from a joint distribution (population) and the training set $$T$$ is fixed.
	The training error is defined as the average loss over a training sample.
	$$
	\frac{1}{N}\sum_{i=1}^{N} L(y_i, \hat{f}(x_i))
	$$
	
1. State the uniform convergence theorem and derive it.
1. What is sample complexity bound of uniform convergence theorem? 
1. What is error bound of uniform convergence theorem? 
1. What is the bias-variance trade-off theorem? 
1. From the bias-variance trade-off, can you derive the bound on training set size?
1. What is the VC dimension? 
1. What does the training set size depend on for a finite and infinite hypothesis set? Compare and contrast. 
1. What is the VC dimension for an n-dimensional linear classifier? 
1. How is the VC dimension of a SVM bounded although it is projected to an infinite dimension? 
1. Considering that Empirical Risk Minimization is a NP-hard problem, how does logistic regression and SVM loss work? 

## Model and feature selection
1. Why are model selection methods needed?
	* Model selection methods estimate the performance of various model to choose the best one which generalizes well
1. How do you do a trade-off between bias and variance?
	* Assuming $$ Y = f(X) + \epsilon$$ which $$ E(\epsilon)~=~0$$ and $$Var(\epsilon) = \sigma_\epsilon^2$$~we can derive an expression for our error as
	
	$$
	Err(x_0) = E[Y - \hat{f}(x_0)^2\mid X = x_0]$$
	$$= \sigma_\epsilon^2 + [E[\hat{f}(x_0)] - f(x_0)]^2 +    E[\hat{f}(x_0) -  E[\hat{f}(x_0)]]^2$$
	$$= sigma_\epsilon^2 + Bias^2 + Variance
	$$
	
	Variance refers to the model variance and the square bias is the average which the estimate differs from the true mean. Typically the more complex the model becomes, the square bias decreases and the variance increases.
	
1. What are the different attributes that can be selected by model selection methods?
1. Why is cross-validation required?
1. Describe different cross-validation techniques.
1. What is hold-out cross validation? What are its advantages and disadvantages?
1. What is k-fold cross validation? What are its advantages and disadvantages?
1. What is leave-one-out cross validation? What are its advantages and disadvantages?
1. Why is feature selection required?
1. Describe some feature selection methods.
1. What is forward feature selection method? What are its advantages and disadvantages?
1. What is backward feature selection method? What are its advantages and disadvantages?
1. What is filter feature selection method and describe two of them?
1. What is mutual information and KL divergence?
1. Describe KL divergence intuitively.

## Curse of dimensionality 
1. Describe the curse of dimensionality with examples.
1. What is local constancy or smoothness prior or regularization?

## Universal approximation of neural networks
1. State the universal approximation theorem? What is the technique used to prove that?
1. What is a Borel measurable function?
1. Given the universal approximation theorem, why can't a MLP still reach a arbitrarily small positive error?

## Deep Learning motivation
1. What is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
1. In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?
1. What are the reasons for choosing a deep model as opposed to shallow model? (1. Number of regions O(2^k) vs O(k) where k is the number of training examples 2. # linear regions carved out in the function space depends exponentially on the depth. )
1. How Deep Learning tackles the curse of dimensionality? 

## Support Vector Machine
1. How can the SVM optimization function be derived from the logistic regression optimization function?
1. What is a large margin classifier?
1. Why SVM is an example of a large margin classifier?
1. SVM being a large margin classifier, is it influenced by outliers? (Yes, if C is large, otherwise not)
1. What is the role of C in SVM?
1. In SVM, what is the angle between the decision boundary and theta?
1. What is the mathematical intuition of a large margin classifier?
1. What is a kernel in SVM? Why do we use kernels in SVM?
1. What is a similarity function in SVM? Why it is named so?
1. How are the landmarks initially chosen in an SVM? How many and where?
1. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
1. What is the difference between logistic regression and SVM without a kernel? (Only in implementation – one is much more efficient and has good optimization packages)
1. How does the SVM parameter C affect the bias/variance trade off? (Remember C = 1/lambda; lambda increases means variance decreases)
1. How does the SVM kernel parameter sigma^2 affect the bias/variance trade off?
1. Can any similarity function be used for SVM? (No, have to satisfy Mercer’s theorem)
1. Logistic regression vs. SVMs: When to use which one? 
( Let's say n and m are the number of features and training samples respectively. If n is large relative to m use log. Reg. or SVM with linear kernel, If n is small and m is intermediate, SVM with Gaussian kernel, If n is small and m is massive, Create or add more fetaures then use log. Reg. or SVM without a kernel)

## Bayesian Machine Learning
1. What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
1. Compare and contrast maximum likelihood and maximum a posteriori estimation.
1. How does Bayesian methods do automatic feature selection?
1. What do you mean by Bayesian regularization?
1. When will you use Bayesian methods instead of Frequentist methods? (Small dataset, large feature set)

## Regularization
1. What is L1 regularization?
1. What is L2 regularization?
1. Compare L1 and L2 regularization.
1. Why does L1 regularization result in sparse models? [here](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)

## Evaluation of Machine Learning systems
1. What are accuracy, sensitivity, specificity, ROC?
1. What are precision and recall?
1. Describe t-test in the context of Machine Learning.

## Clustering
1. Describe the k-means algorithm.
1. What is distortion function? Is it convex or non-convex?
1. Tell me about the convergence of the distortion function.
1. Topic: EM algorithm
1. What is the Gaussian Mixture Model?
1. Describe the EM algorithm intuitively. 
1. What are the two steps of the EM algorithm
1. Compare GMM vs GDA.

## Dimensionality Reduction
1. Why do we need dimensionality reduction techniques? (data compression, speeds up learning algorithm and visualizing data)
1. What do we need PCA and what does it do? (PCA tries to find a lower dimensional surface such the sum of the squared projection error is minimized)
1. What is the difference between logistic regression and PCA?
1. What are the two pre-processing steps that should be applied before doing PCA? (mean normalization and feature scaling)

## Basics of Natural Language Processing
1. What is WORD2VEC?
1. What is t-SNE? Why do we use PCA instead of t-SNE?
1. What is sampled softmax?
1. Why is it difficult to train a RNN with SGD?
1. How do you tackle the problem of exploding gradients? (By gradient clipping)
1. What is the problem of vanishing gradients? (RNN doesn't tend to remember much things from the past)
1. How do you tackle the problem of vanishing gradients? (By using LSTM)
1. Explain the memory cell of a LSTM. (LSTM allows forgetting of data and using long memory when appropriate.)
1. What type of regularization do one use in LSTM?
1. What is Beam Search?
1. How to automatically caption an image? (CNN + LSTM)

## Miscellaneous
1. What is the difference between loss function, cost function and objective function?
