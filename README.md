## Simple Simulation Studies

In this experiment, we analyze whether the dependency between samples will impact the generalization. To this end, we consider the following experiment setup. 

For the control group, we consider a standard timeseries data generating process, let $X_{-k+1}, \ldots, X_1, \ldots X_T$ be a sequence of random variables satisfying
$$
X_i = g(X_{i-1}, \ldots, X_{i-k}) + U_i
$$
where $U_1,\ldots, U_T$ is a sequence of random variables, We consider the following least squares estimator
$$
\hat{f}_{c} = \mathrm{argmin}_{f\in \mathcal{F}} \frac{1}{T} \sum_{t=1}^T \{f(X_{i-1},\ldots, X_{i-p}) - X_i\}^2
$$
that targets to estimate $g$. Here we allow slight overparameterization, that $p\ge k$. 



For the experimental group, we consider that we fit $T$ observations, $(Z_i, Y_i)_{i=1}^T \overset{i.i.d.}{\sim} (Z,Y)$, the data generating process of $Z,Y$ is defined as follows
$$
I \sim \mathrm{uniform}\{1,\ldots, T\}, X_1,\ldots, X_T\text{ same as above },Z=(X_{I-1},\ldots, X_{I-p}), Y=X_I
$$
It runs the following least squares
$$
\hat{f}_e =\mathrm{argmin}_{f\in \mathcal{F}} \frac{1}{T}\sum_{i=1}^T\{f(Z_i)-Y_i\}^2
$$


The only difference in the two setups are whether the observations are independent or not. The marginal distributions of a single observation are the same. 

All the estimates are evaluated via prediction errors on another sequence of length $T$. 



The question is when the sequence is highly dependent, can we find some significant gap between the two methods, thus quantifying whether the dependency itself will lead to poor generalization.