# BackPropagation

https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm


<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [BackPropagation](#backpropagation)
	- [Algorithm](#algorithm)
		- [Definitions](#definitions)
		- [Steps](#steps)
	- [Derivatives](#derivatives)
		- [3. Step - Calculating $\frac{\partial E}{\partial z^{l}} = \delta^{l}$](#3-step-calculating-fracpartial-epartial-zl-deltal)
			- [1 Calculating $\frac{\partial E}{\partial z^{L}} = \delta^{L}$](#1-calculating-fracpartial-epartial-zl-deltal)
			- [2 Calculating $\frac{\partial E}{\partial z^{L-1}}=\delta^{(L-1)}, \frac{\partial E}{\partial z^{L-2}}=\delta^{(L-2)},\dots, \frac{\partial E}{\partial z^{2}}=\delta^{(2)}$ GENERAL CASE](#2-calculating-fracpartial-epartial-zl-1deltal-1-fracpartial-epartial-zl-2deltal-2dots-fracpartial-epartial-z2delta2-general-case)
		- [4. Step - Calculating $\frac{\partial E}{\partial \theta^{l}}$](#4-step-calculating-fracpartial-epartial-thetal)
	- [How to Check whether Gradient is correct.](#how-to-check-whether-gradient-is-correct)

<!-- /TOC -->


**Why it is important to know this**

https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

## Algorithm

### Definitions

+ K = number of output units    
+ L = number of layers in network    
+ $s_{l} =$ number of units  in layer l  
+ m = number of training examples  
+ E - another symbol for cost function J
+ $\lambda$ - regularization term   


### Steps


Initialize $\Delta^{l}_{ij}=0,  \forall ij\text{ and }l$

**REPEAT FOR $1\dots m$**
1. Perform Forward Propagation. This can be done using the training matrix as well. Result is going be in the size of (m * K ).

**Example of forward propagation**   
![](forwardProp.png)

2. Calculate the cost
    + Definitions
        + $\theta$ - Hypothesis function parameters
        + $h_{\theta}()_ {k}$ Hypothesis function
        + $x^{(i)}$  i-th training data
    + $J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_ k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_ k)\right] + \frac{\lambda}{2m}\sum_ {l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2$

3. Calculate derivatives of Cost Function   according to every  $z^{l}_{s}$  $\frac{\partial E}{\partial z^{l}_{s}}$  
4. Calculate derivatives for every theta
5. Add cumulatively to $\Delta$    
 $\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$ or in vectorial form  $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$  

**LOOP**

Update $D^{l} = \frac{1}{m} \Delta^{l} + \lambda \theta$



## Derivatives

### 3. Step - Calculating $\frac{\partial E}{\partial z^{l}} = \delta^{l}$

#### 1 Calculating $\frac{\partial E}{\partial z^{L}} = \delta^{L}$

It is important to note that $\delta$ is pretty much $\frac{\partial E}{\partial z}$ for all z.  
Calculating  $\delta^{L}$ is different from calculating $\delta^{L-1}$ $\delta^{L-2}$ ... $\delta^{2}$

1.  You have to calculate
$\frac{\partial E}{\partial a^{L}} = \frac{y}{a^{L}}+\frac{(1-y)}{(1-a^{L})}$ and
 $\frac{\partial a}{\partial z^{L}}=(a^{L}* (1-a^{L}))=\sigma(z^{L})(1-\sigma(z^{L}))$
 _<div style="width:50%">![](sigmoidDerivative.png)</div>_  
 Here is the derivation for $\frac{\partial E}{\partial a}$

2. $\frac{\partial E}{\partial z^{L}}=\frac{\partial E}{\partial a^{L}}* \frac{\partial a^{L}}{\partial z^{L}}= \delta^{(L)} = a^{L}* (1-a^{L})* (\frac{y}{a^{L}}+\frac{(1-y)}{(1-a^{L})})=a^{(L)} - y$



#### 2 Calculating $\frac{\partial E}{\partial z^{L-1}}=\delta^{(L-1)}, \frac{\partial E}{\partial z^{L-2}}=\delta^{(L-2)},\dots, \frac{\partial E}{\partial z^{2}}=\delta^{(2)}$ GENERAL CASE

**Reminder: In each level l  derivative we can have multiple  $\delta^{l}$, like we have multiple  $z^{l}$**  

Calculation is  done somewhat recursively. For every smaller level $\delta^{l}$  we need to use  $\delta^{l+1}$ in our calculation of the derivative because of the chain derivative rule: $\frac{d}{{dx}}\left[ {f\left( u(x) \right)} \right] = \frac{d}{{du}}\left[ {f\left( u(x) \right)} \right]\frac{{du}}{{dx}}$ . If, inside the formula we go further from the output in layer L towards the input in layer 1, we need to always take functions in between under consideration.


**In other words**

$\frac{\partial E}{z^{l}}=\frac{\partial E}{z^{l+1}}$ **$\frac{\partial z^{l+1}}{ a^{l}} \frac{\partial a^{l}}{z^{l}}$** Here tha part in bold stands for the part thats being added.

**This translates to**

$\frac{\partial E}{\partial Z^{l}} = \delta^{(l)} =$ $((\Theta^{(l)})^T \delta^{(l+1)})$ $\ .* \ a^{(l)}\ .* \ (1 - a^{(l)})$  

**Where each part stands for**  
+ $((\Theta^{(l)})^T)=\frac{\partial z^{l+1}}{\partial a^{l}}$
+ $((\Theta^{(l)})^T \delta^{(l+1)})=\frac{\partial E}{\partial a^{l}}$ in $\theta^{l}$ i-th row a j-th we have $\frac{\partial z^{l+1}_{j}}{\partial a^{l}_{i}}$
    + This rule is somewhat more interesting, because it applies a somewhat complicated concept where 1 variable in a function affects the end result through multiple other functions. E.g $E=E(d(x),u(x))$ Such a function, when taking a derivative is solved by just summing up all the derivatives. **This is somewhat intuitive but I wont go to deep into it**   
    + Explanation in Estonian of this rule  
    ![explanation of the this concept](matAnal1.png)  
    ![explanation of the this concept](matAnal2.png)
+ $\ a^{(l)}\ .* \ (1 - a^{(l)}) = \frac{\partial a^{l}}{z^{l}}$

### 4. Step - Calculating $\frac{\partial E}{\partial \theta^{l}}$

Reminder
+ $z=\theta a$
+ 1 scalar $\theta$ has effect only on 1 z

$\frac{\partial E}{\partial \theta^{l}_{ij}}=\frac{\partial E}{z^{l+1}_{i}} \frac{\partial z^{l+1}_{i}}{\partial \theta^{l}_{ij}}$


**This in regular derivative form translates to**

$\frac{\partial E}{\partial \theta^{l}_{ij}}=\delta^{l+1}_{i}a^{l}_{j}$

**This translates in vectorial form to**  

$\delta^{(l+1)}(a^{(l)})^T$  results in i x j matrix, like $\theta$


## How to Check whether Gradient is correct.
