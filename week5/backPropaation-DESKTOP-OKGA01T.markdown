# BackPropagation

https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm

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
1. Perform Forward Propagation -> Result is going be in the size of (m * K )

  Example of forward propagation   
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

#### 1 Calculating $\frac{\partial E}{\partial z^{L-2}} = \delta^{L}$

It is important to note that $\delta$ is pretty much $\frac{\partial E}{\partial z}$ for all z.  
Calculating  $\delta^{L}$ is different from calculating $\delta^{L-1}$ $\delta^{L-2}$ ... $\delta^{2}$

1. You have to calculate $\frac{\partial E}{\partial a}$
2. Then you have to calculate
$\frac{\partial E}{\partial a^{L}} = \frac{y}{a^{L}}+\frac{(1-y)}{(1-a^{L})}$ and
 $\frac{\partial a}{\partial z^{L}}=(a*(1-a))=\sigma(z)(1-\sigma(z))$
 _<div style="width:50%">![](sigmoidDerivative.png)</div>_  

3. $\frac{\partial E}{\partial z^{L}}=\frac{\partial E}{\partial a^{L}}* \frac{\partial a^{L}}{\partial z^{L}}= \delta^{(L)} = a^{L}* (1-a^{L})* (\frac{y}{a^{L}}+\frac{(1-y)}{(1-a^{L})})=a^{(L)} - y$



#### 2 Calculating $\frac{\partial E}{\partial z^{L-1}}=\delta^{(L-1)}, \frac{\partial E}{\partial z^{L-2}}=\delta^{(L-2)},\dots, \frac{\partial E}{\partial z^{L-2}}=\delta^{(2)}$ GENERAL CASE

**Reminder: In each level l  derivative we can have multiple  $\delta^{l}$, like we have multiple  $z^{l}$**  

Calculation is  done somewhat recursively. For every smaller level $\delta^{l}$  we need to use  $\delta^{l+1}$ in our calculation of the derivative because of the chain derivative rule: $\frac{d}{{dx}}\left[ {f\left( u \right)} \right] = \frac{d}{{du}}\left[ {f\left( u \right)} \right]\frac{{du}}{{dx}}$ . If, inside the formula we go further from the output towards the input we need to always take functions in between under consideration.


**In other words**

$\frac{\partial E}{z^{l}}=\frac{\partial E}{z^{l+1}}$ **$\frac{\partial z^{l+1}}{ a^{l}} \frac{\partial a^{l}}{z^{l}}$**

**This translates to**

$\frac{\partial E}{\partial Z^{l}} = \delta^{(l)} =$ $((\Theta^{(l)})^T \delta^{(l+1)})$ $\ .* \ a^{(l)}\ .* \ (1 - a^{(l)})$  

**Where each part stands for**  
+ $((\Theta^{(l)})^T \delta^{(l+1)})=\frac{\partial z^{l+1}}{\partial a^{l}}$
    + This rule is somewhat more interesting, because it applies a somewhat complicated concept where 1 variable in a function affects the end result through multiple other functions. E.g $E=E(d(x),u(x))$ Such a function, when taking a derivative is solved by just summing up all the derivatives. **This is somewhat intuitive but I wont go to deep into it**   
    + Explanation in Estonian   
    ![explanation of the this concept](matAnal1.png)  
    ![explanation of the this concept](matAnal2.png)
+ $\ a^{(l)}\ .* \ (1 - a^{(l)}) = \frac{\partial a^{l}}{z^{l}}$

### 4. Step - Calculating $\frac{\partial E}{\partial \theta^{l}}$

Reminder
+ $z=\theta a$
+ 1 $\theta$ has effect only on 1 z

$\frac{\partial E}{\partial \theta^{l}_{ij}}=\frac{\partial E}{z^{l+1}_{i}} \frac{\partial z^{l+1}_{i}}{\partial \theta^{l}_{ij}}$


**This in regular derivative form translates to**

$\frac{\partial E}{\partial \theta^{l}_{ij}}=\delta^{l+1}_{i}a^{l}_{j}$

**This translates in vectorial form to**  

$\delta^{(l+1)}(a^{(l)})^T$  results in i x j matrix, like $\theta$













## How to Check whether Gradient is correct.
