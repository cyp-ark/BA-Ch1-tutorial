# Perplexity 값의 변화에 따른 t-SNE 모양 양상

## 0. Introduction
데이터 분석을 하다보면 하나의 샘플에 대해 여러 변수들이 측정되는 것을 볼 수 있다. 적게는 변수 하나가 될 수도 있으나 많은 경우 수천개 이상의 변수들로 하나의 샘플이 표현 되기도 한다. 하지만 우리가 인지할 수 있는 차원은 3차원 정도이고, 한눈에 데이터의 분포등을 잘 이해할 수 있는 차원은 2차원 평면 정도이기 때문에 만약 우리가 4개 이상의 변수들로 이루어진 샘플들을 시각화 하려 한다면 그 데이터들 자체로써는 표현하기가 불가능 할 것이다. 그렇기 때문에 우리는 고차원의 데이터를 우리가 인지할 수 있는 저차원의 평면으로 변형시켜야 각 샘플 별 특징이나 샘플 간 거리, 유사도 등을 한눈에 파악 할 수 있다.
<br/><br/>
고차원의 데이터를 저차원으로 변형 시키는 것을 차원 축소(dimensionality reduction)이라 한다. 그러나 고차원의 데이터를 저차원으로 차원축소하는 과정에서는 필연적으로 고차원의 데이터가 가지고 있는 정보에 대한 손실이 발생하게 된다. 그렇기 때문에 모든 차원 축소 방법론들은 차원 축소 과정에서 손실되는 정보를 최소화하는 방식으로 진행되고, 그 과정에서 기존 데이터들의 거리 또는 유사성을 최대한 보존하는 것을 목표로 한다.
<br/><br/>
일반적으로 데이터 시각화에 많이 사용되는 차원축소 방법론에는 주성분 분석(PCA, Principal Components Analysis)와 t-SNE(t-distributed Stochastic Neighbor Embedding)이 있고, 최근에는 UMAP(Uniform Manifold Approximation and Projection)이라는 기법이 떠오르고 있다. 그 중 우리는 이번 튜토리얼에서 확률론적 방법을 이용하면서 주변 데이터들의 거리를 보존하는 방법인 t-SNE에 대해 알아보고자 한다.

## 1. t-SNE(t-distributed Stochastic Neighbor Embedding)
t-SNE는 2008년 JMLR(Jounal of machine learing research)에 소개 된 방법론으로 기존의 방법론인 LLE(Locally Linear Embedding)에서의 선형적인 변환을 확률론적으로 발전시킨 방법이다. LLE는 어떠한 점을 고차원에서 저차원으로 변형할 때 그 점 주변의 이웃들의 선형결합으로 자신을 표현하고, 이후 저차원에서 그 선형결합의 조합을 이용해 점들을 다시 복원시키는 방법이다. t-SNE는 이웃들을 선형결합이 아닌 자신 주변에 분포해 있을 확률로 표현을 하고 그 확률을 보존하는 방식으로 저차원에서 데이터를 복원 시킨다. 이를 수식으로 나타내면 다음과 같다.

$$
p_{j|i} = { e^{-  {||x_{i}-x_{j}||^{2} \over 2\sigma_{i}^{2}}} \over \sum_{k\neq i} e^{ -{||x_{i}-x_{k}||^{2} \over 2\sigma_{i}^{2}} }}
$$

$$
q_{j|i} = { e^{-||x_{i}-x_{j}||^{2}} \over \sum_{k\neq i} e^{ -||x_{i}-x_{k}||^{2} }}
$$

$p_{j|i}$는 고차원에서 점 $i$에 대한 점$j$의 확률을 나타내고, $q_{j|i}$는 저차원에서 점 $i$에 대한 점 $j$의 확률을 나타낸다. 앞서 언급한대로 고차원의 데이터를 저차원의 데이터로 차원축소하는 과정에서 정보의 손실은 필연적인데, 이를 KL divergence(Kullback-Leibler divergence)라는 함수를 이용해 표현하고자 한다. KL divergence는 두 분포간의 정보의 차이를 계산한다. 우리는 이를 이용해 고차원에서의 분포와 저차원에서의 분포 간의 정보의 차이를 계산하고 이를 최소화 하는 방식으로 차원축소를 진행한다.

$$
D_{KL}(P|Q) = \sum_{i}\sum_{j}p_{j|i} log{p_{j|i} \over q_{j|i}}
$$

<br/>

고차원 데이터의 분포에 대한 식을 다시한번 잘 살펴보면 $p_{j|i}$는 계산되는 값이고 $x_{i},x_{j}$는 데이터를 통해 주어진 값이다. 그렇다면 수식 중 $\sigma_{i}^{2}$만이 남게 되는데 이는 주어진 값인지, 계산되는 값인지, 아니면 사용자가 지정해줘야 되는 값인지 정확히 파악이 되지 않는다. 이는 사용자가 지정해줘야하는 hyperparameter로 이후 내용인 perplexity와 연관이 있다.

## 2. Perplexity

주변의 이웃값을 이용하는 모델들은 필연적으로 그 이웃의 범위를 설정해줄 필요가 있다. 예를들어 분류문제에서의 KNN(K-Nearest Neighborhood)나 군집분석에서의 K-means Clustring 등이 있다. 물론 이웃의 개수에 대한 정보를 전적으로 사용자에게 맡기는 것이 아닌 어느정도 범위의 값을 설정해두고 모델의 결과값을 비교해 그 중 가장 결과가 좋은 값을 선택하는 방식으로 설정해준다. t-SNE의 경우 perplexity라는 hyperparameter를 이용해 주변 이웃의 범위를 결정하게 되는데 이를 수식으로 나타내면 다음과 같다.

$$
Perplexity(P_{i}) = 2^{H(P_{i})}
$$

$$
H(P_{i}) = \sum_{j} p_{j|i}log_{2}p_{j|i}
$$

방법론을 제안한 연구자들에 의하면 t-SNE의 성능은 perplexity 값의 변화(5~50)에 강건하다고 한다. 그렇다면 perplexity값이 실제 t-SNE를 이용한 차원축소 과정에서 저자들의 주장대로 그 값이 강건한지, 그리고 Perplexity 값이 실제 모델에 어떻게 영향을 주는지 이번 튜토리얼을 통해 파악하고자 한다.

## 3. Code
Scikit-learn의 TSNE 모듈을 이용해 실습을 진행하고자 한다.  
튜토리얼에 이용할 샘플을 만들고 시각화하면 다음과 같다.
```python
#Sample 생성

a = 2
rv1 = sp.stats.multivariate_normal([+a, +a], [[1, 0], [0, 1]])
rv2 = sp.stats.multivariate_normal([+a, -a], [[1, 0], [0, 1]])
rv3 = sp.stats.multivariate_normal([-a, +a], [[1, 0], [0, 1]])
rv4 = sp.stats.multivariate_normal([-a, -a], [[1, 0], [0, 1]])
#%%
X0 = rv1.rvs(100)
X1 = rv2.rvs(100)
X2 = rv3.rvs(100)
X3 = rv4.rvs(100)

X = np.vstack([X0,X1,X2,X3])
Y = np.hstack([np.zeros(100), np.ones(100),np.ones(100)*2,np.ones(100)*3])
```


![](https://github.com/cyp-ark/BA-Ch1-tutorial/blob/main/sample.png)


```python
#Perplexity 설정 및 t-SNE 연산

n_perplexity = [1,2,3,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,175,200]
color = ['red','green','blue','orange']

X_t = np.zeros(shape=(len(n_perplexity),len(X),2))

for i in range(len(n_perplexity)):
    X_t[i] = TSNE(n_components=2,perplexity=n_perplexity[i],random_state=0).fit_transform(X)
```




## 4. Results
Perplexity를 1부터 50까지 변화해가면서 t-SNE를 진행한 후 시각화를 진행한 결과 perplexity 값이 약 10 전후부터 안정적인 모양이 나오기 시작했으며, 그 후 크게 움직임이 없었다. 물론 데이터셋이 다를 경우 그 결과가 다를 수 있겠으나 그 데이터 셋의 크기가 충분히 클 경우 일정 값 이상부터는 robust한것으로 보여진다. 다만 TSNE 모듈에서 기본적으로 perplexity=30으로 설정되어있는데, 상대적으로 크기가 작은 데이터셋의 경우 이 값을 조절할 필요가 있어보인다.

![](https://github.com/cyp-ark/BA-Ch1-tutorial/blob/main/perplexity.gif)

## 5. Reference

Van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of machine learning research 9.11 (2008).[[Link]](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl)
Wattenberg, Martin, Fernanda Viégas, and Ian Johnson. "How to use t-SNE effectively." Distill 1.10 (2016): e2.[[Link]](https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600)
