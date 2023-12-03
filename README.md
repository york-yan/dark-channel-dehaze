# dark-channel-dehaze



------------

# 暗通道去雾（Kaiming He）

### 数学模型

$$
I(x)=J(x)t(x)+A(1-t(x))
\\
有雾图像I=无雾图像J×大气折射率t+大气光(1-大气折射率)
\\
x表示位置，需要求解的就是t和A
$$

### 暗通道先验条件

$$
J^{dark}(x)=\min_{c\in\{r,g,b\}}(\min_{y\in\Omega(x)}J^c(y))
\\
定义暗通道为以x为中心的一个区域内所有通道上的最小值，可以认为J^{dark}(x)\to0
$$

### 暗通道去雾

$$
无雾图像的复原公式为：J(x)=\frac{I(x)-A}{t(x)}+A
\\
增加一个强制下限t_0，避免t\to0时失效:J(x)=\frac{I(x)-A}{\max(t(x),t_0)}+A
$$

### 估计大气光强A

暗通道去雾的精髓在于对大气光强A的估计：**选择暗通道图像中前0.1％的像素**，暗通道的先验条件保证了挑选的像素不会受到自然图像景物的干扰，找到在原始图像中对应的这0.1%的像素的每个通道的最亮点，将其作为大气光强A



### 估计t

$$
假定在\Omega(x)区域内大气透色率t(x)为常数{\tilde{t}\left(x\right)}，对数学模型同时取暗通道\\
\min_{c}(\min_{y\in\Omega(x)}(\frac{I^{c}(y)}{A^{c}}))=\tilde{t}\left(x\right)\min_{c}(\min_{y\in\Omega(x)}(\frac{J_{(}^{c}y)}{A^{c}}))+(1-\tilde{t}\left(x\right))\\
又因为暗通道的先验条件为J^{dark}(x)\to0 \\
所以\min_c(\min_{y\in\Omega(x)}(\frac{J_(^cy)}{A^c}))=0\\
即\tilde{t}\left(x\right)=1-\min_{c}(\min_{y\in\Omega(x)}(\frac{I^{c}\left(y\right)}{A^{c}}))\\
上式中可以理解为 ： 模糊率=1-透明率\\
在天空场景中暗通道先验条件不适用，要另做分析，天空的亮度与大气光强的亮度十分接近，因此可以得到\\
\min_c(\min_{y\in\Omega(x)}(\frac{I^c(y)}{A^c}))\to1,\quad \tilde{t}\left(x\right)\to0\\
总之，模糊率\tilde{t}(x)的取值仍然可以认为满足1-透明率，但实际上需要考虑透明率并非完全透明，因此可以修正为\\
模糊率t=1-\omega * 透明率\frac I A，\omega一般取0.95
$$

### 导向滤波

用于图像平滑，基本思想是利用一个图像来引导滤波过程，导向滤波器能够保留图像中的一些重要细节，并且能够在滤波过程中保持边缘信息。如果不加导向滤波，涂抹感比较严重。