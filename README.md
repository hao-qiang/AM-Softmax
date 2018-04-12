# AM-Softmax
this is a AM-Softmax tutorial and keras implement.
$$
\begin{aligned}
L_ams & = -\frac{1}{n}\sum_{i=1}^{n}{\log{\frac{e^{s(\cos\theta_{y_i}-m)}}{e^{s(\cos\theta_{y_i}-m)}+\sum_{j=1,j\neq y_i}^{c}{e^{s\cos\theta_j}}}}} \\
& = -\frac{1}{n}\sum_{i=1}^{n}{\log{\frac{e^{s(w_{y_i}^T x_i -m)}}{e^{s(w_{y_i}^T x_i-m)}+\sum_{j=1,j\neq y_i}^{c}{e^{sw_j^T x_i}}}}}
\end{aligned}
$$
