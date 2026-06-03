# Training an open quantum classifier

# Ufukkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk

Department of Electrical Engineering
Istanbul Technical University
Istanbul, Turkey\nufukkorkmaz@itu.edu.tr

# Melih Can Topal

Department of Electrical Engineering
Istanbul Technical University
Istanbul, Turkey
topal17@itu.edu.tr

# Ekin

Department of Electrical Engineering
Istanbul Technical University
Istanbul, Turkey
aygule16@itu.edu.tr

# Deniz Türkpençe

Istanbul Technical University
Istanbul, Turkey
dturkpence@itu.edu.tr

Abstract—Quantum machine learning (QML) aims to embed the power of quantum computation with learning theory. Quantum noise and finding the best recipe for encoding classical information into a quantum register could be seen as challenges to overcome for computational performance. Classification of quantum information is a subtask for QML. In this study, we adopt a dissipative route for quantum data classification and examine the developed theory on a gradient descent-based learning task. In particular, we follow repeated interactions based on open quantum dynamics where the binary decision is encoded on a steady state. Based on the analytical results, we develop a cost function for training an open quantum neuron. We demonstrate that the dissipation-driven protocol is suitable for a supervised learning scheme.

Index Terms—quantum learning, open quantum system, cost function, quantum classifier, training

# I. INTRODUCTION

Perceptron is a mathematical representation associated with biological neuron cells as basic units of neurocomputing. [1]–[3]. Artificial neural network (ANN)-based ML algorithms performed by the binary classification employed by perceptrons have numerous applications in the study of data analysis [1], [2], [4]–[13]. According to Moore's law, regarding state-of-the-art semiconductors, the number of transistors increases linearly over time [14]. The issues with the production of chips in the nm scale reveal the fact that Moore's law has reached a physical limit.

However, alternative computing paradigms, such as quantum computation (QC) [15] which has four decades of history, have reached a physical realization level outperforming classical computer performance. In addition, ideas on a QML theory proposing to take the existing advantages of QC have been established [16]–[26]. Today's quantum computers operate in quantum noise, which limits circuit model performance. QML algorithms comprise parametrized gates whose parameters should be updated through a learning process. This iterative repetition of learning protocols deteriorates the performance of the algorithms due to the quantum noise. Moreover, the

978-1-6654-7013-1/22/\$31.00 ©2022 IEEE

QML algorithms simulating a quantum neural network (QNN) should contain multi-qubit gates with time-dependent control to achieve a classification routine. Therefore, the classification task with multi-qubit gates constitutes the most error-prone part of QML algorithms.

In our previous work, we proposed an open system-based dissipative model for quantum data classification and derived analytical expressions for the classification rule. [27]–[30]. It's been reported that the circuit model turns out to be equivalent to the dissipative model of QC [31]. In our model, a probe qubit is in contact with, distinct, idealized qubit reservoirs. We dub them information reservoirs [32], [33] as each consists identical qubit units with specific parameters. In the scenario, each unit interacts with the probe qubit sequentially. At the end of the open system evolution, the probe qubit reaches a steady state where the binary decision takes place. The parameters of the reservoir units could be regarded as input quantum information, while the steady-state probe qubit magnetization is the output. The repeated interaction process described above is called a collision model [34], [35]. And it is extremely usefllllllllllllllllllllllllllllllllllll in describing quantum reservoirs with specific parametrization.

In this study, we examine the trainability of the open quantum perceptron model we offer. We follow a supervised learning scheme and define a cost function and minimize it with a gradient-descent algorithm. We show that the open quantum classifier is suitable for training processes as its response to the variation of system parameters is smooth and easily differentiable.

# II. MODEL AND SYSTEM DYNAMICS

In general, supervised learning schemes define a mpping from an input data set expressing an abstract feature space to a binary label set

As shown in Eq. (1), with binary labels, while  $\mathcal{X}$  represents the input data frame,  $\mathcal{Y}$  represents the output data frame for a given training set  $\mathcal{S} = (\mathcal{X}, \mathcal{Y})$ . The simplest possible model

for data discrimination is a binary classifier. Perceptron is the mathematical description of this model. In mathematical terms, a vector  $\mathbf{x} = [\mathbf{x_1}, \dots \mathbf{x_N}]^T$  defining the input data with a corresponding weight vector  $\mathbf{w} = [\mathbf{w_1}, \dots \mathbf{w_N}]^T$ , the binary decision structured by  $z = f(\mathbf{x^T}\mathbf{w})$ . Here, z is the binary output and f is, in general, a non-linear function. The classification rule is  $z \equiv 0$  if  $z = f(\mathbf{x^T}\mathbf{w}) \geq \mathbf{0}$  and  $z \equiv 1$ , otherwise

In essence, the input is a weighted combination of data items given by

Reminding that as we develop an open quantum equivalent of a perceptron model, the input data should be accompanied by a quantum channel [36], [37]. In words, dissipative transfer of quantum information weighted by reservoir couplings could provide an open quantum equivalent of Eq. (2) as

Here,  $P_i$  is a positive number depicting the probability of the system coupling to ith reservoir and  $\Phi_t^{(i)}$  is a quantum dynamical map acting on the probe qubit defined by  $\varrho_0$ . We assume that the dynamical map  $\Phi_t^{(i)}$  expressed above, defines a physical process. That is,

$$\Phi_t^{(i)}[\varrho_0] = \operatorname{Tr}_{\mathcal{R}_i} \{ U_t(\varrho_0 \otimes \rho_{\mathcal{R}_i}) U_t^{\dagger} \}$$
 (4)

is a completely positive trace preserving quantum dynamical map (CPTP) where  $\rho_{\mathcal{R}_i}$  the *i*th reservoir density matrix and  $U_t$  stands for a unitary propagator acting both the probe qubit and the reservoir degrees of freedom. Here, we should note that Eq. (3) applies only in the weak coupling condition where the additivity of the quantum dynamical maps is valid [38]. Furthermore, validation of this additivity applies in terms of collisional models only when the CP divisibility condition is satisfied [39], [40].

#### A. System dynamiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

Before describing the model more particularly, we highlight some points. First, the dynamics of proposed classifier, where one by one a probe qubit interacts with identical, non-interacting units of a reservoir with a finite time  $\tau$ , is characterized by a repeated-interactions process. Second, these reservoirs are information reservoirs which are the idealized identical qubits with specific parameters defining input data. Initially, the reservoir plus probe qubit systems are assumed to be  $\varrho(0) = \varrho_0(0) \otimes \rho_{\mathcal{R}_i}$  disentangled. The information reservoirs are composed of n identical qubits also in a product state. For instance, the ith information reservoir reads

$$\rho_{\mathcal{R}_i} = \bigotimes_{k=1}^n \rho_k(\theta_i, \phi_i). \tag{5}$$

where  $\rho_k(\theta_i, \phi_i)$  is the kth subunit of the reservoir. Next, the system plus ith information reservoir Hamiltonian is written as  $\mathcal{H} = \mathcal{H}_0 + \mathcal{H}_{int}$ . Here, free terms part is given by

$$\mathcal{H}_0 = \frac{\hbar\omega_0}{2}\sigma_0^z + \frac{\hbar\omega_i}{2}\sum_{k=1}^n \sigma_{k_i}^z \tag{6}$$

and the interaction term is given by

$$\mathcal{H}_{int} = \hbar \sum_{k=1}^{n} g_i(\sigma_0^+ \sigma_{k_i}^- + \text{H.c.}),$$
 (7)

Here,  $\hbar$ ,  $\sigma^z$ ,  $\sigma^+$  and  $\sigma^-$  are, respectively, the Plack constant, the Pauli-z operator, the Pauli raising and lowering operators. For simplicity, the probe qubit frequency and reservoir qubit frequencies were set  $\omega_0 = \omega_i$  equal. As a critical point, we take the coupling strength denoted by g as an adjustable parameter for learning tasks. In addition, regarding Eq. (3), we note that the coupling and the probability experienced from each reservoir are related, such that  $g_i \propto P_i$  without overlooking the weak coupling condition.

To define a grant and a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant a grant

$$\Phi_{n\tau}^{(i)} = \operatorname{Tr}_{n} \left[ \mathcal{U}_{0i_{n}} \dots \operatorname{Tr}_{1} \left[ \mathcal{U}_{0i_{1}} \left( \varrho_{0} \otimes \rho_{\mathcal{R}_{i_{1}}} \right) \mathcal{U}_{0i_{1}}^{\dagger} \right] \otimes \dots \right]$$

$$\dots \otimes \rho_{\mathfrak{R}_{i_{n}}} \mathcal{U}_{0i_{n}}^{\dagger} \right]. \tag{8}$$

where  $n\tau$  is the time for n collisions. Here,  $\mathcal{U}_{0i_k} = \exp[-i\mathcal{H}_{0i}^k\tau]$  is the unitary propagator acting both the reservoir and system (the probe qubit) degrees of freedom. With finite n collisions, it's assumed that the probe qubit can reach the steady state in which the binary decision encoded.

Reminding that the proposed model operates as an open quantum system, we benefit from the results of our previous work to characterize the system with a master equation approach derived from a micromaser-like repeated interactions approach [28]–[30]. To this end, the unitary propagator in the interaction picture obtained with respect to  $\mathcal{H}_0$  as

$$\mathcal{U}(\tau) = \mathbf{1} - i\tau(\sigma_0^+ \mathbf{J}_{\mathbf{g_i}}^- + \sigma_0^- \mathbf{J}_{\mathbf{g_i}}^+) - \frac{\tau^2}{2} (\sigma_0^+ \sigma_0^- J_{g_i}^- J_{g_i}^+ + \sigma_0^- \sigma_0^+ J_{g_i}^+ J_{g_i}^-)$$
(9)

with  $\mathcal{U}(\tau) = \exp[-\mathrm{i}\mathcal{H}_{\mathrm{int}}\tau]$ ,  $(\hbar=1)$  where  $J_{g_i}^{\pm} = \sum_{i=1}^N g_i \sigma_i^{\pm}$  are the collective operators weighted by  $g_i$ . In compliance with the collision model approach, reservoir states are reset to their initial state, hence the whole system is assumed to be factorized  $\varrho(t) = \varrho_0(t) \otimes \rho_{\mathcal{R}_i}$  after each interaction.

With these specifications, the obtained master equation reads [29]

$$\dot{\varrho}_{0} = -i[\mathcal{H}_{\text{eff}}, \varrho] + \sum_{i=1}^{N} g_{i}^{2} \left( \zeta^{+} \mathcal{L}[\sigma_{0}^{+}] + \zeta^{-} \mathcal{L}[\sigma_{0}^{-}] \right) + \sum_{i < j}^{N'} g_{i} g_{j} \left( \zeta_{s}^{+} \mathcal{L}_{s}[\sigma_{0}^{-}] + \zeta_{s}^{-} \mathcal{L}_{s}[\sigma_{0}^{+}] \right)$$
(10)

where  $\mathcal{H}_{\text{eff}} = r\tau \sum_{i}^{N} g_{i} \left( \langle \sigma_{i}^{-} \rangle \sigma_{0}^{+} + \langle \sigma_{i}^{+} \rangle \sigma_{0}^{-} \right)$  is the effective Hamiltonian pointing out a coherent drive on the probe qubit.

The calculated averages over ith information reservoir denoted as  $\langle \mathbb{O}_i \rangle = \mathrm{Tr}[\mathbb{O} \rho_{\mathcal{R}_i}]$  and  $\mathcal{L}[o] \equiv 2o\rho o^\dagger - o^\dagger o\rho - \rho o^\dagger o$  defines a standard Lindblad super operator with  $\mathcal{L}_s[o] \equiv 2o\varrho o - o^2\varrho - \varrho o^2$  denoting a squeezing effect by the reservoir. The coefficients containing diagonal  $\zeta^\pm = r \tau^2 \langle \sigma_i^\pm \sigma_i^\mp \rangle / 2$  and the off-diagonal terms  $\zeta_s^\pm = 2r \tau^2 \langle \sigma_i^\pm \rangle \langle \sigma_j^\pm \rangle$  in front of the Lindblad terms carry the input information from the reservoirs to the probe qubit.

As the ultimate goal is to readout the binary classification result from the probe qubit in a steady state, we obtain the the steady state density matrix of the probe qubit as

$$\varrho_0^{\text{ss}} = \frac{1}{\sum_{i}^{N} g_i^2} \sum_{i=1}^{N} g_i^2 \left( \langle \sigma_i^+ \sigma_i^- \rangle | e \rangle \langle e | + \langle \sigma_i^- \sigma_i^+ \rangle | g \rangle \langle g | + i \gamma_1^- \left( \langle \sigma_i^+ \sigma_i^- \rangle - \langle \sigma_i^- \sigma_i^+ \rangle \right) | e \rangle \langle g | + \text{H.c.} \right). \tag{11}$$

In this regime, the solution of the Bloch equations read

$$\langle \sigma_x^0 \rangle^{ss} = \frac{i(\gamma_1^- - \gamma_2^+)}{g_{\sum}} \sum_i^N g_i^2 \langle \sigma_z \rangle_i$$

$$\langle \sigma_y^0 \rangle^{ss} = \frac{-(\gamma_1^- + \gamma_2^+)}{g_{\sum}} \sum_i^N g_i^2 \langle \sigma_z \rangle_i$$

$$\langle \sigma_z^0 \rangle^{ss} = \frac{1}{g_{\sum}} \sum_i^N g_i^2 \langle \sigma_z \rangle_i$$
(12)

where, respectively,  $\gamma_1^- = r\tau \sum_{i=1}^N g_i \langle \sigma_i^- \rangle, \ \gamma_2^+ = r\tau \sum_{i=1}^N g_i \langle \sigma_i^+ \rangle, \ \gamma_3^+ = \frac{r\tau^2}{2} \sum_{i=1}^N g_i^2 \langle \sigma_i^+ \sigma_i^- \rangle, \ \gamma_4^- = \frac{r\tau^2}{2} \sum_{i=1}^N g_i^2 \langle \sigma_i^- \sigma_i^+ \rangle, \ \gamma_5^- = 2r\tau^2 \sum_{i< l}^{N'} g_i g_j \langle \sigma_i^- \rangle \langle \sigma_j^- \rangle$  and  $\gamma_6^+ = 2r\tau^2 \sum_{i< j}^{N'} g_i g_j \langle \sigma_i^+ \rangle \langle \sigma_j^+ \rangle.$ 

# B. TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

With all these results, the operating principle of the quantum neuron is expressed in the steady state. We choose steadystate magnetization as the merit quantifier for the binary classification decision. Using the results of the Bloch equations binary classification rule reads

Decision: 
$$\begin{cases} 0, & \langle \sigma_z^0 \rangle^{ss} = \frac{1}{g_{\Sigma}} \sum_i^N g_i^2 \langle \sigma_z \rangle_i \ge 0 \\ 1, & \text{else} \end{cases}$$
 (13)

where  $g_{\sum} = \sum_{i}^{N} g_{i}^{2}$ . Here,  $\langle \sigma_{z} \rangle_{i}$  is the *i*th information reservoir magnetization of each identical units. That is, *i*th input information is weighted by coupling strength  $g_{i}$ . Hence, the developed classification rule manifests the similarity between a classical perceptron.

Now, we are equipped to examine the learning tasks of an open quantum neuron. However, before, we would like to examine the steady response magnetization of the probe qubit contacted to multiple reservoirs by altering the couplings. Throughout the manuscript, we use the parameters of superconducting quantum circuits operating in the microwave regime [41]–[44]. Weakly coupled superconducting transmon qubits typically operate with a resonator frequency  $\omega_r \sim 1-10$  GHz with  $q \sim 1-500$  MHz qubit-resonator coupling and

 $J \sim 1-100$  MHz effective qubit-qubit coupling. In this regime, the qubit energy relaxation time ranges  $T_1 \sim 70-150$   $\mu s$  and the dephasing time ranges  $T_2 \sim 60-220$   $\mu s$  [41]–[44].

![](_page_2_Figure_12.jpeg)

Fig. 1. The variation of the steady state magnetization of the system qubit depending on  $g_1=g/2+\delta g,\ g_2=g/2-\delta g$  coupling coefficients where  $\delta g$  is a fraction of g with g=0.01 and the average number of interactions  $\langle k \rangle$  within a time interval  $T=1/\Gamma_0$ . The probe qubit prepared initially in  $|+\rangle=(|e\rangle+|g\rangle)/\sqrt{2}$  state and contacted collisionally with the identical reservoir units  $|\Psi(\theta,\phi)\rangle$  with  $\theta=0,\ \phi=0$  and  $\theta=\pi,\ \phi=0$ . The decay rate of the probe qubit is  $\Gamma_0=2\times 10^{-5}$ . The probe qubit-reservoir interaction time  $\tau=3$  and the coupling strength to the reservoir g=0.01 are dimensionless and scaled by  $\omega_r$ .

In our first demonstration, a quantum neuron contacted two different information reservoirs. In Fig. 1, we examine the steady state response of the neuron (the probe qubit) for the variation of the couplings. We monitor the equilibration dynamics through smooth variation of the steady state magnetization (response) of the system qubit. We evaluate this smooth response as a favourable result for training tasks where differentiability condition is met.

# III. THE COST FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU

In this section, we describe the calculation of the cost function to optimise, in the context of the response of open quantum dynamics. In compliance with the supervised learning schemes, the error distance between an input quantum state and a target quantum state, should be iteratively minimized. Here, the parameter to be updated is the coupling strengths

where  $\delta g$  denotes a small change in g. Similar to the least squares methods, the cost function is defined as the square difference between the vectors expressing the actual A (input) and the target values Y as

In our scenario, the cost function reads

where the  $\langle \sigma_z^0 \rangle_{des}^{ss}$  desired and the  $\langle \sigma_z^0 \rangle_{act}^{ss}$  actual steady state magnetizations of the proposed open classifier. The applied algorithm to minimize the cost is the well-known gradient

descent [45]. In the task, the couplings (weights) are iteratively altered as

$$\delta g_i = -\eta \frac{\partial C}{\partial q_i}.\tag{17}$$

where  $\eta$  is a non-negative parameter—the learning rate—defining the speed of the learning process. As its mathematical definition implies, the partial derivative expresses the variation of the coupling rates in the direction of the greatest descent.

![](_page_3_Figure_3.jpeg)

Fig. 2. Results for three different learning rates for GD.

In our current example, we have two information reservoirs corresponding to specific magnetizations. Therefore, the actual steady state magnetization reads as

$$A = \langle \sigma_z^0 \rangle_{act}^{ss} = \frac{g_1^2 \langle \sigma_z^1 \rangle + g_2^2 \langle \sigma_z^2 \rangle}{g_1^2 + g_2^2}.$$
 (18)

a constant of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of teeeee e e e e t ttttttttttttttttt

$$\frac{\partial A}{\partial g_1} = \frac{2g_1 \langle \sigma_z^1 \rangle (g_1^2 + g_2^2) - 2g_1 (g_1^2 \langle \sigma_z^1 \rangle + g_2^2 \langle \sigma_z^2 \rangle)}{(g_1^2 + g_2^2)^2} 
\frac{\partial A}{\partial g_2} = \frac{2g_2 \langle \sigma_z^2 \rangle (g_1^2 + g_2^2) - 2g_2 (g_1^2 \langle \sigma_z^1 \rangle + g_2^2 \langle \sigma_z^2 \rangle)}{(g_1^2 + g_2^2)^2}$$
(19)

In our example, the desired magnetization is  $\langle \sigma_z^0 \rangle_{des}^{ss} = 0.4$  a constant value in the cost function. Substituting Eqs. (18) and (19) in Eq. (17), Eq. (14) becomes as follows:

$$(\mathbf{g_1})_{\mathbf{k+1}} = (\mathbf{g_1})_{\mathbf{k}} + \delta(\mathbf{g_1})_{\mathbf{k}}$$

$$(\mathbf{g_2})_{\mathbf{k+1}} = (\mathbf{g_2})_{\mathbf{k}} + \delta(\mathbf{g_2})_{\mathbf{k}}.$$
(20)

# IV. TRAINING AND LEARNING

As we have pointed out, open systems dynamics are favourable as it has a smooth response to the changes in learning parameters. We trained our model in Fig. (2) for three different learning rates using the GD algorithm using Eq. (20). We defined the learning rate in terms of coupling rates for relating the process with system parameters.

We observe that even for high learning rates, the cost function successfully descents in the model. Our model also presents a suitable visualization portrait of the cost function in 3D as in Fig. (3). The dashed line in Fig. (3), depicts the decrease of the cost function corresponding to successful updates of  $g_1$  and  $g_2$  parameters.

![](_page_3_Figure_14.jpeg)

Fig. 3. The 3D surface plot of the cost function according to  $g_1$  and  $g_2$ . Here, the initial values for the dashed line are the learning rate  $\eta=2.5\times 10^{-5},$   $\langle\sigma_z^1\rangle=0.9,$   $\langle\sigma_z^2\rangle=-0.1,$   $g_1=0.001,$   $g_2=0.06$  and  $\langle\sigma_z^0\rangle_{des}^{ss}=0.4,$  respectively.

#### V. COOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

In conclusion, we have investigated the possibility of the trainability of an open quantum neuron model. As a first demonstration, we have applied the conventional GD algorithm to a previously developed open quantum classifier model. We observe that open-system dynamics allow for smooth variations in the training process favourable for the application of GD-based algorithms needing differentiability of system parameters. We successfully demonstrated the trainability of a quantum classifier operating in open system dynamics.

Since the classification decision of the model is obtained in a steady state, which is a natural open system balancing process, no additional optimization is required as in the quantum computing circuit model. Note that data encoding and time-depending gate optimization with controlled quantum gates are the most error-prone parts of quantum algorithms. Therefore, our results show that the proposed learning task is suitable for the conventional circuit model of QC in a hybrid manner and allows for minimizing errors.

# ACKNOWLEDGMENT

The authors acknowledge support from the Scientific and Technological Research Council of Turkey (TÜBİTAK-Grant No. 120F353). The authors also wish to extend special thanks to the Cognitive Systems Lab in the Department of Electrical Engineering providing the atmosphere for motivational and stimulating discussions.

#### REFERENCES

- [1] W. S. McCulloch and W. Pitts, "A logical calculus of the ideas immanent in nervous activity," *Bulletin of Mathematical Biophysics*, vol. 5, no. 4, pp. 115–133, Dec. 1943.
- [2] J. Misra and I. Saha, "Artificial neural networks in hardware: A survey of two decades of progress," *Neurocomputing*, vol. 74, no. 1, pp. 239–255, 2010.
- results for the second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second second seco s s s s se se se se se se s s s s s s
- [4] S. S. Haykin, Neural Networks and Learning Machines. Prentice Hall, 2009.

- [5] J. Gu, Z. Wang, J. Kuen, L. Ma, A. Shahroudy, B. Shuai, T. Liu, X. Wang, G. Wang, J. Cai, and T. Chen, "Recent advances in convolutional neural networks," *Pattern Recognition*, vol. 77, pp. 354–377, 2018.
- [6] J. Schmidhuber, "Deep learning in neural networks: An overview," *Neural Networks*, vol. 61, pp. 85–117, 2015.
- [7] E. W. T. Ngai, L. Xiu, and D. C. K. Chau, "Application of data mining techniques in customer relationship management: A literature review and classification," *Expert Systems with Applications*, vol. 36, no. 2, Part 2, pp. 2592–2602, 2009.
- [8] Z.-G. Hou, L. Cheng, and M. Tan, "Decentralized Robust Adaptive Control for the Multiagent System Consensus Problem Using Neural Networks," *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, vol. 39, no. 3, pp. 636–647, 2009.
- [9] N. Tajbakhsh, J. Y. Shin, S. R. Gurudu, R. T. Hurst, C. B. Kendall, M. B. Gotway, and J. Liang, "Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?" *IEEE Transactions on Medical Imaging*, vol. 35, no. 5, pp. 1299–1312, May 2016.
- [10] J. Tang, C. Deng, and G.-B. Huang, "Extreme Learning Machine for Multilayer Perceptron," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 27, no. 4, pp. 809–821, 2016.
- [11] B. Shi, X. Bai, and C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 39, no. 11, pp. 2298–2304, Nov. 2017.
- [12] A. C. Lorena, A. C. P. L. F. de Carvalho, and J. M. P. Gama, "A review on the combination of binary classifiers in multiclass problems," *Artif Intell Rev*, vol. 30, no. 1, p. 19, 2009.
- [13] M. Galar, A. Fernandez, E. Barrenechea, H. Bustince, and F. Herrera, ´ "An overview of ensemble methods for binary classifiers in multi-class problems: Experimental study on one-vs-one and one-vs-all schemes," *Pattern Recognition*, vol. 44, no. 8, pp. 1761–1776, 2011.
- [14] G. E. Moore, "Cramming more components onto integrated circuits," *Electronics*, vol. 38, no. 8, p. 4, 1965.
- [15] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information: 10th Anniversary Edition*, anniversary edition ed. Cambridge ; New York: Cambridge University Press, 2011.
- [16] M. Schuld and F. Petruccione, "Learning with Quantum Models," in *Supervised Learning with Quantum Computers*, M. Schuld and F. Petruccione, Eds. Cham: Springer International Publishing, 2018, pp. 247– 272.
- [17] M. Schuld, I. Sinayskiy, and F. Petruccione, "The quest for a Quantum Neural Network," *Quantum Inf Process*, vol. 13, no. 11, pp. 2567–2586, 2014.
- [18] P. Rebentrost, M. Mohseni, and S. Lloyd, "Quantum Support Vector Machine for Big Data Classification," *Phys. Rev. Lett.*, vol. 113, no. 13, p. 130503, 2014.
- [19] L. Banchi, N. Pancotti, and S. Bose, "Quantum gate learning in qubit networks: Toffoli gate without time-dependent control," *npj Quantum Information*, vol. 2, no. 1, p. 16019, 2016.
- [20] S. Lloyd and C. Weedbrook, "Quantum Generative Adversarial Learning," *Phys. Rev. Lett.*, vol. 121, no. 4, p. 040502, Jul. 2018.
- [21] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, "Power of data in quantum machine learning," *Nat Commun*, vol. 12, no. 1, p. 2631, May 2021.
- [22] A. Y. Yamamoto, K. M. Sundqvist, P. Li, and H. R. Harris, "Simulation of a Multidimensional Input Quantum Perceptron," *Quantum Inf Process*, vol. 17, no. 6, p. 128, 2018.
- [23] F. Tacchino, C. Macchiavello, D. Gerace, and D. Bajoni, "An artificial neuron implemented on an actual quantum processor," *npj Quantum Inf*, vol. 5, no. 1, pp. 1–8, Mar. 2019.
- [24] E. Torrontegui and J. J. Garc´ıa-Ripoll, "Unitary quantum perceptron as efficient universal approximator," *EPL*, vol. 125, no. 3, p. 30004, Mar. 2019.
- [25] A. Abbas, M. Schuld, and F. Petruccione, "On quantum ensembles of quantum classifiers," *Quantum Mach. Intell.*, vol. 2, no. 1, p. 6, 2020.

- [26] S. Mangini, F. Tacchino, D. Gerace, D. Bajoni, and C. Macchiavello, "Quantum computing models for artificial neural networks," *EPL*, vol. 134, no. 1, p. 10002, 2021.
- [27] D. Turkpenc¸e, T. C¸ . Akıncı, and S. S¸ eker, "A steady state quantum ¨ classifier," *Physics Letters A*, vol. 383, no. 13, pp. 1410–1418, 2019.
- [28] U. Korkmaz, C. Sanga, and D. Turkpenc¸e, "Mimicking an Informa- ¨ tion Reservoir by Superconducting Quantum Circuits," in *2021 5th International Symposium on Multidisciplinary Studies and Innovative Technologies (ISMSIT)*, 2021, pp. 105–109.
- [29] U. Korkmaz and D. Turkpenc¸e, "Transfer of quantum information via a ¨ dissipative protocol for data classification," *Physics Letters A*, vol. 426, p. 127887, 2022.
- [30] U. Korkmaz, C. Sanga, and D. Turkpenc¸e, "Quantum Data Classification ¨ by a Dissipative Protocol with a Superconducting Quantum Circuit Implementation," in *Electrical and Computer Engineering*, ser. Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, M. N. Seyman, Ed. Springer International Publishing, 2022, pp. 159–170.
- [31] F. Verstraete, M. M. Wolf, and J. Ignacio Cirac, "Quantum computation and quantum-state engineering driven by dissipation," *Nature Phys*, vol. 5, no. 9, pp. 633–636, Sep. 2009.
- [32] S. Deffner and C. Jarzynski, "Information Processing and the Second Law of Thermodynamics: An Inclusive, Hamiltonian Approach," *Phys. Rev. X*, vol. 3, no. 4, p. 041003, Oct. 2013.
- [33] S. Deffner, "Information-driven current in a quantum Maxwell demon," *Phys. Rev. E*, vol. 88, no. 6, p. 062128, Dec. 2013.
- [34] V. Scarani, M. Ziman, P. Stelmachovi ˇ c, N. Gisin, and V. Bu ˇ zek, "Ther- ˇ malizing Quantum Machines: Dissipation and Entanglement," *Physical Review Letters*, vol. 88, no. 9, p. 097905, 2002.
- [35] M. Ziman, P. Stelmachovi ˇ c, V. Bu ˇ zek, M. Hillery, V. Scarani, and ˇ N. Gisin, "Diluting quantum information: An analysis of information transfer in system-reservoir interactions," *Physical Review A*, vol. 65, no. 4, p. 042105, 2002.
- [36] R. Blume-Kohout and W. H. Zurek, "A Simple Example of "Quantum Darwinism": Redundant Information Storage in Many-Spin Environments," *Foundations of Physics*, vol. 35, no. 11, pp. 1857–1876, 2005.
- [37] M. Zwolak and W. H. Zurek, "Redundancy of einselected information in quantum Darwinism: The irrelevance of irrelevant environment bits," *Physical Review A*, vol. 95, no. 3, p. 030101, 2017.
- [38] J. Kołodynski, J. B. Brask, M. Perarnau-Llobet, and B. Bylicka, "Adding ´ dynamical generators in quantum master equations," *Phys. Rev. A*, vol. 97, no. 6, p. 062124, 2018.
- [39] M. M. Wolf and J. I. Cirac, "Dividing Quantum Channels," *Communications in Mathematical Physics*, vol. 279, no. 1, pp. 147–168, Apr. 2008.
- [40] S. N. Filippov, J. Piilo, S. Maniscalco, and M. Ziman, "Divisibility of quantum dynamical maps and collision models," *Physical Review A*, vol. 96, no. 3, p. 032111, 2017.
- [41] J. Majer, J. M. Chow, J. M. Gambetta, J. Koch, B. R. Johnson, J. A. Schreier, L. Frunzio, D. I. Schuster, A. A. Houck, A. Wallraff, A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Coupling superconducting qubits via a cavity bus," *Nature*, vol. 449, no. 7161, pp. 443–447, 2007.
- [42] X.-H. Deng, E. Barnes, and S. E. Economou, "Robustness of errorsuppressing entangling gates in cavity-coupled transmon qubits," *Phys. Rev. B*, vol. 96, no. 3, p. 035441, 2017.
- [43] P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.
- [44] A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," *Rev. Mod. Phys.*, vol. 93, no. 2, p. 025005, 2021.
- [45] K. H. Wan, O. Dahlsten, H. Kristjansson, R. Gardner, and M. S. Kim, ´ "Quantum generalisation of feedforward neural networks," *npj Quantum Inf*, vol. 3, no. 1, p. 36, Dec. 2017.