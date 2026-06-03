# Dissipative learning of a quantum classifier

Ufuk Korkmaz<sup>1</sup> and Deniz Türkpençe<sup>1,\*</sup>

<sup>1</sup>Department of Electrical Engineering, İstanbul Technical University, 34469 İstanbul, Turkey (Dated: July 25, 2023)

The expectation that quantum computation might bring performance advantages in machine learning algorithms motivates the work on the quantum versions of artificial neural networks. In this study, we analyze the learning dynamics of a quantum classifier model that works as an open quantum system which is an alternative to the standard quantum circuit model. According to the obtained results, the model can be successfully trained with a gradient descent (GD) based algorithm. The fact that these optimization processes have been obtained with continuous dynamics, shows promise for the development of a differentiable activation function for the classifier model.

### I. INTRODUCTION

The theory of learning artificial neural networks is foun-ded on mathematical models adapted to the working principle of the human brain, introduced by McCulloch, Pitts and Rosenblatt [1, 2]. Particularly in the new millennium, in which the computing capacities of computers have increased, it has become a period in which deep learning methods outperform other methods in multi-layer artificial neural networks, which bring many useful applications [3–5].

Quantum computation (QC) brings exciting advantages to computer science and all relevant computational sciences [6, 7]. Although many efforts have been paid for quantum versions of neural networks (QNN), there is no broadly accepted QNN, even at a single neuron level [8–13]. In addition, quantum noise severely limits the performance of gate-based quantum network proposals. Therefore hardware-efficient solutions have began to emerge [14, 15].

In past work, we proposed a dissipative quantum classifier as a basic unit of QNN hardware, based on repeated interactions protocol [16-19]. Dissipation-based quantum computing has been shown to be equivalent to the standard QC model [20]. In the protocol, identical qubit sequences with pure initial quantum state successively interact with a target qubit. The repeated interactions are unitary in the weak coupling limit in a vanishingly small time portion. However, the quantum state of the target qubit is obtained by calculating the reduced dynamics, so that global evolution is a non-unitary process. We dub these identical qubit sequences quantum information reservoir [21, 22]. As a result of repeated interactions, the target qubit reaches a steady state in which the diagonal entries of its density matrix become identical to the information reservoir units. This process is known as quantum homogenization [23].

In this task, some amount of information is transferred from the reservoir to the target qubit at the steady state. This can be interpreted as quantum reservoirs being quantum channels that transfer information to open

systems [24, 25]. All these approaches make sense for open quantum neuron design when the target qubit is connected to more than one information reservoir with arbitrary coupling strengths. In this case, the target qubit reaches a non-trivial steady state depending on the coupling coefficients (weights) and the input data parameters. We have numerically and analytically proposed that this model is an open quantum classifier that returns a binary decision at the steady state when measured by Pauli observables [16, 17].

In the current work, we study this model in the framework of supervised learning schemes by adopting a gradient descent-based model. To this end, we derive a cost function setting different parameters of the system as variables and examine the availability of the model for learning tasks. We observe that the cost function can be smoothly minimized for all relevant parameters with appropriate differentiability.

## II. MODEL AND SYSTEM DYNAMICS

## A. Classic model

Binary classification is a subtask for machine learning (ML) covering ANN alongside different models. However, If we discuss, in particular the artificial neural network model, the perceptron is referred to as a basic unit of ANN computing performing binary classification tasks. Technically speaking, a perceptron performs a binary decision z with binary labels  $\{0,1\}$  depending on the input. In the model, input is formulated as  $\varphi_{in} = \mathbf{x}^T \mathbf{w}$  where  $\mathbf{x} = [\mathbf{x}_1, \dots \mathbf{x}_N]^T$  defines input feature instances and  $\mathbf{w} = [\mathbf{w}_1, \dots \mathbf{w}_N]^T$  is the set of corresponding weight vectors.

The binary output is modulated by, in general a nonlinear function f(.) where  $z = f(\varphi_{in})$ . The decision rule reads z = 0 if  $z = f(\varphi_{in}) \ge 0$  and z = 1 else. The choice of binary labels is arbitrary and can be defined variously depending on the expressivity requirements. Note that, in principle, a perceptron with identity activation can still achieve linear classification. However, non-linear activation functions are desirable for multi-layer ANN learning tasks. Although our model, in principle, is a

<span id="page-0-0"></span><sup>\*</sup> dturkpence@itu.edu.tr

quantum perceptron with identity activation, we prefer to present our model and related learning tasks as "quantum classifier learning".

Supervised learning can be defined as a mapping from a feature space to a binary label set

$$\mathcal{X}, \mathcal{Y} \to \{0, 1\} \tag{1}$$

where  $\mathcal{X}$  and  $\mathcal{Y}$  are, respectively, the input and output data of a given a training set  $\mathcal{S} = (\mathcal{X}, \mathcal{Y})$ . In this scheme, the  $\mathcal{Y}$  part of the training set is the desired output, and the cost function C quantifies how close the actual output is to the desired output.

In analogy with the least squares method, the cost function expression reads

where A is the actual and Y is the desired output. In general, the weight instances are updated

$$\mathbf{w_{k+1}} = \mathbf{w_k} + \delta \mathbf{w_k} \tag{3}$$

iteratively by back-propagation. However, any desired parameter can be adjusted to minimize the cost. Among different procedures, we adopt a gradient-descent based method for the training task. In this method, the change in the parameter reads

<span id="page-1-1"></span>

where  $\eta$  is a non-negative number, the so-called learning rate, characterizing the speed of the learning task. As the name of the method implies, the partial derivative expresses the change of the parameter to be adjusted in the direction of the largest descent.

## B. Quantum dissipative dynamics

In this subsection, we discuss the open system dynamics with preliminary definitions. As we have pointed out in the previous sections, the model operates by a dissipative protocol. The input data expressed classically can be rephrased as

<span id="page-1-2"></span>

the weighted summation of the input features. In our view, the quantum equivalent of the classic description above reads

where  $\Phi_t^{(i)}$  is a completely positive trace preserving (CPTP) quantum dynamical map acting on the target qubit  $\rho_0$ ,  $P_i$  is the probability of the map interacting with

the ith information reservoir. The subscript t stands for the time dependence of the maps generated by a physical process

$$\Phi_t^{(i)}[\varrho_0] = \operatorname{Tr}_{\mathcal{R}_i} \{ U_t(\varrho_0 \otimes \varrho_{\mathcal{R}_i}) U_t^{\dagger} \}$$
 (7)

with a unitary propagator  $U_t$  acting on both the target qubit and the reservoir. Here,  $\rho_{\mathcal{R}_i}$  is the *i*th reservoir quantum state and  $\text{Tr}_{\mathcal{R}_i}$  is the partial trace over the *i*th reservoir.

The quantum reservoirs provide initial quantum data in pure states. Each reservoir is composed of noncorrelated, non-interacting two-level quantum systems (subunits) defined by

the tensor product of finite n subunits. As each subunit is in a pure quantum state, they could initially be prepared by identical Bloch parameters  $\rho_k(\theta_i, \phi_i)$ . This parametrization allows for a dissipative equivalence of the model with parametrized quantum circuits.

# C. Quantum collision model and the quantum

As mentioned above, the dynamical process of the introduced model relies on a standard quantum collisional model [23, 26, 27]. In our proposal, the target qubit undergoes a collisional dissipative process under multiple, independent information reservoirs with arbitrary couplings. In this scheme, the steady state readout of the target qubit by Pauli observables gives the binary classification output. The dynamical process in the presence of the *i*th information reservoir reads

$$\Phi_{n\tau}^{(i)} = \operatorname{Tr}_{n} \left[ \mathcal{U}_{0i_{n}} \dots \operatorname{Tr}_{1} \left[ \mathcal{U}_{0i_{1}} \left( \varrho_{0} \otimes \rho_{\mathcal{R}_{i_{1}}} \right) \mathcal{U}_{0i_{1}}^{\dagger} \right] \otimes \dots \right]$$

$$\dots \otimes \rho_{\mathfrak{R}_{i_{n}}} \mathcal{U}_{0i_{n}}^{\dagger} \right]. \tag{9}$$

Here,  $n\tau$  is the time elapsed of the dynamical map for n collisions and  $\mathcal{U}_{0i_k} = \exp[-\mathrm{i}\mathcal{H}_{0i}^k\tau]$  is the unitary propagator. Initially, system plus reservoir quantum states prepared in  $\varrho(0) = \varrho_0(0) \otimes \rho_{\mathcal{R}_i}$  a tensor product state. Note that the time dependence is only relevant for the target qubit, and after every collision, the reservoir states are reset to their initial state.

On the other hand, the Hamiltonian governing the system plus reservoir dynamics depicted as  $\mathcal{H} = \mathcal{H}_0 + \mathcal{H}_{int}$  where

$$\mathcal{H}_0 = \frac{\hbar\omega_0}{2}\sigma_0^z + \frac{\hbar\omega_i}{2}\sum_{k=1}^n \sigma_{k_i}^z \tag{10}$$

<span id="page-1-0"></span>is the free part and

$$\mathcal{H}_{int} = \hbar \sum_{k=1}^{n} g_i(\sigma_0^+ \sigma_{k_i}^- + \text{H.c.}),$$
 (11)

<span id="page-2-2"></span>![](_page_2_Figure_1.jpeg)

FIG. 1: (Colour online.) The steady state magnetization of the target qubit depending on the variation of the couplings to the reservoirs. Variations of the coupling rates are  $g_1 = g/2 - \Delta g$ ,  $g_2 = g/2 + \Delta g$ . Here,  $\Delta g$  represents a fraction of g with g = 0.01. The probe qubit prepared initially in  $|+\rangle = (|e\rangle + |g\rangle)/\sqrt{2}$  state and interacted collisionally with the reservoir units  $|\Psi(\theta,\phi)\rangle$  with  $\theta=0$ ,  $\phi=0$  and  $\theta=\pi$ ,  $\phi=0$ . The target qubit-reservoir interaction time  $\tau=3$  and the coupling coefficient g are dimensionless and scaled by

is the interaction part. Here, respectively, the Pauli-z operator, the Pauli raising and lowering operators read as  $\sigma^z$ ,  $\sigma^+$  and  $\sigma^-$ . The Planck's constant divided by  $2\pi$  is taken as  $\hbar=1$  throughout the calculations. As a notable point, the value of the coupling coefficient  $g_i \ll \omega_0$  ranges within the weak coupling regime where the cross-talk between the reservoirs is avoided. Moreover, the coefficients are proportional to the probabilities  $g_i \propto P_i$  in eq. (6) as the quantum equivalent to the weights in the classic model.

get qubit in the presence of N distinct reservoirs reported as the solution of the collisional master equation [17]

$$\varrho_0^{\text{ss}} = \frac{1}{\sum_{i}^{N} g_i^2} \sum_{i=1}^{N} g_i^2 \left( \langle \sigma_i^+ \sigma_i^- \rangle | e \rangle \langle e | + \langle \sigma_i^- \sigma_i^+ \rangle | g \rangle \langle g | + i \gamma_1^- \left( \langle \sigma_i^+ \sigma_i^- \rangle - \langle \sigma_i^- \sigma_i^+ \rangle \right) | e \rangle \langle g | + \text{H.c.} \right)$$
(12)

where  $|e\rangle$  and  $|g\rangle$  are the computational basis and  $\gamma_1^-=r\tau\sum_{i=1}^Ng_i\langle\sigma_i^-\rangle$ , r being the interaction rate of the master equation. The binary decision at the steady state is read upon the Pauli-z operator acting on the target qubit

$$\langle \sigma_z^0 \rangle^{ss} = \frac{1}{g_{\sum}} \sum_{i}^{N} g_i^2 \langle \sigma_z \rangle_i \tag{13}$$

as the classification identifier where  $g_{\sum} = \sum_i g_i^2$ . Based the eqs (12) and (13), finally the binary classification rule

reads

$$Decision: \begin{cases} 0, & \langle \sigma_z^0 \rangle^{ss} = \frac{1}{g_{\Sigma}} \sum_i^N g_i^2 \langle \sigma_z \rangle_i \ge 0 \\ 1, & \text{else} \end{cases}$$
 (14)

where  $\langle \sigma_z \rangle_i$  is the *i*th information reservoir magnetization. The steady state binary decision expressed by the Pauli-z observable is a summation of the input quantum data weighted by respective couplings. This is reasonable as the classic model has a similar expression.

Figure 1 depicts the numerical verification of the introduced model as a benchmark calculation. Here, the target qubit contacted two different information reservoirs with  $g_1$  and  $g_2$  couplings. The quantum state of the reservoirs are  $|\Psi(\theta=0,\phi=0)\rangle \equiv |\uparrow\rangle$  and  $|\Psi(\theta=\pi,\phi=0)\rangle \equiv |\downarrow\rangle$ , respectively. The dots on the curve represents the steady state magnetization of the target qubit corresponding to  $g_1, g_2$  coupling values. These values modulated as  $g_1 = g/2 - \Delta g$  and  $g_2 = g/2 + \Delta g$  where  $-0.5g < \Delta g < 0.5g$ . For instance, when  $\Delta g = -0.5g$ ,  $g_1 = g$  and  $g_2 = 0$  means that the target qubit is in contact only with the first reservoir with the quantum state  $|\uparrow\rangle$  and vice versa. In these limits, the steady state magnetization gets  $\langle \sigma_z \rangle = 1, -1$  and takes intermediate values when  $-0.5g < \Delta g < 0.5g$  as expected. In the numerical simulation, we have used the realistic parameters of the superconducting circuits in the weak coupling range [28]. Transmon qubits operate at a resonator frequency  $\omega_r \sim 1-10$  GHz with  $g \sim 1-100$ MHz effective qubit-qubit coupling.

As depicted above, the relevant parameters are the Bloch parameters  $\{\theta,\phi\}$ , characterizing the input quantum data. Looking more closely at eqs 12 and 13, one can see the signatures of input data at the steady state as expected values. The expected values can be related to the Bloch parameters as

<span id="page-2-4"></span>
$$\rho_{\mathcal{R}_i} = \begin{bmatrix} \frac{1 + \cos \theta_i}{2} & \frac{e^{-i\phi_i}}{2} \sin \theta_i \\ \frac{e^{i\phi_i}}{2} \sin \theta_i & \frac{1 - \cos \theta_i}{2} \end{bmatrix}$$

$$:= \begin{bmatrix} \langle \sigma_i^+ \sigma_i^- \rangle & \langle \sigma_i^- \rangle \\ \langle \sigma_i^+ \rangle & \langle \sigma_i^- \sigma_i^+ \rangle \end{bmatrix}$$
(15)

<span id="page-2-0"></span>where  $\rho_{\mathcal{R}_i}$  is the quantum state of the *i*th reservoir. Therefore in our model, Pauli-z and Pauli-y observables can be chosen to extract relevant information for  $\theta$  and  $\phi$  parameters, respectively, at the steady state. Expectation value of the Pauli-y observable of the target qubit at the steady state reads

<span id="page-2-3"></span>
$$\langle \sigma_y^0 \rangle^{ss} = \frac{-(\gamma_1^- + \gamma_2^+)}{g_{\Sigma}} \sum_i^N g_i^2 \langle \sigma_z \rangle_i$$
 (16)

<span id="page-2-1"></span>where  $\gamma_1^- = r\tau \sum_{i=1}^N g_i \langle \sigma_i^- \rangle$ ,  $\gamma_2^+ = r\tau \sum_{i=1}^N g_i \langle \sigma_i^+ \rangle$ . Regarding eqs (13) and (16) together, one evaluates that relevant information of the Bloch parameters can be extracted at the steady state of the target qubit through Pauli observables.

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

![](_page_3_Figure_2.jpeg)

FIG. 2: Cost function minimization depending on different learning rates against the variation of g and  $\theta$  respectively. (a) The initial values of the reservoirs are  $\langle \sigma_z^1 \rangle = 0.95$ ,  $\langle \sigma_z^2 \rangle = -0.15$ ,  $g_1 = 0.001$ ,  $g_2 = 0.06$  and  $\langle \sigma_z^0 \rangle_{des}^{ss} = 0.4$ , respectively. (b) The initial values  $\theta_1 = 170^{\circ}$ ,  $\theta_2 = 160^{\circ}$ ,  $g = g_1 = g_2 = 0.05$  and  $\langle \sigma_z^0 \rangle_{des}^{ss} = 0$ , respectively.

#### III. LEARNING OF THE MODEL

In this section, we explore the gradient descent-based learning of the introduced open classifier model. First we define the cost function to be optimized as

$$C = \frac{1}{2} (\langle \sigma_{\lambda}^0 \rangle_{des}^{ss} - \langle \sigma_{\lambda}^0 \rangle_{act}^{ss})^2, \tag{17}$$

where  $\lambda = \{y, z\}$  denotes the Pauli matrices we choose for specific parameters. Here,  $\sigma_{\lambda}^{0}\rangle_{des}^{ss}$  is the desired and  $\langle \sigma_{\lambda}^{0}\rangle_{act}^{ss}$  is the actual steady state expectation values of the target qubit for the Pauli observable  $\sigma_{\lambda}$ . Definition of the cost function above is similar to [29], however, note that the expected values are obtained in steady states in our task.

Following the classic definitions, we rephrase eqs (4) and (5) as

$$\nu_{\mathbf{k+1}} = \nu_{\mathbf{k}} + \delta\nu_{\mathbf{k}} \tag{18}$$

where  $\nu = \{g, \theta, \phi\}$ . Therefore, the relevant parameters are the Bloch parameters and the couplings of the target qubit with the reservoirs. We first derive the cost function for g corresponding to the weights in the classic model (see A). However, we also examine the learning tasks for the Bloch parameters  $\theta$  and  $\phi$  corresponding to fixed values of g.

Figure 2a depicts the cost minimization given the parameters against the episodes (the k index) of  $\nu = g$  in eqs (18) and (19) depending on different values of  $\eta$  when the target qubit contacted to two reservoirs. That is, we examine the model using different learning rates (or different optimization speeds). We observe that the optimization always has a smooth feature for different  $\eta$ s.

or the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of th

Figures 3a and 3b present the same minimization problem when considering the surface topology of the cost function. In the single target qubit case coupled to two information reservoirs, the structure of the surface cost function seems trivial to optimize without any local plateaus. Therefore, the success of optimization depends on the selection of the learning rate value. In figure 3a, the model successfully performs the optimization task with an appropriate learning rate. However, an unstable procedure occurs when a very large value of  $\eta$  is selected, as in figure 3b. Although, in the figure, cost function minimization seems to have been successfully achieved, in most similar problems the iteration value extends beyond the cost function surface. This is known as 'overshooting' the minimum.

<span id="page-3-2"></span><span id="page-3-1"></span>Conversely, extremely small learning rate values lead to being stuck in the local minimums. Therefore adaptive tasks, in which the learning rates might take different values during the process, are developed for GD-based methods [30]. We find that for the training of the open classifier model, one order of magnitude around the coupling rate in the weak coupling regime seems a reasonable choice for  $\eta$ .

As we have pointed out above, we also examine the cases where the couplings to the reservoirs are fixed. In this case, input data parameters are assumed to be adjustable to obtain the desired output. Regarding figure 2b, we observe, again, smooth convergence with three orders of magnitude greater learning rate than the coupling coefficient. Corresponding cost function is depicted in figure 4a. In this case, the Bloch parameter  $\theta$  is iterated to minimize the cost. See eqs (A2) and (A4) to obtain the cost function in case of the  $\theta$  parameter-

<span id="page-4-1"></span><span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

FIG. 3: Cost function minimization with surface depiction against the variation of  $g_1$  and  $g_2$ . (a) The learning rate  $\eta=2.8\times 10^{-5}, \langle \sigma_z^1\rangle=0.95, \langle \sigma_z^2\rangle=-0.15,$  the target magnetization is  $\langle \sigma_z^0\rangle_{des}^{ss}=0.4$  and the initial coupling rates before optimization is  $g_1=0.001,$   $g_2=0.06$ . (b) The learning rate  $\eta=2.5\times 10^{-2},$   $\langle \sigma_z^1\rangle=0.95, \langle \sigma_z^2\rangle=-0.15,$  the target magnetization is  $\langle \sigma_z^0\rangle_{des}^{ss}=0.4$  and the initial coupling rates before optimization is  $g_1=0.001, g_2=0.06$ .

FIG. 4: (a) Cost function minimization with surface depiction against the variation of  $\theta_1$  and  $\theta_2$ . The learning rate  $\eta=0.5$ , the couplings are  $g=g_1=g_2=0.03$  and  $\langle\sigma_z^0\rangle_{des}^{ss}=0$ . Initial values for the problem are  $\theta_1=170\,^\circ,\,\theta_2=160\,^\circ.$  (b) Cost function minimization depending on different learning rates against the variation of  $\phi$ . Here,  $\phi_1=350\,^\circ,\,\phi_2=340\,^\circ,\,\theta_1=\theta_2=60\,^\circ,\,g=g_1=g_2=0.01$  and  $\langle\sigma_u^0\rangle_{des}^{ss}=0,\,r=0.36,\,\tau=3.$ 

dependent iteration. The Pauli-z observable is, again, relevant in the calculations.

Next, we consider the training task concerning the Bloch parameter  $\phi$ . The Pauli-y observable was chosen to extract  $\phi$  parameter-dependent data. This task requires special attention as the proposed classifier operates as an open quantum system, driven by non-equilibrium reservoirs. Steady states bear mixed quantum states in which quantum coherent information is irreversibly lost. However, some non-vanishing quantum coherence may exist when the system is driven by non-equilibrium environments. [31, 32]. In our case, eq. (12) demonstrates that the target qubit retains quantum coherence at the steady state as the non-diagonal part of the density matrix is non-zero. In addition, eq. (16) states that the steady coherence is weighted by the coupling coefficients where it can be parametrized by  $\phi$  through the Pauli-y observable.

Figure 4b shows the cost minimization depending on

different learning rates. The cost function values around the  $\times 10^{-5}$  scale reveals a small coherence value at the steady state compared to the diagonal elements of the target qubit density matrix. In addition, the  $\eta$  value for the  $\phi$  parameter optimization has the largest value compared to the optimizations for the  $g,\theta$  parameters. Finally, figure 5 represents he cost function minimization considering the update of Bloch parameters  $\phi_1$  and  $\phi_2$ . The 3D surface of the cost function is similar to figure 4a, only differing by the value of the learning rate.

If a comment is made by evaluating all the results together, we see that the proposed classifier is suitable for GD-based training schemes. Moreover, open system dynamics allows for smooth convergence in learning tasks which makes the model favourable for multi layer feedforward networks once an activation function is introduced. Since binary classification is a task in itself for ML, the model we propose is a candidate to be a train-

<span id="page-5-3"></span>![](_page_5_Figure_1.jpeg)

FIG. 5: (Colour online.) Cost function minimization with surface depiction against the variation of  $\phi_1$  and  $\phi_2$ . Here, the initial values for the dashed (optimization) line are  $\eta = 500$ ,  $\phi_1 = 350\,^{\circ}$ ,  $\phi_2 = 340\,^{\circ}$ ,  $\theta_1 = \theta_2 = 60\,^{\circ}$ ,  $g = g_1 = g_2 = 0.01$  and  $\langle \sigma_y^0 \rangle_{des}^{ss} = 0$ , r = 0.36,  $\tau = 3$  respectively.

able model in ML processes, even when considered alone.

### IV. CONCLUSIONS

In this study, we examined the training of a classifier model based on the open quantum model in different parameter spaces with the GD-based method. Using our analytical results, we have derived cost functions for three different parameters for training our model and made calculations that minimize the cost functions with the gradient descent algorithm. Obtaining the classification response of the model in a stationary state makes the system dynamics continuous dynamics. As a result of this, we achieved optimization of the model, namely its training with smooth, continuous results. Since the training processes are continuous, which means that they are differentiable, it is concluded that the model we propose is suitable for developing an activation function and using it in larger quantum networks. In addition, although the classification result is taken in a stationary state, it becomes possible to train in all Bloch parameter spaces as well as the coupling coefficients by the steady quantum coherence.

Our study revealed that the derived cost functions are trained at different values of learning rates for corresponding parameters. In our model, cost functions successfully minimized with appropriate learning coefficients.

# ACKNOWLEDGMENT

The authors acknowledge support from the Scientific and Technological Research Council of Turkey (TÜBİTAK-Grant No. 120F353). The authors also wish to extend special thanks to the Cognitive Systems Lab in

the Department of Electrical Engineering providing the atmosphere for motivational and stimulating discussions.

## <span id="page-5-0"></span>or a series of the cost function

In this section, we present the mathematical justifications for numerical calculations in the text. First, we substitute  $\nu=g$  in eq. (19)

<span id="page-5-5"></span>
$$\delta g_i = -\eta \frac{\partial C}{\partial g_i}.\tag{A1}$$

and obtain the cost function expression taking the partial derivative with respect to the coupling constant g.

<span id="page-5-1"></span>
$$\frac{\partial C}{\partial q_i} = (\langle \sigma_z^0 \rangle_{des}^{ss} - \langle \sigma_z^0 \rangle_{act}^{ss})(-\frac{\partial \langle \sigma_z^0 \rangle_{act}^{ss}}{\partial q_i})$$
(A2)

In our current example, we have two information reservoirs corresponding to specific magnetizations. Therefore, the actual steady state magnetization (eq. (13)) reads as

<span id="page-5-4"></span>
$$A = \langle \sigma_z^0 \rangle_{act}^{ss} = \frac{g_1^2 \langle \sigma_z^1 \rangle + g_2^2 \langle \sigma_z^2 \rangle}{g_1^2 + g_2^2}.$$
 (A3)

According to the recipe to derive the cost function, the partial derivatives with respect to  $g_1$  and  $g_2$  separately obtained as

$$\frac{\partial A}{\partial g_1} = \frac{2g_1 \langle \sigma_z^1 \rangle (g_1^2 + g_2^2) - 2g_1 (g_1^2 \langle \sigma_z^1 \rangle + g_2^2 \langle \sigma_z^2 \rangle)}{(g_1^2 + g_2^2)^2} 
\frac{\partial A}{\partial g_2} = \frac{2g_2 \langle \sigma_z^2 \rangle (g_1^2 + g_2^2) - 2g_2 (g_1^2 \langle \sigma_z^1 \rangle + g_2^2 \langle \sigma_z^2 \rangle)}{(g_1^2 + g_2^2)^2}$$
(A4)

In our example, the desired magnetization is  $\langle \sigma_z^0 \rangle_{des}^{ss} = 0.4$  a constant value in the cost function. After substituting eqs. (A3) and (A4) in eq. (A2), the expression obtained after substituting them in eq. (A1), eq. (18) becomes as follows:

<span id="page-5-2"></span>
$$(g_1)_{k+1} = (g_1)_k + \delta(g_1)_k$$
  

$$(g_2)_{k+1} = (g_2)_k + \delta(g_2)_k.$$
 (A5)

Next, we substitute  $\nu = \theta$  in eq. (19) as

<span id="page-5-7"></span>

Regarding eq. (15), one can easily see that the magnetization of the *i*th reservoir is  $\langle \sigma_z \rangle_i = \langle \sigma_i^+ \sigma_i^- \rangle - \langle \sigma_i^- \sigma_i^+ \rangle$ . Therefore, azimuth parameter-dependent expression of the magnetization can be easily written as  $\langle \sigma_z \rangle_i = \cos \theta_i$ .

Equation (A7) is obtained when we take the partial derivative of the cost function with respect to  $\theta$ .

<span id="page-5-6"></span>
$$\frac{\partial C}{\partial \theta_i} = (\langle \sigma_z^0 \rangle_{des}^{ss} - \langle \sigma_z^0 \rangle_{act}^{ss}) \left( -\frac{\partial \langle \sigma_z^0 \rangle_{act}^{ss}}{\partial \theta_i} \right)$$
 (A7)

In our current example, we have two information reservoirs corresponding to specific magnetizations. Therefore, the actual steady state magnetization (eq. (13)) reads as

<span id="page-6-12"></span>
$$A = \langle \sigma_z^0 \rangle_{act}^{ss} = \frac{g_1^2 \cos \theta_1 + g_2^2 \cos \theta_2}{g_1^2 + g_2^2}.$$
 (A8)

According to the recipe to derive the cost function, the partial derivatives with respect to  $\theta_1$  and  $\theta_2$  separately obtained as

$$\frac{\partial A}{\partial \theta_1} = -\frac{g_1^2 \sin \theta_1}{g_1^2 + g_2^2}$$

$$\frac{\partial A}{\partial \theta_2} = -\frac{g_2^2 \sin \theta_2}{g_1^2 + g_2^2}$$
(A9)

In our example, the desired magnetization is  $\langle \sigma_z^0 \rangle_{des}^{ss} = 0$  a constant value in the cost function. After substituting eqs (A8) and (A9) in eq. (A7), the expression obtained after substituting them in eq. (A6), eq. (18) becomes as

follows:

$$(\theta)_{\mathbf{k+1}} = (\theta_{\mathbf{1}})_{\mathbf{k}} + \delta(\theta_{\mathbf{1}})_{\mathbf{k}}$$
  

$$(\theta_{\mathbf{2}})_{\mathbf{k+1}} = (\theta_{\mathbf{2}})_{\mathbf{k}} + \delta(\theta_{\mathbf{2}})_{\mathbf{k}}.$$
 (A10)

Let's edit Eq. (18) for  $\nu = \phi$ 

<span id="page-6-17"></span>

Equation (A12) is obtained when we take the partial derivative of the cost function with respect to  $\phi$ .

<span id="page-6-14"></span>
$$\frac{\partial C}{\partial \phi_i} = (\langle \sigma_y^0 \rangle_{des}^{ss} - \langle \sigma_y^0 \rangle_{act}^{ss})(-\frac{\partial \langle \sigma_y^0 \rangle_{act}^{ss}}{\partial \phi_i}) \tag{A12}$$

<span id="page-6-13"></span>In our current example, we have two information reservoirs corresponding to specific magnetizations. Therefore, the actual steady state magnetization (eq. (13)) by using eq. (15) reads as

$$A = \langle \sigma_y^0 \rangle_{act}^{ss} = -r\tau \frac{g_1^3 \sin \theta_1 \cos \theta_1 \cos \phi_1 + g_1 g_2^2 \sin \theta_1 \cos \theta_2 \cos \phi_1 + g_1^2 g_2 \cos \theta_1 \sin \theta_2 \cos \phi_2 + g_2^3 \sin \theta_2 \cos \theta_2 \cos \phi_2}{g_1^2 + g_2^2}.$$
(A13)

According to the recipe to derive the cost function, the partial derivatives with respect to  $\phi_1$  and  $\phi_2$  separately obtained as

$$\frac{\partial A}{\partial \phi_1} = r\tau \frac{g_1^3 \sin \theta_1 \cos \theta_1 \sin \phi_1 + g_1 g_2^2 \sin \theta_1 \cos \theta_2 \sin \phi_1}{g_1^2 + g_2^2} 
\frac{\partial A}{\partial \phi_2} = r\tau \frac{g_1^2 g_2 \cos \theta_1 \sin \theta_2 \sin \phi_2 + g_2^3 \sin \theta_2 \cos \theta_2 \sin \phi_2}{g_1^2 + g_2^2}$$
(A14)

In our example, the desired magnetization is  $\langle \sigma_y^0 \rangle_{des}^{ss} = 0$  a constant value in the cost function. After substituting eqs (A13) and (A14) in eq. (A12), the expression obtained after substituting them in eq. (A11), eq. (A15) becomes as follows:

<span id="page-6-18"></span><span id="page-6-15"></span>
$$(\phi)_{\mathbf{k}+\mathbf{1}} = (\phi_{\mathbf{1}})_{\mathbf{k}} + \delta(\phi_{\mathbf{1}})_{\mathbf{k}}$$
  

$$(\phi_{\mathbf{2}})_{\mathbf{k}+\mathbf{1}} = (\phi_{\mathbf{2}})_{\mathbf{k}} + \delta(\phi_{\mathbf{2}})_{\mathbf{k}}.$$
 (A15)

- <span id="page-6-0"></span>[1] W. S. McCulloch and W. Pitts, Bulletin of Mathematical Biophysics 5, 115 (1943).
- <span id="page-6-1"></span>[2] F. Rosenblatt, Psychological Review 65, 386 (1958).
- <span id="page-6-2"></span>[3] J. Misra and I. Saha, Neurocomputing **74**, 239 (2010).
- [4] J. Gu, Z. Wang, J. Kuen, L. Ma, A. Shahroudy, B. Shuai, T. Liu, X. Wang, G. Wang, J. Cai, and T. Chen, Pattern Recognition 77, 354 (2018).
- <span id="page-6-3"></span>[5] J. Schmidhuber, Neural Networks **61**, 85 (2015).
- <span id="page-6-4"></span>[6] C. H. Bennett and D. P. DiVincenzo, Nature 404, 247 (2000).
- <span id="page-6-5"></span>[7] A. Montanaro, npj Quantum Information 2, 1 (2016).
- <span id="page-6-6"></span>[8] L. Banchi, N. Pancotti, and S. Bose, npj Quantum Information 2, 16019 (2016).
- [9] A. Y. Yamamoto, K. M. Sundqvist, P. Li, and H. R. Harris, Quantum Inf Process 17, 128 (2018).
- [10] F. Tacchino, C. Macchiavello, D. Gerace, and D. Bajoni,

<span id="page-6-16"></span>npj Quantum Inf **5**, 1 (2019).

- [11] E. Torrontegui and J. J. García-Ripoll, EPL 125, 30004 (2019).
- [12] S. Mangini, F. Tacchino, D. Gerace, D. Bajoni, and C. Macchiavello, EPL 134, 10002 (2021).
- <span id="page-6-7"></span>[13] S. Yan, H. Qi, and W. Cui, Physical Review A **102**, 052421 (2020).
- <span id="page-6-8"></span>[14] M. Pechal, F. Roy, S. A. Wilkinson, G. Salis, M. Werninghaus, M. J. Hartmann, and S. Filipp, Physical Review Research 4, 033190 (2022).
- <span id="page-6-9"></span>[15] T. Nguyen, I. Paik, Y. Watanobe, and T. C. Thang, Electronics 11, 437 (2022).
- <span id="page-6-10"></span>[16] D. Türkpençe, T. Ç. Akıncı, and S. Şeker, Physics Letters A 383, 1410 (2019).
- <span id="page-6-11"></span>[17] U. Korkmaz and D. Türkpençe, Physics Letters A 426, 127887 (2022).

- [18] U. Korkmaz, C. Sanga, and D. Türkpençe, in 2021 5th International Symposium on Multidisciplinary Studies and lybricade Review & & 1018 (12917). (2021) pp. 105–109.
- <span id="page-7-0"></span>[19] U. Korkmaz, C. Sanga, and D. Türkpençe, in Electrical and Computer Engineering, Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, edited by M. N. Seyman (Springer International Publishing, 2022) pp. 159-170.
- <span id="page-7-1"></span>[20] F. Verstraete, M. M. Wolf, and J. Ignacio Cirac, Nature Phys 5, 633 (2009).
- <span id="page-7-2"></span>[21] S. Deffner and C. Jarzynski, Phys. Rev. X 3, 041003 (2013).
- <span id="page-7-3"></span>[22] S. Deffner, Phys. Rev. E 88, 062128 (2013).
- <span id="page-7-4"></span>Ziman, Ρ. Štelmachovič, V. Bužek, M.Hillery, V. Scarani, and N. Gisin, Physical Review A **65**, 042105 (2002).
- <span id="page-7-5"></span>[24] R. Blume-Kohout and W. Η. Zurek, Foundations of Physics 35, 1857 (2005).

- <span id="page-7-6"></span>[25] M. Zwolak and W. Η.
- <span id="page-7-7"></span>[26] V. Scarani, M. Ziman, P. Štelmachovič, N. Gisin, and V. Bužek, Physical Review Letters 88, 097905 (2002).
- <span id="page-7-8"></span>[27] D. Nagaj, P. Štelmachovič, V. Bužek, and M. Kim, Physical Review A 66, 062307 (2002).
- <span id="page-7-9"></span>[28] P. Krantz, M. Kjaergaard, F. Yan, S. Gustavsson, and W. D. Orlando, Oliver, Appl. Phys. Rev. 6, 021318 (2019).
- <span id="page-7-10"></span>[29] K. H. Wan, O. Dahlsten, H. Kristjánsson, R. Gardner, and M. S. Kim, npj Quantum Inf 3, 36 (2017).
- <span id="page-7-11"></span>[30] S. Ruder, An overview of gradient descent optimization algorithms (2017), arXiv:1609.04747 [cs].
- <span id="page-7-12"></span>[31] M. O. Scully, M. S. Zubairy, G. S. Agarwal, and H. Walther, Science **299**, 862 (2003).
- <span id="page-7-13"></span>Т. [32] D. Karevski and Platini, Physical Review Letters **102**, 207207 (2009).