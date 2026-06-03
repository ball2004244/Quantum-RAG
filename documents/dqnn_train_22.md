## Trainability of Dissipative Perceptron-Based Quantum Neural Networks

<span id="page-0-0"></span>Kunal Sharma [,](https://orcid.org/0000-0003-3132-1088)1,2[,\\*](#page-5-0) M. Cerezo [,](https://orcid.org/0000-0002-2757-3170)1,3[,\\*](#page-5-0) Lukasz Cincio [,](https://orcid.org/0000-0002-6758-4376)1 and Patrick J. Coles1 1 Theoretical Division, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, USA <sup>2</sup> Hearne Institute for Theoretical Physics and Department of Physics and Astronomy, Louisiana State University, Baton Rouge, Louisiana 70803, USA <sup>3</sup> Center for Nonlinear Studies, Los Alamos National Laboratory, Los Alamos, New Mexico 87545, USA

(Received 31 July 2020; revised 22 September 2021; accepted 6 April 2022; published 6 May 2022)

Several architectures have been proposed for quantum neural networks (QNNs), with the goal of efficiently performing machine learning tasks on quantum data. Rigorous scaling results are urgently needed for specific QNN constructions to understand which, if any, will be trainable at a large scale. Here, we analyze the gradient scaling (and hence the trainability) for a recently proposed architecture that we call dissipative QNNs (DQNNs), where the input qubits of each layer are discarded at the layer's output. We find that DQNNs can exhibit barren plateaus, i.e., gradients that vanish exponentially in the number of qubits. Moreover, we provide quantitative bounds on the scaling of the gradient for DQNNs under different conditions, such as different cost functions and circuit depths, and show that trainability is not always guaranteed. Our work represents the first rigorous analysis of the scalability of a perceptron-based QNN.

DOI: [10.1103/PhysRevLett.128.180505](https://doi.org/10.1103/PhysRevLett.128.180505)

Introduction.—Neural networks (NN) have impacted many fields such as neuroscience, engineering, computer science, chemistry, and physics [[1](#page-5-1)]. However, their historical development has seen periods of great progress interleaved with periods of stagnation, due to serious technical challenges [[2](#page-5-2)]. The perceptron was introduced early on as an artificial neuron [\[3\]](#page-5-3), but it was only realized later that a multilayer perceptron (now known as a feed forward NN) had much greater power than a single-layer one [\[1,](#page-5-1)[2](#page-5-2)]. Still there was the major issue of how to train multiple layers, and this was eventually addressed by the backpropagation method [\[4\]](#page-5-4).

Motivated by the success of NNs and the advent of noisy intermediate-scale quantum devices [\[5](#page-5-5)], there has been tremendous effort to develop quantum neural networks (QNNs) [\[6\]](#page-5-6). The hope is that QNNs will harness the power of quantum computers to outperform their classical counterparts on machine learning tasks [[7,](#page-5-7)[8\]](#page-5-8), especially for quantum data or tasks that are inherently quantum in nature [[9](#page-5-9)].

Despite several QNN proposals that have been successfully implemented [\[10](#page-5-10)–[17](#page-5-11)], more research is needed on the advantages and limitations of specific architectures. Delving into potential scalability issues of QNNs could help to prevent a "winter" for these models, like what was seen historically for classical NNs. This has motivated recent works studying the scaling of gradients in QNNs [\[18](#page-5-12)[,19\]](#page-5-13). There, it was shown that variational quantum algorithms [[20](#page-5-14)–[30\]](#page-5-15), which aim to train QNNs to accomplish specific tasks, may exhibit gradients that vanish exponentially with the system size. This so-called barrenplateau phenomenon, where the parameters cannot be efficiently trained for large implementations, was demonstrated for hardware-efficient QNNs, where quantum gates are arranged in a bricklike structure that matches the connectivity of the quantum device [\[18,](#page-5-12)[19](#page-5-13)].

Analyzing the existence of barren plateaus in QNNs is paramount to determining if they can lead to a quantum speedup. This is due to the fact that exponentially vanishing gradients imply that the precision needed to estimate such gradients grows exponentially. Since the standard goal of quantum algorithms is polynomial scaling as opposed to the typical exponential scaling of classical algorithms, a QNN with exponentially vanishing gradients has no hope of achieving this goal. On the other hand, a QNN with gradients that vanish polynomially means that the algorithm requires a polynomial precision, and hence that the hope of quantum speedup is preserved.

Here, we analyze the trainability and the existence of barren plateaus in a class of QNNs that we refer to as dissipative QNNs (DQNNs). In a DQNN each node within the network corresponds to a qubit [\[31\]](#page-5-16), and the connections in the network are modelled by quantum perceptrons [\[32](#page-5-17)–[37](#page-6-0)]. The term dissipative refers to the fact that ancillary qubits form the output layer, while the qubits from the input layer are discarded. This architecture has seen significant recent attention and has been proposed as a scalable approach to QNNs [\[37](#page-6-0)–[39](#page-6-1)]. In particular, in Ref. [[37](#page-6-0)], based on small scale numerical experiments, it was speculated that dissipative quantum neural networks do not suffer from the barren plateau (vanishing gradient) problem. However, contrary to Ref. [\[37\]](#page-6-0), we here analytically prove that DQNNs are not immune to barren plateaus. For example, DQNNs with deep global perceptrons are untrainable despite the dissipative nature of the architecture.

Here we study the large-scale trainability of DQNNs. In particular, we focus on tasks where DQNNs are employed to learn a unitary matrix connecting input and output quantum states and for general supervised quantum machine learning tasks where training data consists of quantum states and corresponding classical labels. For these tasks, we show that the barren plateau phenomenon can also arise in DQNNs. We also discuss certain conditions (e.g., the structure and depth of the DQNN) under which one could avoid a barren plateau and achieve trainability. In particular, our work implies that scalability is not guaranteed, and without careful thought of the structure of DQNNs, their gradients may vanish exponentially in the system size. As a by-product of our analysis of specific perceptron architectures, we also show that hardware-efficient QNNs are special cases of DQNNs. Therefore, many important results for hardware-efficient QNNs, such as the ones studied in Refs. [18,19] also hold for DONNs. Finally, we remark that we employ novel analytical techniques in our proofs (different from those used in Refs. [18,19]), which were necessary to develop due to the dissipative nature of DQNNs. Our techniques may be broadly useful in the study of the scaling of other ONN architectures.

Preliminaries.—Let us first introduce the DQNN architecture. As schematically shown in Fig. 1, the DQNN is composed of a series of layers (input, hidden, and ouput)

<span id="page-1-0"></span>![](_page_1_Picture_4.jpeg)

FIG. 1. Schematic diagram of a dissipative perceptron-based quantum neural network (DQNN). Top: The DQNN is composed of input, hidden, and output layers. Each node in the network corresponds to a qubit, which can be connected to qubits in adjacent layers via perceptrons (depicted as lines). The input and output of the DQNN are quantum states denoted as  $\rho^{\rm in}$  and  $\rho^{\rm out}$ , respectively. Bottom: Quantum circuit description of the DQNN. The jth qubit of the lth layer is denoted  $q_j^l$ . Each perceptron corresponds to a unitary operation on the qubits it connects, with  $V_j^l$  denoting the jth perceptron in the lth layer.

where the qubits at each node are connected via perceptrons. A quantum perceptron is defined as an arbitrary unitary operator with m input and k output qubits. For simplicity, we consider the case when k = 1, so that each perceptron acts on m + 1 qubits. The case of arbitrary k is presented in the Supplemental Material [40].

The qubits in the input layer are initialized to a state  $\rho^{\rm in}$ , while all qubits in the hidden and output layers are initialized to a fiduciary state such as  $|\mathbf{0}\rangle_{\rm hid,out} = |0...0\rangle_{\rm hid,out}$ . Henceforth we employ the notation "in," "hid," and "out" to indicate operators on qubits in the input, hidden, and output layers, respectively. The output state of the DQNN is a quantum state  $\rho^{\rm out}$  (generally mixed) that can be expressed as

Let us now make two important remarks. First, note that the order in which the perceptrons act is relevant, as in general the unitaries  $V_j^l$  will not commute. Second, we remark that for this architecture the perceptrons are applied layer by layer, meaning that once all  $V_j^l$  (for fixed l) have been applied and the information has propagated forward between layers l-1 and l, one can discard the qubits in layer l-1. This implies that the width of the DQNN depends on the number of qubits in two adjacent layers and not in the total number of qubits in the network.

To train the DQNN, we assume repeatable access to training data in the form of pairs  $\{|\phi_x^{\rm in}\rangle, |\phi_x^{\rm out}\rangle\}$ , with x=1,...,N. We note that, as discussed in the Supplemental Material [40], our results also hold more generally for supervised quantum machine learning tasks where the training data is of the form  $\{|\phi_x^{\rm in}\rangle, y_x\}$ , with  $y_x$  a label assigned to the input state  $|\phi_x^{\rm in}\rangle$  [43].

We then define a cost function (or loss function) which quantifies how well the DQNN reproduces the training data. We assume that the cost is of the form

$$C = \frac{1}{N} \sum_{x=1}^{N} C_x, \quad \text{with} \quad C_x = \text{Tr}[O_x \rho_x^{\text{out}}]. \tag{2}$$

As discussed below, in general there are multiple choices for the operator  $O_x$  which lead to faithful cost functions, i.e., costs that are extremized if and only if one perfectly learns the mapping on the training data. If the circuit description of output states is provided, one can employ the inverse of the corresponding unitary on the output of a DQNN [44]. Then a measurement in the computational basis estimates the cost function. Otherwise, one can employ a recently developed procedure based on classical shadows to estimate the state overlap [45].

When  $O_x$  acts nontrivially on all qubits of the output layer, we use the term global cost function, denoted as  $C^G$ . Here one usually compares objects (states or operators) living in exponentially large Hilbert spaces. For instance, choosing

$$O_x^G = 1 - |\phi_x^{\text{out}}\rangle\langle\phi_x^{\text{out}}|,\tag{3}$$

<span id="page-2-0"></span>leads to a global cost function that quantifies the average fidelity between each  $\rho_x^{\text{out}}$  and  $|\phi_x^{\text{out}}\rangle$ .

As shown in Ref. [19], local cost functions do not exhibit a barren plateau for shallow hardware-efficient QNNs. Therefore, it is important to study if local observables can also lead to trainability guarantees in DQNNs. Henceforth, we use the term *local cost function*, denoted as  $C^L$ , for the cases when the operator  $O_x$  acts nontrivially on a small number of qubits in the output layer. Since the global cost in Eq. (3) is a state fidelity function, in general it will not be possible to design a corresponding faithful local cost. Therefore, we restrict ourselves to the case when  $|\phi_x^{\text{out}}\rangle = |\psi_{x,n}^{\text{out}}\rangle \otimes ... \otimes |\psi_{x,n_{\text{out}}}^{\text{out}}\rangle$ . Then, we can define the following local observable:

<span id="page-2-1"></span>
$$O_{x}^{L} = \mathbb{1} - \frac{1}{n_{\text{out}}} \sum_{i=1}^{n_{\text{out}}} |\psi_{x,j}^{\text{out}}\rangle \langle \psi_{x,j}^{\text{out}}| \otimes \mathbb{1}_{\bar{j}}, \tag{4}$$

where  $\mathbb{1}_{\bar{j}}$  denotes the identity over all qubits in the output layer except for qubit j. Equation (4) leads to a faithful local cost that vanishes under the same condition as the global cost defined from Eq. (3) [44,46].

Finally let us introduce the term *global perceptron* to refer to the case when the perceptron  $V_j^l$  acts nontrivially on *all* qubits in the *l*th layer, i.e., when  $m = n_{l-1}$ . On the other hand, a local perceptron is defined as a unitary  $V_j^l$  acting on a number of qubits  $m \in \mathcal{O}(1)$  which is independent of  $n_{l-1}$ . Figure 2 schematically shows a global and a local perceptron.

<span id="page-2-2"></span>![](_page_2_Figure_8.jpeg)

FIG. 2. Global and local perceptrons. (a) The global perceptron acts nontrivially on all input qubits, i.e., m = n. (b) The local perceptron acts nontrivially only on a small number of input qubits. For the case shown, m = 3.

To analyze the existence of barren plateaus and the trainability of the DQNN one needs to define an ansatz and a training method for the perceptrons. In what follows we consider two general training approaches.

<span id="page-2-3"></span>Random parametrized quantum circuits.—We first consider the case where the perceptrons are parametrized quantum circuits (i.e., variational circuits) that can be expressed as a sequence of parametrized and unparametrized gates from a given gate alphabet [18,47]. That is, the perceptrons are of the form

$$V_j^l(\boldsymbol{\theta}_j^l) = \prod_{k=1}^{\eta_j^l} R_k(\boldsymbol{\theta}^k) W_k, \tag{5}$$

with  $R_k(\theta^k) = e^{-(i/2)\theta^k\Gamma_k}$ ,  $W_k$  an unparametrized unitary, and where  $\Gamma_k$  is a Hermitian operator with  ${\rm Tr}[\Gamma_k^2] \leq 2^{n+1}$ . Such parametrization is widely used as it can allow for a straightforward evaluation of the cost function gradients, and since in general its quantum circuit description can be easily obtained [48–50].

A common strategy for training random parametrized quantum circuits is to randomly initialize the parameters in Eq. (5), and employ a training loop to minimize the cost function. To analyze the trainability of the DQNN we compute the variance of the partial derivative  $\partial C/\partial\theta^{\nu} \equiv \partial_{\nu}C$ , where  $\theta^{\nu}$  belongs to a given  $V_{i}^{l}$ 

$$Var[\partial_{\nu}C] = \langle (\partial_{\nu}C)^2 \rangle - \langle \partial_{\nu}C \rangle^2. \tag{6}$$

<span id="page-2-4"></span>Here the notation  $\langle \cdots \rangle$  indicates the average over all randomly initialized perceptrons. From Eq. (5), we find

$$\partial_{\nu}C = \frac{i}{2N} \sum_{x=1}^{N} \operatorname{Tr}[A_{j}^{l} \tilde{\rho}_{x}^{\text{in}} (A_{j}^{l})^{\dagger} [\mathbb{1}_{\bar{j}}^{\bar{l}} \otimes \Gamma_{k}, (B_{j}^{l})^{\dagger} \tilde{O}_{x} B_{j}^{l}]], \tag{7}$$

<span id="page-2-5"></span>where we have defined

$$B_j^l = \mathbb{I}_{\bar{j}}^{\bar{l}} \otimes \prod_{k=1}^{\nu-1} R_k(\theta^k) W_k, \quad A_j^l = \mathbb{I}_{\bar{j}}^{\bar{l}} \otimes \prod_{k=\nu}^{\eta_j^l} R_k(\theta^k) W_k, \quad (8)$$

such that  $\mathbb{1}^{\bar{l}}_{\bar{j}} \otimes V^l_j = A^l_j B^l_j$ , and where  $\mathbb{1}^{\bar{l}}_{\bar{j}}$  indicates the identity on all qubits on which  $V^l_j$  does not act. Note that the trace in Eq. (7) is over *all* qubits in the DQNN. In addition, we define

$$\begin{split} \tilde{\rho}_{x}^{\text{in}} &= V_{j-1}^{l}...V_{1}^{l}(\rho_{x}^{\text{in}} \otimes |\mathbf{0}\rangle\langle\mathbf{0}|_{\text{hid,out}})(V_{1}^{l})^{\dagger}...(V_{j-1}^{l})^{\dagger}, \\ \tilde{O}_{x} &= (V_{j+1}^{l})^{\dagger}...(V_{n_{\text{out}}}^{\text{out}})^{\dagger}(\mathbb{1}_{\text{in,hid}} \otimes O_{x})V_{n_{\text{out}}}^{\text{out}}...V_{j+1}^{l}. \end{split}$$

If the perceptron  $V_j^l$  is sufficiently random so that  $A_j^l$ ,  $B_j^l$ , or both, form independent unitary one-designs, then we find that  $\langle \partial_\nu C \rangle = 0$  (see Supplemental Material [40]). In this case,  $\operatorname{Var}[\partial_s C]$  quantifies (on average) how much the

gradient concentrates around zero. Hence, exponentially small  $Var[\partial_s C]$  values would imply that the slope of the cost function landscape is insufficient to provide cost-minimizing directions.

Here we recall that a t design is a set of unitaries  $\{V_y \in U(d)\}_{y \in Y}$  (of size |Y|) on a d-dimensional Hilbert space such that for every polynomial  $P_t(V_y)$  of degree at most t in the matrix elements of  $V_y$ , and of  $V_y^{\dagger}$  one has [51]  $\langle P_t(V)\rangle_V = (1/|Y|) \sum_{y \in Y} P_t(V_y) = \int d\mu(V) P_t(V)$ , where the integral is over the unitary group U(d).

Let us assume for simplicity the case when the DQNN input and output layers have the same number of qubits  $(n_{\rm in} = n_{\rm out} = n)$ . As shown in the Supplemental Material [40], the following theorem holds.

Theorem 1.—Consider a DQNN with deep global perceptrons parametrized as in Eq. (5), such that  $A_j^l$ ,  $B_j^l$  in Eq. (8) and  $V_j^l$  ( $\forall j, l$ ) form independent two-designs over n+1 qubits. Then, the variance of the partial derivative of the cost function with respect to  $\theta^\nu$  in  $V_j^l$  is upper bounded as

if  $O_x$  is the global operator of Eq. (3), and upper bounded as

$$\operatorname{Var}[\partial_{\nu}C^{L}] \le h(n), \quad \text{with} \quad h(n) \in \mathcal{O}(1/2^{n}), \quad (10)$$

when  $O_x$  is the local operator in Eq. (4).

Theorem 1 shows that DQNNs with deep global perceptron unitaries that form two-designs [52,53] exhibit barren plateaus for global and local cost functions. An immediate question that follows is whether barren plateaus still arise for shallow perceptrons, which cannot form two-designs on n + 1 qubits. In what follows we analyze specific cases of shallow local perceptrons for which results can be obtained.

Let us first consider the simple perceptrons of Fig. 3(a), where m=1, and where  $R_y$  denotes a single qubit rotation around the y axis:  $R_y(\theta^\nu)=e^{-i\theta^\nu Y/2}$  (with all angles randomly initialized). In this case one recovers the toy model example of Ref. [19], and we know that if  $O_x$  is the global operator of Eq. (3), then  $\operatorname{Var}[\partial_\nu C^G]=\frac{1}{8}(\frac{3}{8})^{n-1}$ . On the other hand, if  $O_x$  is the local operator in Eq. (4), then  $\operatorname{Var}[\partial_\nu C^L]=[1/(8n^2)]$ .

These results suggest that DQNNs with simple shallow local perceptrons and global cost functions are untrainable when randomly initialized. On the other hand, they also indicate that barren plateaus for DQNNs might be avoided by employing: (i) shallow (local) perceptrons, and (ii) local cost functions.

Let us now consider the shallow local perceptron of Fig. 3(b), where each unitary W forms a local two-design on two qubits. For this architecture the ensuing DQNN can be *exactly* mapped into a layered hardware-efficient ansatz as in Ref. [19], where two layers of the DQNN correspond

<span id="page-3-0"></span>![](_page_3_Picture_13.jpeg)

FIG. 3. Shallow local perceptrons ansatzes. (a) Here m=1 so that each perceptron acts on a single input and output qubit. Moreover, for all j and l we have  $V_j^l = V$ . The unitaries V are simply given by a SWAP operator followed by a single qubit rotation around the y axis. (b) Local perceptrons  $V_j^l$  with m=2. The local perceptrons are given by the unitaries  $V_1$ , or  $V_2$ . Specifically, for l odd on j odd (even)  $V_j^l = V_1(V_2)$ , while for l even and j odd (even) we have  $V_j^l = V_2(V_1)$ . Here we also show the order in which the perceptrons are applied so that we first implement the unitaries with j odd, followed by the unitaries with j even. The W gate in  $V_1$  forms a local two-design on two qubits.

to a single layer of the hardware-efficient ansatz [54]. Note that this mapping is not general, but rather valid for the specific architecture in Fig. 3(b). As shown in Ref. [19], when employing a global cost function, with  $O_x$  given by Eq. (3), one finds that if the number is layers is  $\mathcal{O}(\text{poly}[\log(n)])$ , then the DQNN cost function exhibits barren plateaus as

$$\operatorname{Var}[\partial_{\nu}C^{G}] \le \hat{f}(n), \quad \text{with} \quad \hat{f}(n) \in \mathcal{O}[(\sqrt{3}/4)^{n}].$$
 (11)

<span id="page-3-1"></span>On the other hand, for a local cost function with  $O_x$  given by Eq. (4), if the number of layers is in  $\mathcal{O}[\log(n)]$ , then there is no barren plateau [19] as

$$\hat{g}(n) \le \operatorname{Var}[\partial_{\nu}C^{L}], \quad \text{with} \quad \hat{g}(n) \in \Omega[1/\operatorname{poly}(n)].$$
 (12)

Here we remark that Eq. (12) was obtained following the same assumptions as those used in Corollary 2 of Ref. [19]. Note that obtaining a lower bound for the variance implies that the DQNN trainability is guaranteed.

Parameter matrix multiplication.—While in random parametrized quantum circuits one optimizes and trains a single gate angle at a time, other optimization approaches can also be considered. In what follows we analyze the

results a superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the superior of the supe

<span id="page-4-0"></span>In this training approach, which we call parameter matrix multiplication, the perceptrons are not explicitly decomposed into quantum circuits, but rather are treated as unitary matrices. The perceptrons  $V_j^l(0)$  are randomly initialized at time-step zero, and at each step s they are updated via

The matrices  $H^l_j$  are such that  $\mathrm{Tr}[(H^l_j)^2] \leq 2^{n+1}$  and are parameterized as  $H^l_j(s) = \sum_{uv} h^l_{j,u,v} X^u Z^v$ , with  $X^u Z^v = X_1^{u_1} Z_1^{v_1} \otimes X_2^{u_2} Z_2^{v_2} \ldots$ , and where  $X_j$  and  $Z_j$  are Pauli operators on qubit j. The matrices  $K^l_j(s)$  are called parameter matrices, and at each time step the coefficients  $h^l_{j,u,v}$  need to be optimized. As shown in the Supplemental Material [40], if at least one perceptron  $V^l_j(0)$  is sufficiently random so that it forms a global unitary one-design, then we find  $\langle \partial C/\partial s \rangle \equiv \langle \partial_s C \rangle = 0$ .

As proved in the Supplemental Material [40], the following theorem holds.

Theorem 2.—Consider a DQNN with deep global perceptrons, which are updated via the parameter matrix multiplication of Eq. (13). Suppose that for all j, l, the  $V_j^l(0)$  perceptrons form independent two-designs over n+1 qubits. Then the variance of the partial derivative of the cost function with respect to the time-step parameter s is upper bounded as

when  $O_x$  is the global operator of Eq. (3), or the local operator in Eq. (4).

Although the updating method in Eq. (13) simultaneously updates all perceptrons at each time step, Theorem 2 implies that barren plateaus also arise when using the parameter matrix multiplication method.

We note that our proof techniques invoke the pure state properties of input and output states. Since the output state of a randomly initialized DQNN will be close to a maximally mixed across any bipartite cut [55], we speculate that our results can be extended to expectation values of the arbitrary Hamiltonian. We leave this question for future work.

Conclusions.—In this Letter, we analyzed the trainability of a special class of QNNs called DQNNs. We first proved that the trainability of DQNNs is not always guaranteed as they can exhibit barren plateaus in their cost function landscape. The existence of such barren plateaus was linked to the localities (i.e., the number of qubits they act nontrivially on) of the perceptrons and of the cost function. Specifically, we showed that (i) DQNNs with deep global perceptrons are untrainable despite the

dissipative nature of the architecture, and (ii) for shallow and local perceptrons, employing global cost functions leads to barren plateaus, while using local costs avoids them. We note that our results are completely general for DQNN architectures, e.g., covering arbitrary numbers of hidden layers and general perceptrons acting on any number of qubits.

In addition, we provided a specific architecture for DQNNs with local shallow perceptrons that can be exactly mapped to a layered hardware-efficient ansatz. This result not only indicates that hardware-efficient QNNs can be represented as DQNNs, but it also allows us to derive trainability guarantees for these DQNNs. In this case, since the perceptrons are local, each neuron only receives information from a small number of qubits in the previous layer. Such architecture is reminiscent of classical convolutional neural networks, which are known to avoid some of the trainability problems of fully connected networks [56].

These results show that much work needs to be done to understand the trainability of QNNs and guarantee that they can provide a quantum speedup over classical neural networks. For instance, interesting future research directions are QNN-specific optimizers [57–60], analyzing the resilience of QNNs to noise [22,44], and strategies to prevent barren plateaus [61–64]. Another interesting direction is to extend our results to the case when the input and output states are mixed states, particularly when the goal is to match marginals of the target output state and the output of a DQNN [65]. Furthermore, exploring architectures beyond DQNNs and hardware-efficient QNNs would be of interest, particularly if such architectures have large-scale trainability.

We thank Jarrod McClean, Tobias Osborne, and Andrew Sornborger for helpful conversations. All authors acknowledge support from LANL's Laboratory Directed Research and Development (LDRD) program. M. C. was also supported by the Center for Nonlinear Studies at LANL. P. J. C. also acknowledges support from the LANL ASC Beyond Moore's Law project. This work was also supported by the U.S. Department of Energy (DOE), Office of Science, Office of Advanced Scientific Computing Research, under the Accelerated Research in Quantum Computing (ARQC) program.

Note added.—Our work is the first to analyze barren plateaus in the context of data science applications, and also the first to consider perceptron-based quantum neural networks. Our work has inspired more recent studies of trainability for other QNN architectures, such as quantum convolutional neural networks [66], tree-based architectures [67], and others [68–70]. We also note that our results can also be interpreted as a type of entanglement-induced barren plateau. Here, a large amount of entanglement in a

parametrized quantum circuit can lead to trainability issues when qubits are discarded, and the output qubits are concentrated around the maximally mixed state. This phenomenon was further studied in Refs. [\[55](#page-6-14)[,71\]](#page-6-25).

- <span id="page-5-1"></span><span id="page-5-0"></span>[\\*](#page-0-0) K. S. and M. C. contributed equally to this work.
- <span id="page-5-2"></span>[1] Simon Haykin, Neural Networks: A Comprehensive Foundation (Prentice Hall PTR, NJ, 1994).
- <span id="page-5-3"></span>[2] Marvin Minsky and Seymour A. Papert, Perceptrons: An Introduction to Computational Geometry (MIT Press, Cambridge, MA, 2017).
- <span id="page-5-4"></span>[3] Frank Rosenblatt, The Perceptron, a Perceiving and Recognizing Automaton Project Para (Cornell Aeronautical Laboratory, 1957).
- <span id="page-5-5"></span>[4] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams, Learning representations by back-propagating errors, [Nature \(London\)](https://doi.org/10.1038/323533a0) 323, 533 (1986).
- <span id="page-5-6"></span>[5] J. Preskill, Quantum computing in the NISQ era and beyond, Quantum 2[, 79 \(2018\)](https://doi.org/10.22331/q-2018-08-06-79).
- <span id="page-5-7"></span>[6] Maria Schuld, Ilya Sinayskiy, and Francesco Petruccione, The quest for a quantum neural network, [Quantum Inf.](https://doi.org/10.1007/s11128-014-0809-8) Process. 13[, 2567 \(2014\).](https://doi.org/10.1007/s11128-014-0809-8)
- <span id="page-5-8"></span>[7] Michael A. Nielsen, Neural Networks and Deep Learning (Determination Press San Francisco, CA, USA, 2015), Vol. 2018.
- <span id="page-5-9"></span>[8] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd, Quantum machine learning, [Nature \(London\)](https://doi.org/10.1038/nature23474) 549, 195 (2017).
- [9] Kunal Sharma, M. Cerezo, Zoë Holmes, Lukasz Cincio, Andrew Sornborger, and Patrick J. Coles, Reformulation of the No-Free-Lunch Theorem for Entangled Data Sets, [Phys.](https://doi.org/10.1103/PhysRevLett.128.070501) Rev. Lett. 128[, 070501 \(2022\).](https://doi.org/10.1103/PhysRevLett.128.070501)
- <span id="page-5-10"></span>[10] J. Romero, J. P. Olson, and A. Aspuru-Guzik, Quantum autoencoders for efficient compression of quantum data, [Quantum Sci. Technol.](https://doi.org/10.1088/2058-9565/aa8072) 2, 045001 (2017).
- [11] Vedran Dunjko and Hans J. Briegel, Machine learning & artificial intelligence in the quantum domain: A review of recent progress, [Rep. Prog. Phys.](https://doi.org/10.1088/1361-6633/aab406) 81, 074001 (2018).
- [12] Guillaume Verdon, Jason Pye, and Michael Broughton, A universal training algorithm for quantum deep learning, [arXiv:1806.09729.](https://arXiv.org/abs/1806.09729)
- [13] Edward Farhi and Hartmut Neven, Classification with quantum neural networks on near term processors, [arXiv:](https://arXiv.org/abs/1802.06002) [1802.06002.](https://arXiv.org/abs/1802.06002)
- [14] Carlo Ciliberto, Mark Herbster, Alessandro Davide Ialongo, Massimiliano Pontil, Andrea Rocchetto, Simone Severini, and Leonard Wossnig, Quantum machine learning: A classical perspective, [Proc. R. Soc. A](https://doi.org/10.1098/rspa.2017.0551) 474, 20170551 [\(2018\).](https://doi.org/10.1098/rspa.2017.0551)
- [15] Nathan Killoran, Thomas R. Bromley, Juan Miguel Arrazola, Maria Schuld, Nicolás Quesada, and Seth Lloyd, Continuous-variable quantum neural networks, [Phys. Rev.](https://doi.org/10.1103/PhysRevResearch.1.033063) Research 1[, 033063 \(2019\).](https://doi.org/10.1103/PhysRevResearch.1.033063)
- <span id="page-5-11"></span>[16] Iris Cong, Soonwon Choi, and Mikhail D. Lukin, Quantum convolutional neural networks, Nat. Phys. 15[, 1273 \(2019\).](https://doi.org/10.1038/s41567-019-0648-8)
- [17] Zhih-Ahn Jia, Biao Yi, Rui Zhai, Yu-Chun Wu, Guang-Can Guo, and Guo-Ping Guo, Quantum neural network states: A

- brief review of methods and applications, [Adv. Quantum](https://doi.org/10.1002/qute.201800077) Technol. 2[, 1800077 \(2019\).](https://doi.org/10.1002/qute.201800077)
- <span id="page-5-12"></span>[18] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven, Barren plateaus in quantum neural network training landscapes, [Nat. Commun.](https://doi.org/10.1038/s41467-018-07090-4) 9[, 4812 \(2018\).](https://doi.org/10.1038/s41467-018-07090-4)
- <span id="page-5-13"></span>[19] Marco Cerezo, Akira Sone, Tyler Volkoff, Lukasz Cincio, and Patrick J. Coles, Cost function dependent barren plateaus in shallow parametrized quantum circuits, [Nat.](https://doi.org/10.1038/s41467-021-21728-w) Commun. 12[, 1791 \(2021\)](https://doi.org/10.1038/s41467-021-21728-w).
- <span id="page-5-14"></span>[20] A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O'Brien, A variational eigenvalue solver on a photonic quantum processor, [Nat. Commun.](https://doi.org/10.1038/ncomms5213) 5, 4213 (2014).
- [21] Bela Bauer, Dave Wecker, Andrew J. Millis, Matthew B. Hastings, and Matthias Troyer, Hybrid Quantum-Classical Approach to Correlated Materials, [Phys. Rev. X](https://doi.org/10.1103/PhysRevX.6.031045) 6, 031045 [\(2016\).](https://doi.org/10.1103/PhysRevX.6.031045)
- <span id="page-5-18"></span>[22] Jarrod R. McClean, Jonathan Romero, Ryan Babbush, and Alán Aspuru-Guzik, The theory of variational hybrid quantum-classical algorithms, [New J. Phys.](https://doi.org/10.1088/1367-2630/18/2/023023) 18, 023023 [\(2016\).](https://doi.org/10.1088/1367-2630/18/2/023023)
- [23] A. Arrasmith, L. Cincio, A. T. Sornborger, W. H. Zurek, and P. J. Coles, Variational consistent histories as a hybrid algorithm for quantum foundations, [Nat. Commun.](https://doi.org/10.1038/s41467-019-11417-0) 10, [3438 \(2019\)](https://doi.org/10.1038/s41467-019-11417-0).
- [24] Tyson Jones, Suguru Endo, Sam McArdle, Xiao Yuan, and Simon C. Benjamin, Variational quantum algorithms for discovering hamiltonian spectra, [Phys. Rev. A](https://doi.org/10.1103/PhysRevA.99.062304) 99, 062304 [\(2019\).](https://doi.org/10.1103/PhysRevA.99.062304)
- [25] X. Xu, J. Sun, S. Endo, Y. Li, S. C. Benjamin, and X. Yuan, Variational algorithms for linear algebra, [Sci. Bull.](https://doi.org/10.1016/j.scib.2021.06.023) 66, 2181 [\(2021\).](https://doi.org/10.1016/j.scib.2021.06.023)
- [26] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio, and Patrick J. Coles, Variational quantum linear solver: A hybrid algorithm for linear systems, [arXiv:1909.05820](https://arXiv.org/abs/1909.05820).
- [27] Xiao Yuan, Suguru Endo, Qi Zhao, Ying Li, and Simon C. Benjamin, Theory of variational quantum simulation, [Quan](https://doi.org/10.22331/q-2019-10-07-191)tum 3[, 191 \(2019\).](https://doi.org/10.22331/q-2019-10-07-191)
- [28] Cristina Cirstoiu, Zoe Holmes, Joseph Iosue, Lukasz Cincio, Patrick J. Coles, and Andrew Sornborger, Variational fast forwarding for quantum simulation beyond the coherence time, [npj Quantum Inf.](https://doi.org/10.1038/s41534-020-00302-0) 6, 82 (2020).
- [29] Marco Cerezo, Alexander Poremba, Lukasz Cincio, and Patrick J. Coles, Variational quantum fidelity estimation, Quantum 4[, 248 \(2020\)](https://doi.org/10.22331/q-2020-03-26-248).
- <span id="page-5-15"></span>[30] M. Cerezo, Kunal Sharma, Andrew Arrasmith, and Patrick J. Coles, Variational quantum state eigensolver, [arXiv:](https://arXiv.org/abs/2004.01372) [2004.01372.](https://arXiv.org/abs/2004.01372)
- <span id="page-5-16"></span>[31] Noriaki Kouda, Nobuyuki Matsui, Haruhiko Nishimura, and Ferdinand Peper, Qubit neural network and its learning efficiency, [Neural Comput. Appl.](https://doi.org/10.1007/s00521-004-0446-8) 14, 114 (2005).
- <span id="page-5-17"></span>[32] MV Altaisky, Quantum neural network, [arxiv:quant-ph/](https://arXiv.org/abs/arxiv:quant-ph/0107012) [0107012.](https://arXiv.org/abs/arxiv:quant-ph/0107012)
- [33] Alaa Sagheer and Mohammed Zidan, Autonomous quantum perceptron neural network, [arXiv:1312.4149.](https://arXiv.org/abs/1312.4149)
- [34] Michael Siomau, A quantum model for autonomous learning automata, [Quantum Inf. Process.](https://doi.org/10.1007/s11128-013-0723-5) 13, 1211 [\(2014\).](https://doi.org/10.1007/s11128-013-0723-5)

- [35] Erik Torrontegui and Juan Jos´e García-Ripoll, Unitary quantum perceptron as efficient universal approximator, [Europhys. Lett.](https://doi.org/10.1209/0295-5075/125/30004) 125, 30004 (2019).
- [36] Francesco Tacchino, Chiara Macchiavello, Dario Gerace, and Daniele Bajoni, An artificial neuron implemented on an actual quantum processor, [Quantum Inf.](https://doi.org/10.1038/s41534-019-0140-4) 5, 1 (2019).
- <span id="page-6-0"></span>[37] Kerstin Beer, Dmytro Bondarenko, Terry Farrelly, Tobias J. Osborne, Robert Salzmann, Daniel Scheiermann, and Ramona Wolf, Training deep quantum neural networks, [Nat. Commun.](https://doi.org/10.1038/s41467-020-14454-2) 11, 808 (2020).
- [38] Dmytro Bondarenko and Polina Feldmann, Quantum Autoencoders to Denoise Quantum Data, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.124.130502) 124[, 130502 \(2020\).](https://doi.org/10.1103/PhysRevLett.124.130502)
- <span id="page-6-1"></span>[39] Kyle Poland, Kerstin Beer, and Tobias J. Osborne, No free lunch for quantum machine learning, [arXiv:2003.14103](https://arXiv.org/abs/2003.14103).
- <span id="page-6-2"></span>[40] See Supplemental Material at [http://link.aps.org/](http://link.aps.org/supplemental/10.1103/PhysRevLett.128.180505) [supplemental/10.1103/PhysRevLett.128.180505](http://link.aps.org/supplemental/10.1103/PhysRevLett.128.180505) for details of our proofs and Refs. [41,42].
- [41] Zbigniew Puchała and Jaroslaw Adam Miszczak, Symbolic integration with respect to the haar measure on the unitary groups, [Bull. Pol. Acad. Sci.](https://doi.org/10.1515/bpasts-2017-0003) 65, 21 (2017).
- [42] Motohisa Fukuda, Robert König, and Ion Nechita, RTNI—a symbolic integrator for Haar-random tensor networks, J. Phys. A 52[, 425303 \(2019\).](https://doi.org/10.1088/1751-8121/ab434b)
- <span id="page-6-3"></span>[43] Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M Gambetta, Supervised learning with quantumenhanced feature spaces, [Nature \(London\)](https://doi.org/10.1038/s41586-019-0980-2) 567, 209 (2019).
- <span id="page-6-4"></span>[44] Kunal Sharma, Sumeet Khatri, Marco Cerezo, and Patrick J. Coles, Noise resilience of variational quantum compiling, New J. Phys. 22[, 043006 \(2020\)](https://doi.org/10.1088/1367-2630/ab784c).
- <span id="page-6-5"></span>[45] Hsin-Yuan Huang, Richard Kueng, and John Preskill, Predicting many properties of a quantum system from very few measurements, Nat. Phys. 16[, 1050 \(2020\).](https://doi.org/10.1038/s41567-020-0932-7)
- <span id="page-6-6"></span>[46] S. Khatri, R. LaRose, A. Poremba, L. Cincio, A. T. Sornborger, and P. J. Coles, Quantum-assisted quantum compiling, Quantum 3[, 140 \(2019\)](https://doi.org/10.22331/q-2019-05-13-140).
- <span id="page-6-7"></span>[47] Yuxuan Du, Min-Hsiu Hsieh, Tongliang Liu, and Dacheng Tao, The expressive power of parameterized quantum circuits, [Phys. Rev. Research](https://doi.org/10.1103/PhysRevResearch.2.033125) 2, 033125 (2020).
- <span id="page-6-8"></span>[48] Gian Giacomo Guerreschi and Mikhail Smelyanskiy, Practical optimization for hybrid quantum-classical algorithms, [arXiv:1701.01450](https://arXiv.org/abs/1701.01450).
- [49] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, Quantum circuit learning, Phys. Rev. A 98[, 032309 \(2018\)](https://doi.org/10.1103/PhysRevA.98.032309).
- <span id="page-6-9"></span>[50] Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran, Evaluating analytic gradients on quantum hardware, Phys. Rev. A 99[, 032331 \(2019\).](https://doi.org/10.1103/PhysRevA.99.032331)
- <span id="page-6-10"></span>[51] Christoph Dankert, Richard Cleve, Joseph Emerson, and Etera Livine, Exact and approximate unitary 2-designs and their application to fidelity estimation, [Phys. Rev. A](https://doi.org/10.1103/PhysRevA.80.012304) 80, [012304 \(2009\).](https://doi.org/10.1103/PhysRevA.80.012304)
- <span id="page-6-11"></span>[52] Fernando G. S. L. Brandao, Aram W. Harrow, and Michał Horodecki, Local random quantum circuits are approximate polynomial-designs, [Commun. Math. Phys.](https://doi.org/10.1007/s00220-016-2706-8) 346, 397 (2016).
- <span id="page-6-12"></span>[53] Aram Harrow and Saeed Mehraban, Approximate unitary t-designs by short random quantum circuits using nearestneighbor and long-range gates, [arXiv:1809.06957.](https://arXiv.org/abs/1809.06957)
- <span id="page-6-13"></span>[54] A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow, and J. M. Gambetta, Hardware-efficient

- variational quantum eigensolver for small molecules and quantum magnets, [Nature \(London\)](https://doi.org/10.1038/nature23879) 549, 242 (2017).
- <span id="page-6-14"></span>[55] Carlos Ortiz Marrero, Mária Kieferová, and Nathan Wiebe, Entanglement induced barren plateaus, [PRX Quantum](https://doi.org/10.1103/PRXQuantum.2.040316) 2, [040316 \(2022\).](https://doi.org/10.1103/PRXQuantum.2.040316)
- <span id="page-6-15"></span>[56] Neena Aloysius and M. Geetha, A review on deep convolutional neural networks, in 2017 International Conference on Communication and Signal Processing (ICCSP) (IEEE, 2017), pp. 0588–0592.
- <span id="page-6-16"></span>[57] James Stokes, Josh Izaac, Nathan Killoran, and Giuseppe Carleo, Quantum natural gradient, [Quantum](https://doi.org/10.22331/q-2020-05-25-269) 4, 269 [\(2020\).](https://doi.org/10.22331/q-2020-05-25-269)
- [58] Jonas M. Kübler, Andrew Arrasmith, Lukasz Cincio, and Patrick J. Coles, An adaptive optimizer for measurementfrugal variational algorithms, Quantum 4[, 263 \(2020\)](https://doi.org/10.22331/q-2020-05-11-263).
- [59] Bálint Koczor and Simon C. Benjamin, Quantum natural gradient generalised to non-unitary circuits, [arXiv:1912](https://arXiv.org/abs/1912.08660) [.08660.](https://arXiv.org/abs/1912.08660)
- <span id="page-6-17"></span>[60] Andrew Arrasmith, Lukasz Cincio, Rolando D. Somma, and Patrick J. Coles, Operator sampling for shot-frugal optimization in variational algorithms, [arXiv:2004.06252](https://arXiv.org/abs/2004.06252).
- <span id="page-6-18"></span>[61] Ryan LaRose, Arkin Tikku, Étude O'Neel-Judy, Lukasz Cincio, and Patrick J. Coles, Variational quantum state diagonalization, [npj Quantum Inf.](https://doi.org/10.1038/s41534-019-0167-6) 5, 57 (2019).
- [62] Edward Grant, Leonard Wossnig, Mateusz Ostaszewski, and Marcello Benedetti, An initialization strategy for addressing barren plateaus in parametrized quantum circuits, Quantum 3[, 214 \(2019\).](https://doi.org/10.22331/q-2019-12-09-214)
- [63] Guillaume Verdon, Michael Broughton, Jarrod R. McClean, Kevin J. Sung, Ryan Babbush, Zhang Jiang, Hartmut Neven, and Masoud Mohseni, Learning to learn with quantum neural networks via classical neural networks, [arXiv:1907.05415.](https://arXiv.org/abs/1907.05415)
- <span id="page-6-19"></span>[64] Tyler Volkoff and Patrick J. Coles, Large gradients via correlation in random parameterized quantum circuits, [Quantum Sci. Technol.](https://doi.org/10.1088/2058-9565/abd891) 6, 025008 (2021).
- <span id="page-6-20"></span>[65] Adrien Bolens and Markus Heyl, Reinforcement Learning for Digital Quantum Simulation, [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.127.110502) 127, [110502 \(2021\).](https://doi.org/10.1103/PhysRevLett.127.110502)
- <span id="page-6-21"></span>[66] Arthur Pesah, M. Cerezo, Samson Wang, Tyler Volkoff, Andrew T. Sornborger, and Patrick J. Coles, Absence of barren plateaus in quantum convolutional neural networks, Phys. Rev. X 11[, 041011 \(2022\)](https://doi.org/10.1103/PhysRevX.11.041011).
- <span id="page-6-22"></span>[67] Kaining Zhang, Min-Hsiu Hsieh, Liu Liu, and Dacheng Tao, Toward trainability of quantum neural networks, [arXiv:2011.06258.](https://arXiv.org/abs/2011.06258)
- <span id="page-6-23"></span>[68] Chen Zhao and Xiao-Shan Gao, Analyzing the barren plateau phenomenon in training quantum neural networks with the ZX-calculus, Quantum 5[, 466 \(2021\).](https://doi.org/10.22331/q-2021-06-04-466)
- [69] Samson Wang, Enrico Fontana, Marco Cerezo, Kunal Sharma, Akira Sone, Lukasz Cincio, and Patrick J. Coles, Noise-induced barren plateaus in variational quantum algorithms, [Nat. Commun.](https://doi.org/10.1038/s41467-021-27045-6) 12, 6961 (2021).
- <span id="page-6-24"></span>[70] Amira Abbas, David Sutter, Christa Zoufal, Aur´elien Lucchi, Alessio Figalli, and Stefan Woerner, The power of quantum neural networks, [Nat. Comput. Sci.](https://doi.org/10.1038/s43588-021-00084-1) 1, 403 [\(2021\).](https://doi.org/10.1038/s43588-021-00084-1)
- <span id="page-6-25"></span>[71] Taylor L. Patti, Khadijeh Najafi, Xun Gao, and Susanne F. Yelin, Entanglement devised barren plateau mitigation, [Phys. Rev. Research](https://doi.org/10.1103/PhysRevResearch.3.033090) 3, 033090 (2021).