# thesis
Master Thesis

Decentralized Control Policy with Imitation Learning for Multiple Multirotors
Transporting Cable Suspended PayloadsUncrewed aerial vehicles (UAVs) are ideal for robust autonomous tasks that involve accessing remote locations,
shipping objects or executing construction tasks. The agility of the multirotors allows them to collaboratively
execute complex applications such as carrying a payload in uncertain environments being transported by cables.
This application is extremely beneficial at construction sites for rubble removal in search-and-rescue scenarios,
and nuclear power plant decommissioning.
In dynamic environments, centralized control algorithms typically manage all robots from a single point,
providing precise coordination. However, this comes at the expense of communication overhead and notable
scalability challenges. Centralized optimization-based controllers, like nonlinear model predictive control (NMPC),
have been developed to control multirotors transporting payloads while avoiding collisions [1], [2]. However, these
methods often face scalability limitations due to the exhaustive computational demands they impose.
Classical decentralized control algorithms address these issues by distributing decision-making among multiro-
tors, enhancing scalability and resilience [3], [4]. However, Some of the devised control algorithms do not include
collision constraints, or they often struggle with the problem of local minima and deadlocks.
In our previous works [5], [6], we have developed a methodology that integrates a centralized offline motion
planner with a decentralized controller, designed to transport a point mass using multiple aerial multirotors through
environments cluttered with obstacles. This hybrid approach addresses the limitations of purely centralized and
decentralized controllers by leveraging the comprehensive planning capabilities of the centralized solution and
the adaptive execution of the decentralized controller. However, this strategy is primarily effective in static
environments, as it relies on following a pre-determined, offline plan and lacks the flexibility to adapt to dynamic
changes. Moreover, while the decentralized controller is efficient, it simplifies the inter-robot collision problem by
convexifying the constraints, which can sometimes lead to infeasible solutions.
Imitation Learning (IL) can replicate the strategies of an expensive planner [7], [8], effectively substituting the
optimal control or planning solver with a function that approximates these solutions. The goal of this thesis is to
utilize imitation learning using DAGGER to develop a decentralized control policy for the multirotors transporting
a pointmass payload that overcomes the limitations of local minima. Specifically, we use a centralized kinodynamic
motion planner as the global policy (the expert), which the local policy aims to emulate. The process begins by
generating feasible trajectories [6] for various motion planning scenarios and environments. Subsequently, a local
observation model is employed to compile a dataset of observation-action pairs. Finally, we apply imitation
learning to create a local policy that mimics the global expert, thereby avoiding local minima. The imitation
learning control policy will be compared to two baselines: the full framework of the previous works [5], [6] and the
nonlinear model predictive control of the payload system.
