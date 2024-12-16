## research
[IL research](https://docs.google.com/document/d/1qL__5ltoS9RlNtAtyIXOkichVQ9TchGlXO6cpNntQVI/edit?usp=sharing)

## notes
date: 16-12-24

- progress:
	- fixed inital target state (step function was correct)
    - get trajectory from splines 
    - 3d trajectory (spring)
    - changed observation (pos error, vel error, q)
- next steps:
    - change dagger class
    - enable hyperparameter sweeps
- questions:	
  - 
#### dagger performance

| Algorithm | Mean State Error | Std State Error | Mean Velocity Error | Std Velocity Error | Online Burden |
|-----------|-----------------:|----------------:|---------------------:|-------------------:|--------------:|
| DAgger    |        0.05183   |        0.00848  |             0.30400  |           0.12037  |          4200 |
| Thrifty   |          0.09467 |        0.03861  |             0.54615  |           0.19676  |          1095 |


| algo    | mean state error | std state error | mean vel error |       std vel error | online burden |
|---------|:----------------:|----------------:|---------------:|--------------------:|-----------------:|
| dagger  |   0.0518273774799341  |           0.008478114069800105 |          0.3040043319296858 | 0.12036612045213002 |             4200 |
| thrifty |     0.09467390017545278  |            0.03860778658453085 |          0.5461478591497376 |  0.1967641566728824 |             1095 |

online burden thirfty: 1095
online burden dagger: 4200
dagger with fixed target state
![dagger_trajectory.png](images/dagger_trajectory.png)
## old notes


date: 09-12-24


- progress:
	- implemented [thrifty dagger](https://arxiv.org/abs/2109.08273)
- next steps:
    - implement diff dagger
- questions:	
  - thrifty dagger 

### dagger performance
![dagger_trajectory.png](images/24-12-8/dagger_trajectory.png)
![dagger_state_error.png](images/24-12-8/dagger_state_error.png)
![dagger_velocity_error.png](images/24-12-8/dagger_velocity_error.png)

### thrifty-dagger performance
![thrifty_trajectory.png](images/24-12-8/thrifty_trajectory.png)
![thrifty_state_error.png](images/24-12-8/thrifty_state_error.png)
![thrifty_velocity_error.png](images/24-12-8/thrifty_velocity_error.png)



date: 02-12-24

- progress:
	- (circle) trajectory tracking for pd controller
	- db_cbs as expert policy
- next steps:
    - fix ending for db_cbs as expert policy
	- multiple robots at once
- questions:
	- how to handle new state for expert policy
	    (right now i use the closest state (euclidian) but
	    that doesnt work when it overshoots)

date: 25-11-24

- progress:
	- pid controller works properly
	- dagger trains on multiple envs with different target states
- next steps:
    - evaluate model with new problem environment
    - improve reward and cost function?
- questions:
	- is adding target state as the observation for training correct?
	    -- in papers local observation was used for training
    - should i work on reward and cost function?