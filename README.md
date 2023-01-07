Vehicle Routing Problem

Assumptions:
    - Either Multiple Trucks or Multiple Routes

Problem:
    Find the cluster of nodes (pathway) with minimum distance. We have clients as nodes and a centre known as deot. the trucks have to supply products ti these clients keeping the minimum distance wiage.

Groups:
    - Clients(C) : the destinations including depot
    - Nodes(N) : destinations excluding the depot
    - Arches(A) : Stored/Recorded Clusters :
        - The pair of nodes is stored as a multi-dimensional array.
        - The distance from node to itself is neglected/prohibited.
        - Each node may or maynot be visited multiple times.
    - c[i,j] : distance between node i and j.
    - v[i] : Volume of Products to Transfer at node i.
    - Q : Capacity of Vehicle

Decisional Variables:
    - x[i,j] (binary) :
            - True, if node is activated, meaning it appears some where in the cluster
            - False, if when it is not
    - u[i] (integer) :
            - stores the capacity required by the node i. Remeber i is a member of Clients(C).

Objective Function:
    Min. Z = Sigma[i belongs to A](x[i,j])*Sigma[j belongs to A](c[i,j])

Restrictions:
    - Each node can be visited once in any cluster
    - Each node can be visted by only one node during a cluster
    - Limited number of vehicles
    - Limited numebr of Routes
    - Restrict number of loaded vehicles
    - Higlighting pathway of current cluster

Drawback:
    Number of constraints is exponentially proportional to number of nodes.

## Dependencies


* Numpy
* [tensorflow](https://www.tensorflow.org/)>=1.2
* tqdm

## How to Run
### Train
By default, the code is running in the training mode on a single gpu. For running the code, one can use the following command:
```bash
python main.py --task=vrp10
```

It is possible to add other config parameters like:
```bash
python main.py --task=vrp10 --gpu=0 --n_glimpses=1 --use_tanh=False 
```
There is a full list of all configs in the ``config.py`` file. Also, task specific parameters are available in ``task_specific_params.py``
### Inference
For running the trained model for inference, it is possible to turn off the training mode. For this, you need to specify the directory of the trained model, otherwise random model will be used for decoding:
```bash
python main.py --task=vrp10 --is_train=False --model_dir=./path_to_your_saved_checkpoint
```
The default inference is run in batch mode, meaning that all testing instances are fed simultanously. It is also possible to do inference in single mode, which means that we decode instances one-by-one. The latter case is used for reporting the runtimes and it will display detailed reports. For running the inference with single mode, you can try:
```bash
python main.py --task=vrp10 --is_train=False --infer_type=single --model_dir=./path_to_your_saved_checkpoint
```
### Logs
All logs are stored in ``result.txt`` file stored in ``./logs/task_date_time`` directory.
## Sample CVRP solution

![enter image description here](https://lh3.googleusercontent.com/eUh69ZQsIV4SIE6RjwasAEkdw2VZaTmaeR8Fqk33di70-BGU62fvmcp6HLeGLE61lJDS7jLMpFf2 "Sample VRP")