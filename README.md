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
