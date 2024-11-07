# Multi-node

For multi-node experiments, these steps must be followed (shown for the 2-VM example of Fig 8e):

* Create two VMs, vm1 and vm2. The VMs should be on the same cloud zone. We name IP1 the **INTERNAL_IP** of vm1, and IP2 the **INTERNAL_IP** of vm2.

* SSH to vm1 (`gcloud compute ssh vm1`)

* # IMPORTANT! Make sure vm1 has ssh-access to vm2.
      To do so, from within vm1:
        * `gcloud compute init`: to authenticate
        * `gcloud compute ssh vm2`: to allow ssh to vm2 and generate ssh keys. Then `exit` to get back to vm1's shell. This step should have generated keys under `~/.ssh`
        * Create an ~/.ssh/config file and put two entries, one for vm1, and one for vm2. It should look as follows:

        ```
        Host IP1
        HostName IP1
        User "your_username"
        IdentityFile ~/.ssh/google_compute_engine

        Host IP2
        HostName IP2
        User "your_username"
        IdentityFile ~/.ssh/google_compute_engine

        ```
        * Make sure `ssh IP1` and `ssh IP2` work from within vm1 without issues.

* Run bash `get_throughput_multi_node.sh IP1 IP2`. This file:
        * Creates a `hostfile` (used by DeepSpeed to start execution at the two nodes) and a `~/.deepspeed_env` file (containing necessary env variables).
        * Runs all baselines for OPT-2.7B, generates csv files and plots for each model.

This can be adapted for any other model. We currently support distributed checkpointing for DeepSpeed.