Task Monitor Nodes
==================
There are two versions of the task monitor node, v1 and v2. Both nodes take in
ActivityDetection messages and publish TaskUpdate messages that represent the
current state of a task.

task_monitor_v1
+++++++++++++++
task_monitor_v1 is the first iteration of the task monitor that uses the Python
transitions package to create a state machine representing a task. Transitions
are triggered by activity classifications to proceed to the next step.

task_monitor_v2
+++++++++++++++
task_monitor_v2 is the next iteration of that task monitor that uses a Hidden
Markov Model (HMM) to deduce step transitions and determine the current step.
