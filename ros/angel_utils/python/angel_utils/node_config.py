from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import rclpy.node


__all__ = [
    "declare_and_get_parameters",
]


def declare_and_get_parameters(
    node: rclpy.node.Node,
    name_default_tuples: Sequence[Union[Tuple[str], Tuple[str, Any]]],
    namespace="",
) -> Dict[str, Any]:
    """
    Helper method to declare some number of parameters on a given node with
    optional default values, immediately querying those parameters for values
    (as would be set on the CLI) and returning a dictionary mapping the
    parameter names to the values read in.

    The `name_default_tuples` input specifies the parameters to be declared and
    retrieved with optional default values. We expect the format:

        (
            ("parameter1_name",),      # <-- no default value
            ("parameter2_name", 2.5),  # <-- Default int value of 2.5
            ...
        )

    This function will log the values of parameter retrieved to the info
    channel using the logger object retrieved from the given node.

    An exception is thrown if any input parameters that do not have default
    values are not set.

    :param node: Node instance to declare parameters into and get values out of.
    :param name_default_tuples: Parameters with optional default values.
    :param namespace: Namespace value to be passed into
        `node.declare_parameters`.

    :raises ValueError: Some input parameters were not given default values and
        were not set.

    :returns: Dictionary of parameter names to their input or default values.
    """
    log = node.get_logger()
    parameters = node.declare_parameters(
        namespace=namespace,
        parameters=name_default_tuples,
    )
    # Check for not-set parameters
    some_not_set = False
    for p in parameters:
        if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
            some_not_set = True
            log.error(f"Parameter not set: {p.name}")
    if some_not_set:
        raise ValueError("Some input parameters are not set.")

    # Log parameters
    log.info("Input parameters:")
    for p in parameters:
        log.info(f"- {p.name} = ({p.type_}) {p.value}")

    return {p.name: p.value for p in parameters}
