from collections import namedtuple
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import rclpy.node


__all__ = [
    "declare_and_get_parameters",
]


# Convenience instance of a ParameterDescriptor with the dynamic_typing field
# set to True.
DYNAMIC_TYPE = rclpy.node.ParameterDescriptor(dynamic_typing=True)


def declare_and_get_parameters(
    node: rclpy.node.Node,
    name_default_tuples: Sequence[
        Union[
            Tuple[str],
            Tuple[str, rclpy.node.Parameter.Type],
            Tuple[str, Any, rclpy.node.ParameterDescriptor],
        ]
    ],
    namespace="",
) -> Dict[str, Any]:
    """
    Helper method to declare some number of parameters on a given node with
    optional default values, immediately querying those parameters for values
    (as would be set on the CLI) and returning a dictionary mapping the
    parameter names to the values read in.

    The `name_default_tuples` input specifies the parameters to be declared and
    retrieved with optional default values. This argument follows the
    documented requirements for the ``parameters`` argument of
    ``rclpy.node.Node.declare_parameters``, except for we coerce
    parameter-name-only specifications as being dynamically typed.

    Examples of allowed values for the ``name_default_tuples`` argument are::

        (
            # No default value, requires CLI to provide one. Any type of input
            # is accepted from the CLI when only a parameter name is provided.
            ("parameter1_name",),

            # Default double value, user CLI may provide an override of this
            # value, but override values provided must match the type of the
            # default value (double in this case).
            ("parameter2_name", 2.5),

            # No default value, but the user is required to provide an integer
            # value via the CLI, otherwise parameter processing will yield an
            # error.
            ("parameter3_name", Parameter.Type.INTEGER),

            # Specify a parameter to have no default value but be allowed to
            # received values of any type, e.g. integer, double, string, etc.
            # The ``None`` value in between the name and the ``DYNAMIC_TYPE``
            # constant is important is what signifies no default value.
            ("parameter4_name", None, DYNAMIC_TYPE),

            # Specify a parameter to *have* a default value but also be dynamic
            # to user-override value types. E.g. the default value is a string,
            # but marking this ``DYNAMIC_TYPE`` means the user could provide a
            # number instead.
            ("parameter5_name", "some_filename", DYNAMIC_TYPE),

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
        # Declaring a parameter only providing its name is deprecated. This
        # seems to do with static-typing parameters by default and not having a
        # default value to deduce that typing from. If nothing is given, we
        # declare dynamic typing in a description object.
        parameters=(
            t if len(t) > 1 else (t[0], None, DYNAMIC_TYPE) for t in name_default_tuples
        ),
    )
    # Check for not-set parameters
    params_not_set = []
    for p in parameters:
        if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
            params_not_set.append(p)
            log.error(f"Parameter not set: {p.name}")
    if params_not_set:
        raise ValueError("Some input parameters are not set.")

    # Log parameters
    log.info("Input parameters:")
    for p in parameters:
        log.info(f"- {p.name} = ({p.type_}) {p.value}")

    return {p.name: p.value for p in parameters}


def declare_and_get_parameters_nt(
    node: rclpy.node.Node,
    name_default_tuples: Sequence[Union[Tuple[str], Tuple[str, Any]]],
    namespace="",
):
    """
    Name-tuple returning version of `declare_and_get_parameters`.

    See also: ``declare_and_get_parameters``

    :param node: Node instance to declare parameters into and get values out of.
    :param name_default_tuples: Parameters with optional default values.
    :param namespace: Namespace value to be passed into
        `node.declare_parameters`.

    :raises ValueError: Some input parameters were not given default values and
        were not set.

    :returns: Dictionary of parameter names to their input or default values.
    """
    param_values = declare_and_get_parameters(node, name_default_tuples, namespace)
    return namedtuple(
        "ParamContainer",
        field_names=param_values.keys(),
        defaults=param_values.values(),
    )
