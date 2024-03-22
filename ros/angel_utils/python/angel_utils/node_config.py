from collections import namedtuple
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from rclpy.executors import ExternalShutdownException
import rclpy.node


__all__ = [
    "declare_and_get_parameters",
]


NodeType = TypeVar("NodeType", bound=rclpy.node.Node)


def make_default_main(
    node_type: Type[NodeType],
    node_args: Sequence[Any] = (),
    node_kwargs: Optional[Dict[str, Any]] = None,
    pre_spin_callback: Optional[Callable[[NodeType], None]] = None,
    multithreaded_executor: Optional[int] = None,
) -> Callable[[], None]:
    """
    Convenient ROS2 python node main-function to provide a default
    implementation and reference to be expanded upon for advanced use-cases.

    This function can be imported into your top-level
        >>> from angel_utils.node_config import make_default_main
        >>> main = make_default_main()

    :param node_type: The class type, or sequence of class types, of the Node
        derived class or classes that we are instantiating and spinning over.
    :param node_args: Positional arguments to provide to the constructor of the
        node type given above. If multiple node types are specified, then this
        should be a sequence of positional argument tuples of equivalent length
        and each will be provided to node type constructors in parallel
        association. This is empty by default.
    :param node_kwargs: Key-word arguments in the form of a `dict` to provide
        to the consturctor of the node type given above.
        If multiple node types are specified, then this should be a sequence of
        dictionaries of equivalent length and each will be provided to node
        type constructors in parallel assiciation.
        This is empty by default as represented by a None
        input value.
    :param pre_spin_callback: Callback function that will be invoked after
        initializing rclpy and constructing the node instance, but before
        starting to spin. This callback with be given the node instance as a
        positional parameter.
    :param multithreaded_executor: If specified, a multithreaded executor will
        be used and this integer specifies the number of threads to use in the
        executor. Otherwise, a single-threaded executor will be used by
        default.

    :raises ValueError: If `note_type` is provided as a sequence of types and
        either `node_args` or `node_kwargs` are not sequences of an equivalent
        length.

    :returns: Function closure to act as the main function that takes zero
        arguments.
    """

    def closure() -> None:
        # Initialize ROS2
        rclpy.init()
        log = rclpy.logging.get_logger("main")

        # Construct the node instance
        node = node_type(*node_args, **(node_kwargs or {}))

        # Execute pre-spin callback, if provided
        if pre_spin_callback is not None:
            pre_spin_callback(node)

        # Create the executor
        executor = (
            rclpy.executors.MultiThreadedExecutor(num_threads=multithreaded_executor)
            if multithreaded_executor is not None
            else rclpy.executors.SingleThreadedExecutor()
        )

        # Add the node to the executor
        executor.add_node(node)

        try:
            # Spin the node
            executor.spin()
        except (KeyboardInterrupt, ExternalShutdownException):
            log.warn("Interrupt/shutdown signal received.")
        finally:
            # Destroy the node explicitly
            # (optional - otherwise it will be done automatically
            # when the garbage collector destroys the node object)
            node.destroy_node()

            log.info("Final try-shutdown")
            rclpy.try_shutdown()

    return closure


# ROS2 Iron concept.
# # Convenience instance of a ParameterDescriptor with the dynamic_typing field
# # set to True.
# DYNAMIC_TYPE = rclpy.node.ParameterDescriptor(dynamic_typing=True)


def declare_and_get_parameters(
    node: rclpy.node.Node,
    name_default_tuples: Sequence[
        Union[
            Tuple[str],
            Tuple[str, rclpy.node.Parameter.Type],
            Tuple[str, Any, rclpy.node.ParameterDescriptor],
        ]
    ],
    namespace: str = "",
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
            # ROS2 Foxy Specification
            ("parameter1_name",),      # <-- no default value
            ("parameter2_name", 2.5),  # <-- Default int value of 2.5

            # !!!
            # The following is valid only for ROS2 Iron
            # !!!

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
        parameters=name_default_tuples,
        # ROS2 Iron support
        # # Declaring a parameter only providing its name is deprecated. This
        # # seems to do with static-typing parameters by default and not having a
        # # default value to deduce that typing from. If nothing is given, we
        # # declare dynamic typing in a description object.
        # parameters=(
        #     t if len(t) > 1 else (t[0], None, DYNAMIC_TYPE) for t in name_default_tuples
        # ),
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
    namespace: str = "",
) -> type:
    """
    Name-tuple returning version of `declare_and_get_parameters`.

    See also: ``declare_and_get_parameters``

    :param node: Node instance to declare parameters into and get values out of.
    :param name_default_tuples: Parameters with optional default values.
    :param namespace: Namespace value to be passed into
        `node.declare_parameters`.

    :raises ValueError: Some input parameters were not given default values and
        were not set.

    TODO: Make this return a NamedTyple type appropriately.

    :returns: Dictionary of parameter names to their input or default values.
    """
    param_values = declare_and_get_parameters(node, name_default_tuples, namespace)
    return namedtuple(
        "ParamContainer",
        field_names=param_values.keys(),
        defaults=param_values.values(),
    )
