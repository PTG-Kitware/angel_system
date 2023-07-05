__all__ = ["JOINT_LIST"]


# List containing joint names that matches the ordering in the HL2SS
# SI_HandJointKind class. Names semantically match the output from the MRTK API
# though the ordering of the joints is matching that of the windows perception
# API.
# MRTK API: https://learn.microsoft.com/en-us/dotnet/api/microsoft.mixedreality.toolkit.utilities.trackedhandjoint?preserve-view=true&view=mixed-reality-toolkit-unity-2020-dotnet-2.8.0
# Windows Perception API: https://learn.microsoft.com/en-us/uwp/api/windows.perception.people.handjointkind?view=winrt-22621
# Matching the names of the MRTK API for downstream components to continue to
# match against.
JOINT_LIST = [
    "Palm",
    "Wrist",
    "ThumbMetacarpalJoint",
    "ThumbProximalJoint",
    "ThumbDistalJoint",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexKnuckle",
    "IndexMiddleJoint",
    "IndexDistalJoint",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleKnuckle",
    "MiddleMiddleJoint",
    "MiddleDistalJoint",
    "MiddleTip",
    "RingMetacarpal",
    "RingKnuckle",
    "RingMiddleJoint",
    "RingDistalJoint",
    "RingTip",
    "PinkyMetacarpal",
    "PinkyKnuckle",
    "PinkyMiddleJoint",
    "PinkyDistalJoint",
    "PinkyTip",
]
