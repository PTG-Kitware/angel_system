/// Copyright by Rob Jellinghaus.  All rights reserved.

namespace Holofunk.HandPose
{
    /// <summary>
    /// Which pose is each finger in?
    /// </summary>
    /// <remarks>
    /// Curled is as in making a fist; Extended is straight out; Unknown is anything in between.
    /// </remarks>
    public enum FingerPose
    {
        /// <summary>
        /// We don't know what pose this finger is in.
        /// </summary>,
        Unknown,
        
        /// <summary>
        /// We are pretty sure this finger is curled up (as when making a fist).
        /// </summary>
        Curled,

        /// <summary>
        /// We are pretty sure this finger is extended more or less straight out.
        /// </summary>
        Extended
    }

    /// <summary>
    /// For each pair of fingers, how extended and adjacent are they?
    /// </summary>
    /// <remarks>
    /// This is calculated by determining how colinear the fingers are; if two adjacent fingers
    /// are highly colinear, they're guaranteed to be pointing in the same direction, hence together.
    /// </remarks>
    public enum FingerPairExtension
    {
        /// <summary>
        /// We don't know how close this pair of fingers are.
        /// </summary>
        Unknown,

        /// <summary>
        /// We are pretty confident these two fingers are extended side by side.
        /// </summary>
        ExtendedTogether,

        /// <summary>
        /// We are pretty confident these two fingers are NOT extended side by side.
        /// </summary>
        NotExtendedTogether
    }

    /// <summary>
    /// What overall shape do we think the hand is in?
    /// </summary>
    /// <remarks>
    /// This list of poses is heavily informed by what is easy to recognize with some trivial linear
    /// algebra, intersecting with what the HL2 can reliably detect.
    /// </remarks>
    public enum HandPose
    {
        /// <summary>
        /// No particular idea what shape the hand is in.
        /// </summary>
        Unknown,

        /// <summary>
        /// Pretty sure hand is open with all fingers extended and separated.
        /// </summary>
        Opened,
        
        /// <summary>
        /// Pretty sure hand is closed more or less into a fist.
        /// </summary>
        /// <remarks>
        /// If the hand is closed into a fist with fingers on the other side of the hand from the device, the device
        /// is prone to guess that the occluded fingers are extended. So we determine whether the finger vertices are
        /// colinear with a vector from the eye to the knuckle; if so, they are on the other side of the palm and we
        /// err on the side of assuming the hand is closed.
        /// </remarks>
        Closed,

        /// <summary>
        /// Pretty sure the hand is pointing with index finger only.
        /// </summary>
        PointingIndex,

        /// <summary>
        /// Pretty sure the hand is pointing with middle finger only.
        /// </summary>
        /// <remarks>
        /// This is likely enough to be a rude gesture that if the user does this a lot, they should
        /// be warned to cut it out.
        /// </remarks>
        PointingMiddle,

        /// <summary>
        /// Pretty sure hand is pointing with index and middle fingers adjacent.
        /// </summary>
        /// <remarks>
        /// Note that HL2 gets very unreliable at seeing the ring and pinky fingers precisely, for example
        /// it can't reliably see pointing with index, middle, and ring, and nor can it see the Vulcan greeting
        /// gesture.
        /// </remarks>
        PointingIndexAndMiddle,

        /// <summary>
        /// Bringing all fingertips together above the palm; the "bloom" gesture.
        /// </summary>
        Bloom,

        /// <summary>
        /// Pretty sure hand is fully flat with all fingers extended and adjacent.
        /// </summary>
        Flat,

        /// <summary>
        /// Thumbs up!
        /// </summary>
        ThumbsUp,
    }

    /// <summary>
    /// Which finger is which?
    /// </summary>
    public enum Finger
    {
        Thumb,
        Index,
        Middle,
        Ring,
        Pinky,
        Max = Pinky
    }
}