using UnityEngine;

namespace Cogobyte.SmartProceduralIndicators
{
    /// <summary>
    /// Object used to save the outline and edit it standalone.
    /// </summary>
    [CreateAssetMenu(fileName = "OutlineTemplate", menuName = "Cogobyte/SmartIndicators/OutlineTemplate", order = 1)]
    public class OutlineTemplate : ScriptableObject
    {
        public Outline outline = new Outline();
    }
}