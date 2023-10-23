/// Copyright by Rob Jellinghaus.  All rights reserved.

using System;

namespace Holofunk.Core
{
    class HoloDebugException : Exception
    {
        public HoloDebugException(string message) : base(message) 
        {
            HoloDebug.Log(message);
        }
    }

    public class HoloDebug
    {
        public static void Log(string message)
        {
            UnityEngine.Debug.Log(message);
        }

        public static void Assert(bool shouldBeTrue, string message = "")
        {
            if (!shouldBeTrue)
            {
                throw new HoloDebugException(message);
            }
        }
    }
}
