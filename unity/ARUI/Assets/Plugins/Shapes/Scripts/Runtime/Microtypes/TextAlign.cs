using TMPro;
using UnityEngine;

// Shapes © Freya Holmér - https://twitter.com/FreyaHolmer/
// Website & Documentation - https://acegikmo.com/shapes/
namespace Shapes {

	public enum TextAlign {
		TopLeft,
		Top,
		TopRight,
		Left,
		Center,
		Right,
		BottomLeft,
		Bottom,
		BottomRight
	}

	public static class TextAlignExtensions {
		public static Vector2 GetPivot( this TextAlign align ) {
			switch( align ) {
				case TextAlign.TopLeft:     return new Vector2( 0f, 1f );
				case TextAlign.Top:         return new Vector2( 0.5f, 1f );
				case TextAlign.TopRight:    return new Vector2( 1f, 1f );
				case TextAlign.Left:        return new Vector2( 0f, 0.5f );
				case TextAlign.Center:      return new Vector2( 0.5f, 0.5f );
				case TextAlign.Right:       return new Vector2( 1f, 0.5f );
				case TextAlign.BottomLeft:  return new Vector2( 0f, 0f );
				case TextAlign.Bottom:      return new Vector2( 0.5f, 0f );
				case TextAlign.BottomRight: return new Vector2( 1f, 0f );
			}

			return default;
		}

		public static TextAlignmentOptions GetTMPAlignment( this TextAlign align ) {
			switch( align ) {
				case TextAlign.TopLeft:     return TextAlignmentOptions.TopLeft;
				case TextAlign.Top:         return TextAlignmentOptions.Top;
				case TextAlign.TopRight:    return TextAlignmentOptions.TopRight;
				case TextAlign.Left:        return TextAlignmentOptions.Left;
				case TextAlign.Center:      return TextAlignmentOptions.Center;
				case TextAlign.Right:       return TextAlignmentOptions.Right;
				case TextAlign.BottomLeft:  return TextAlignmentOptions.BottomLeft;
				case TextAlign.Bottom:      return TextAlignmentOptions.Bottom;
				case TextAlign.BottomRight: return TextAlignmentOptions.BottomRight;
			}

			return default;
		}
	}

}