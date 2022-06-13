using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

// Shapes © Freya Holmér - https://twitter.com/FreyaHolmer/
// Website & Documentation - https://acegikmo.com/shapes/
namespace Shapes {

	// find matching shader+keywords+render state
	// if it exists

	internal struct RenderState : IEquatable<RenderState> {

		public Shader shader;
		public string[] keywords; // this is gross

		public CompareFunction zTest;
		public float zOffsetFactor;
		public int zOffsetUnits;
		public CompareFunction stencilComp;
		public StencilOp stencilOpPass;
		public byte stencilRefID;
		public byte stencilReadMask;
		public byte stencilWriteMask;

		public RenderState( Material mat ) {
			shader = mat.shader;
			keywords = mat.shaderKeywords;
			zTest = (CompareFunction)mat.GetInt( ShapesMaterialUtils.propZTest );
			zOffsetFactor = mat.GetFloat( ShapesMaterialUtils.propZOffsetFactor );
			zOffsetUnits = mat.GetInt( ShapesMaterialUtils.propZOffsetUnits );
			stencilComp = (CompareFunction)mat.GetInt( ShapesMaterialUtils.propStencilComp );
			stencilOpPass = (StencilOp)mat.GetInt( ShapesMaterialUtils.propStencilOpPass );
			stencilRefID = (byte)mat.GetInt( ShapesMaterialUtils.propStencilID );
			stencilReadMask = (byte)mat.GetInt( ShapesMaterialUtils.propStencilReadMask );
			stencilWriteMask = (byte)mat.GetInt( ShapesMaterialUtils.propStencilWriteMask );
		}

		public Material CreateMaterial() {
			Material mat = new Material( shader ) { shaderKeywords = keywords };
			mat.SetInt( ShapesMaterialUtils.propZTest, (int)zTest );
			mat.SetFloat( ShapesMaterialUtils.propZOffsetFactor, zOffsetFactor );
			mat.SetInt( ShapesMaterialUtils.propZOffsetUnits, zOffsetUnits );
			mat.SetInt( ShapesMaterialUtils.propStencilComp, (int)stencilComp );
			mat.SetInt( ShapesMaterialUtils.propStencilOpPass, (int)stencilOpPass );
			mat.SetInt( ShapesMaterialUtils.propStencilID, stencilRefID );
			mat.SetInt( ShapesMaterialUtils.propStencilReadMask, stencilReadMask );
			mat.SetInt( ShapesMaterialUtils.propStencilWriteMask, stencilWriteMask );
			mat.enableInstancing = true;
			Object.DontDestroyOnLoad( mat );
			return mat;
		}

		static bool StrArrEquals( string[] a, string[] b ) {
			if( a == null || b == null )
				return a == b;
			return a.Length == b.Length && a.SequenceEqual( b );
		}

		public bool Equals( RenderState other ) =>
			Equals( shader, other.shader ) &&
			StrArrEquals( keywords, other.keywords ) &&
			zTest == other.zTest &&
			zOffsetFactor.Equals( other.zOffsetFactor ) &&
			zOffsetUnits == other.zOffsetUnits &&
			stencilComp == other.stencilComp &&
			stencilOpPass == other.stencilOpPass &&
			stencilRefID == other.stencilRefID &&
			stencilReadMask == other.stencilReadMask &&
			stencilWriteMask == other.stencilWriteMask;

		public override bool Equals( object obj ) => obj is RenderState other && Equals( other );

		public override int GetHashCode() {
			unchecked {
				int hashCode = ( shader != null ? shader.GetHashCode() : 0 );
				if( keywords != null ) {
					foreach( string kw in keywords )
						hashCode = ( hashCode * 397 ) ^ ( kw != null ? kw.GetHashCode() : 0 );
				}

				hashCode = ( hashCode * 397 ) ^ (int)zTest;
				hashCode = ( hashCode * 397 ) ^ zOffsetFactor.GetHashCode();
				hashCode = ( hashCode * 397 ) ^ zOffsetUnits;
				hashCode = ( hashCode * 397 ) ^ (int)stencilComp;
				hashCode = ( hashCode * 397 ) ^ (int)stencilOpPass;
				hashCode = ( hashCode * 397 ) ^ stencilRefID.GetHashCode();
				hashCode = ( hashCode * 397 ) ^ stencilReadMask.GetHashCode();
				hashCode = ( hashCode * 397 ) ^ stencilWriteMask.GetHashCode();
				return hashCode;
			}
		}

	}

}