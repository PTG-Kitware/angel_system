using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Cogobyte.SmartProceduralIndicators;

namespace Cogobyte.Demo.SmartProceduralIndicators
{

    public class OutlineDemo : MonoBehaviour
    {
        public SmartArrow mainArrow;
        public List<BodyRenderer> bodyRenderers;
        public List<ArrowTip> tails;
        public List<ArrowTip> heads;

        public SmartArrow[] shapeArrows;

        int currentBody = 0;
        int currentTail = 0;
        int currentHead = 0;

        BodyTipMeshItem tailConnector;
        BodyTipMeshItem headConnector;

        float moveAmount = 6f;

        public float Speed = 20f;
        float[] MoveAmountLeft = new float[3] { 0, 0, 0 };
        float rotateAngle = 0;

        // Start is called before the first frame update
        void Start()
        {
            InitializeShapesArrow((ShapesBodyRenderer)shapeArrows[0].bodyRenderers[0].bodyRenderer, bodyRenderers, Vector3.one * 3);
            InitializeShapesArrow((ShapesBodyRenderer)shapeArrows[1].bodyRenderers[0].bodyRenderer, tails, Vector3.one);
            InitializeShapesArrow((ShapesBodyRenderer)shapeArrows[2].bodyRenderers[0].bodyRenderer, heads, Vector3.one);
            mainArrow.arrowTail = tails[currentTail];
            mainArrow.arrowHead = heads[currentHead];
            SwitchBody();
        }

        public void InitializeShapesArrow(ShapesBodyRenderer s, IEnumerable sourceTemplates, Vector3 shapeScale)
        {
            foreach (var source in sourceTemplates)
            {
                s.shapes.Add(new ShapesBodyRenderer.MeshShape()
                {
                    spaceBefore = 3f,
                    spaceAfter = 3f,
                    offset = Vector3.zero,
                    rotation = new Vector3(0, 45, 45),
                    scale = shapeScale
                });
                s.shapes[s.shapes.Count - 1].meshes.Add(new MeshItem() { colorMode = MeshItem.MeshItemColorMode.VertexColor });
                s.shapes[s.shapes.Count - 1].meshes[s.shapes[s.shapes.Count - 1].meshes.Count - 1].LoadMesh(((MonoBehaviour)source).GetComponent<SmartArrow>().arrowRenderers[0].meshFilter.mesh);
                s.shapes[s.shapes.Count - 1].meshes[s.shapes[s.shapes.Count - 1].meshes.Count - 1].mesh.GenerateMesh();
            }

        }


        public void PreviosBody()
        {
            currentBody = (currentBody - 1 + bodyRenderers.Count) % bodyRenderers.Count;
            SwitchBody();
            MoveAmountLeft[0] += moveAmount;
        }

        public void NextBody()
        {
            currentBody = (currentBody + 1) % bodyRenderers.Count;
            SwitchBody();
            MoveAmountLeft[0] -= moveAmount;
        }

        public void PreviosTail()
        {
            currentTail = (currentTail - 1 + tails.Count) % tails.Count;
            MoveAmountLeft[1] += moveAmount;
            mainArrow.arrowTail = tails[currentTail];
            SwitchBody();
        }

        public void NextTail()
        {
            currentTail = (currentTail + 1) % tails.Count;
            MoveAmountLeft[1] -= moveAmount;
            mainArrow.arrowTail = tails[currentTail];
            SwitchBody();
        }

        public void PreviosHead()
        {
            currentHead = (currentHead - 1 + heads.Count) % heads.Count;
            MoveAmountLeft[2] += moveAmount;
            mainArrow.arrowHead = heads[currentHead];
            SwitchBody();
        }

        public void NextHead()
        {
            currentHead = (currentHead + 1) % heads.Count;
            MoveAmountLeft[2] -= moveAmount;
            mainArrow.arrowHead = heads[currentHead];
            SwitchBody();
        }


        public void SwitchBody()
        {
            mainArrow.bodyRenderers[0].bodyRenderer = bodyRenderers[currentBody];
            tailConnector = ((OutlineBodyRenderer)mainArrow.bodyRenderers[0].bodyRenderer).tailMeshConnector[0];
            headConnector = ((OutlineBodyRenderer)mainArrow.bodyRenderers[0].bodyRenderer).headMeshConnector[0];
            Outline o;
            if (tails[currentTail].GetType() == typeof(OutlineTip))
            {
                o = ((OutlineTip)tails[currentTail]).outline;
                Outline.GenerateFaceMesh(tailConnector, o, true);
            }
            if (heads[currentHead].GetType() == typeof(OutlineTip))
            {
                o = ((OutlineTip)heads[currentHead]).outline;
                Outline.GenerateFaceMesh(headConnector, o, false);
            }
            mainArrow.UpdateArrow();
        }


        // Update is called once per frame
        void Update()
        {
            rotateAngle += Time.deltaTime * 20;
            if (rotateAngle > 360) rotateAngle = 0;
            for (int i = 0; i < 3; i++)
            {
                if (Mathf.Abs(MoveAmountLeft[i]) < Time.deltaTime * Speed) { MoveAmountLeft[i] = 0; }
                else
                {
                    if (MoveAmountLeft[i] > 0)
                    {
                        shapeArrows[i].bodyRenderers[0].displacement += Time.deltaTime * Speed;
                        MoveAmountLeft[i] -= Time.deltaTime * Speed;
                    }
                    else if (MoveAmountLeft[i] < 0)
                    {
                        shapeArrows[i].bodyRenderers[0].displacement -= Time.deltaTime * Speed;
                        MoveAmountLeft[i] += Time.deltaTime * Speed;
                    }
                }
                shapeArrows[i].UpdateArrow();
            }
            ((ShapesBodyRenderer)shapeArrows[0].bodyRenderers[0].bodyRenderer).shapes[currentBody].rotation = new Vector3(0, rotateAngle, rotateAngle);
            ((ShapesBodyRenderer)shapeArrows[1].bodyRenderers[0].bodyRenderer).shapes[currentTail].rotation = new Vector3(0, rotateAngle, rotateAngle);
            ((ShapesBodyRenderer)shapeArrows[2].bodyRenderers[0].bodyRenderer).shapes[currentHead].rotation = new Vector3(0, rotateAngle, rotateAngle);

        }
    }
}