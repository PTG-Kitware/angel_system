using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Cogobyte.Demo.ProceduralIndicators
{
    //Controlls for ball object
    public class RollBallControls : MonoBehaviour
    {
        float speed = 20f;
        public TraveledPath travelScript;
        void Start()
        {
        }

        void Update()
        {
            //WSAD movement
            if (Input.GetKey(KeyCode.D))
            {
                GetComponent<Rigidbody>().AddForce(new Vector3(speed, 0, 0), ForceMode.Acceleration);
            }
            if (Input.GetKey(KeyCode.S))
            {
                GetComponent<Rigidbody>().AddForce(new Vector3(0, 0, -speed), ForceMode.Acceleration);
            }
            if (Input.GetKey(KeyCode.W))
            {
                GetComponent<Rigidbody>().AddForce(new Vector3(0, 0, speed), ForceMode.Acceleration);
            }
            if (Input.GetKey(KeyCode.A))
            {
                GetComponent<Rigidbody>().AddForce(new Vector3(-speed, 0, 0), ForceMode.Acceleration);
            }
            if (Input.GetKey(KeyCode.Space))
            {
                //Jump
                GetComponent<Rigidbody>().AddForce(new Vector3(0, 1, 0), ForceMode.Impulse);
            }
            if (Input.GetKeyDown(KeyCode.T))
            {
                //Activate/Deactivate travel arrow
                if (travelScript.traveling)
                {
                    travelScript.EndTravel();
                }
                else
                {
                    travelScript.StartTravel();
                }
            }
        }
    }
}