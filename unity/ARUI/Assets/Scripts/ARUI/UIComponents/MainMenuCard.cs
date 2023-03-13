using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class MainMenuCard : MonoBehaviour
{
    private string[,] tasks =
    {
        {"0", "Place tortilla on cutting board."},
        {"0", "Use a butter knife to scoop about a tablespoon of nut butter from the jar."},
        {"1", "Spread nut butter onto tortilla, leaving 1/2-inch uncovered at the edges."},
        {"0", "Clean the knife by wiping with a paper towel."},
        {"0", "Use the knife to scoop about a tablespoon of jelly from the jar."},
        {"1", "Spread jelly over the nut butter."}, //4
        {"1", "Clean the knife by wiping with a paper towel."},
        {"0", "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick. Roll it tight enough to prevent gaps, but not so tight that the filling leaks."},
        {"0", "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart."},
        {"0", "Trim the ends of the tortilla roll with the butter knife, leaving 1?2 inch margin between the last toothpick and the end of the roll. Discard ends."},
        {"0", "Slide floss under the tortilla, perpendicular to the length of the roll.Place the floss halfway between two toothpicks."},
        {"0", "Cross the two ends of the floss over the top of the tortilla roll." },
        {"1", "Holding one end of the floss in\r\neach hand, pull the floss ends in opposite directions to slice."},
        {"0", "Continue slicing with floss to create 5 pinwheels."},//12
    };

    private Shapes.Line dwellLine;
    private float max = 0.06f;
    private float dwellTime = 2;
    public bool UserIsLooking;
    private Shapes.Rectangle backgroundRect;

    // Start is called before the first frame update
    void Start()
    {
        dwellLine = GetComponentInChildren<Shapes.Line>();
        dwellLine.enabled = false;

        backgroundRect = GetComponent<Shapes.Rectangle>();
        backgroundRect.enabled = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (FollowEyeTarget.Instance.currentHit == EyeTarget.recipe &&
            FollowEyeTarget.Instance.currentHitObj != null &&
            FollowEyeTarget.Instance.currentHitObj.gameObject.name.Equals(gameObject.name))
        {
            if (dwellLine.enabled== false)
                StartCoroutine(Dwelling());

            UserIsLooking = true;
            backgroundRect.enabled = true;
        }
        else
        {
            UserIsLooking = false;
            backgroundRect.enabled = false;

            StopCoroutine(Dwelling());

            dwellLine.enabled = false;
            dwellLine.End = new Vector3(max, dwellLine.End.y, dwellLine.End.z);
        }
    }


    private IEnumerator Dwelling()
    {
        dwellLine.enabled = true;

        Vector3 xStart = new Vector3(0, dwellLine.End.y, dwellLine.End.z);
        Vector3 xEnd = new Vector3(max, dwellLine.End.y, dwellLine.End.z);
        dwellLine.End = xStart;

        float timeElapsed = 0.00001f;
        float lerpDuration = dwellTime;
        while (timeElapsed < lerpDuration && UserIsLooking)
        {
            dwellLine.End = Vector3.Lerp(xStart, xEnd, (timeElapsed / lerpDuration));
            timeElapsed += Time.deltaTime;
            yield return new WaitForEndOfFrame();
        }

        dwellLine.enabled = false;
        dwellLine.End = new Vector3(max, dwellLine.End.y, dwellLine.End.z);

        if (UserIsLooking)
        {
            yield return new WaitForEndOfFrame();
            AngelARUI.Instance.SetTasks(tasks);
            AngelARUI.Instance.SetCurrentTaskID(0);
        }
    }
}
