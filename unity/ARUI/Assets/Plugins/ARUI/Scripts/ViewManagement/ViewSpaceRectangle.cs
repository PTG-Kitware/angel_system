using System;

public class ViewSpaceRectangle
{
    public int StartX;
    public int StartY;
    public int EndX;
    public int EndY;

    public ViewSpaceRectangle(int startX, int startY, int endX, int endY)
    {
        this.StartX = startX;
        this.StartY = startY;
        this.EndX = endX;
        this.EndY = endY;
    }

    /// <summary>
    /// Checks if the given rectangle is enclosed by this rectangle.
    /// </summary>
    /// <param name="outerRect"></param>
    /// <returns>true if this rectangle is enclosed by outerRect</returns>
    public bool IsEnclosedBy(ViewSpaceRectangle outerRect)
    {
        return (this.StartX >= outerRect.StartX &&
                this.EndX <= outerRect.EndX &&
                this.StartY >= outerRect.StartY &&
                this.EndY <= outerRect.EndY);
    }

    /// <summary>
    /// Checks if the this rectangle is adjacent to the given rectangle in the upper Y direction
    /// </summary>
    /// <param name="rect2">the other rect</param>
    /// <param name="skip">how many rows to skip, when checking for adjency</param>
    /// <returns></returns>
    public bool IsAdjacentinY(ViewSpaceRectangle rect2, int skip)
    {
        if (this.StartX >= rect2.EndX || this.EndX <= rect2.StartX)
            return false;
        else if (this.EndY == rect2.StartY || this.EndY == (rect2.EndY+ skip))
            return true;
        else
            return false;
    }

    /// <summary>
    /// This rectangle and the given one have an overlapping area in the upper Y direction, and the overlapping area is returned.
    /// </summary>
    /// <param name="rect2"></param>
    /// <returns></returns>
    public ViewSpaceRectangle ConsensusInY(ViewSpaceRectangle rect2)
    {
        return new ViewSpaceRectangle(
          Math.Max(this.StartX, rect2.StartX),
          Math.Min(this.StartY, rect2.StartY),
          Math.Min(this.EndX, rect2.EndX),
          Math.Max(this.EndY, rect2.EndY)
        );
    }
};