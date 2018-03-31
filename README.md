this is a [penrose triangle](https://en.wikipedia.org/wiki/Penrose_triangle):

```
       _
      / /\
     / /  \
    / / /\ \
   / / /\ \ \
  / / /  \ \ \
 / /_/____\ \ \
/__________\ \ \
\_____________\/
```

i wrote a program to draw shapes like this (as vectors, not ascii)


## why?
this is always the first thing i doodle, and i as a programmer have no choice but to think about how to create it procedurally. after figuring that out, i kept generalizing:

- triangle
- more than 2 "segments" of variable width
- any regular polygon
- any convex polygon
- any simple polygon (as in non-self-intersecting)

other things to make it do:

- curved sides
- arbitrary "graphs" instead of simple cycles (like an A instead of a triangle - use this to draw text)
