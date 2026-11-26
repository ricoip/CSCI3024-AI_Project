# Independent checks of the coding agent
## Check 1: Data

**Check:** Random rows in `submission/manifest.csv`

7 random rows in the submission folder's crop manifest was checked for errors.  
Images were imported into an image editor (Affinity Photo), and a white 
rectangle was created at the pixel locations from the manifest.

This is *in addition* to the 200 checks done at `audit/audit.csv`.

**Outcome:**
```
s015540 | X = 192 | Y = 006 | Size = 170 | OK
s018360 | FAIL: Artwork of a cat on canvas, does not meet requirement
s022605 | FAIL: The whole body was included, not just the face
s037484 | X = 107 | Y = 114 | Size = 145 | Partial fail: Face not upright
s045067 | FAIL: No face in photo
s053270 | X = 191 | Y = 018 | Size = 130 | OK
s061656 | X = 178 | Y = 0 | Size = 249 | OK
```

This doesn't look all that good, as my manual checks (by scrolling through the 
manifest) is finding more errors than the audit script…

## Check 2: Generation

**Context:** The agent provided me with a demo Python script to allow the 
presentation of a live demo.

**Check:** The `demo.py` script 

The demo script was run multiple times, with the same seed and then with 
different seeds. See if the image SHA-256 is the same.

**Outcome OK:** Ran 3x with seed 42 and 2x with 1204. When ran repeatedly, both 
seeds produce the same hash associated with the seed.


## Check 3: AGENTS.md conformation

**Context:** I manually set up a development environment for my agents to work 
on, and wrote part of the AGENTS.md manually to let them know.

**Check:** Files in the project directories, Git commits, Conda environment

**Outcome:**

- Files/Conda — **FAIL**: The agent found a dependency conflict within my Conda 
environment, but it decided to download the dependencies into the project 
directory instead of modifying the Conda environment. 
- Git commits — **PASS**: Everything looks great. No mistaken attributions.