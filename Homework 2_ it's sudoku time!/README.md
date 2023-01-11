# Overview
Goal: build a CV algorithm to solve Sudoku.

* Part 1: Find tables 

Step 1: find some keypoints (Otsu thresholding, edges, Hough lines, corners of 9x9 table, etc.) \
Step 2: find 9x9 tables, estimate the Sudoku-ness of every table (Hough lines, regular structures, etc.) \
Step 3: apply Projective Transform for every found table

* Part 2: Recognize digits

Step 4: divide the table into separate cells (optionally: remove table artifacts) \
Step 5: build digit classifier on MNIST or manually (semi-supervised) annotated train data: feature extractor (e.g. HoG) + classifier (SVM, Random Forest, NN, etc.) \
Legacy approach. It is less robust, but still working pretty fine. (It is not recommended to use template matching approach) \
Step N: manually create templates using normalized images \
Step N+1: apply match_template Links to an external site.to estimate correlation with your templates. \
Step N+2: make a decision for each cell.

* Part 3: [extra] Solve sudoku -> Draw solution

Step 6: We will provide you with a function to solve [sudoku](https://github.com/maxme1/sudoku). You need to aggregate input in the right format. \
Step 7: Plot solved sudoku on the original image. This step is optional and will result in bonus points.
