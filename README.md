CC-RANSAC

1. Randomly select a sample containing a number of 𝑠 data points from 𝑆 and instantiate the model from this subset.
2. Determine the set of data points 𝑆 𝑖𝑖 which is within a distance threshold 𝑡 of the model. The set 𝑆 𝑖𝑖 is the consensus set of the sample and defines the inliers for model 𝑖𝑖.
3. After determining the consensus set 𝑆𝑖𝑖, perform connected component analysis on this set to identify connected regions.
4. Keep only the connected components that exceed a certain size threshold 𝑇𝑇c. This helps filter out small, isolated groups of inliers.
6. Re-estimate the model using the points in the largest connected component.
7. Terminate the algorithm if the size of the largest connected component is greater than or equal to 𝑇𝑇.
8. If the size of 𝑆 𝑖𝑖 (the number of inliers) is greater than some threshold 𝑇𝑇, re-estimate the model using all the points in 𝑆 𝑖𝑖 and terminate.
9. If the size of 𝑆 𝑖𝑖 is less than 𝑇𝑇, select a new subset and repeat from step 1.
10. After 𝑁𝑁 trials the largest consensus set 𝑆 𝑖𝑖 is selected, and the model is re-estimated using all the points in the subset 𝑆 𝑖𝑖.