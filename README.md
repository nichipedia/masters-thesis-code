# Machine Learning Model Selection for Predicting Global Bathymetry
___
This code base is for the publication "Machine Learning Model Selection for Predicting Global Bathymetry.
This code was written and work for this publication was performed while I was at the University of New Orleans for my Masters of Computer Science.
In general, this code can be used to repeat the experiments performed in my publication.
I want to note that the data will need to be reaggregated and turned into a gridded file.
Any grid format will work, I used files that a colleague of mine gridded into a proprietary formart. 
The studies that the training data originated from are listed in the paper.
Also, any source of baythymetry will also work. Ideally, the resolution needs to match the resolution of the training data.

## Important Components
___
The following section outlines the important components in the project. You can consider these components to be the peices of the ML Pipeline.

### Genetic Algorithm
___
The GeneticAlgorithms class is my implementation of a GA. It is fairly straight forward. The key components are a Individual class and GeneticPopulation class.
A GeneticPopulation is a collection of Individuals and has method hooks for the algorithm. There is a while loop in that class where the termination criteria can be changed.
I used this class to generate a list of "optimum" features for my pipeline.
I found that the size of this problem made the GA run very slow. Ideally, redoing this to be concurrent will be a great improvement.

### Bins File
___

