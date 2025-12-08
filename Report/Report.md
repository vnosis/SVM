# 

Section 1. Why SVMs? Explain the limitations of the perceptron that SVM aims to fix.

Section 2. The concept of the margin. Based on section 12.2.1 from the book, explain in
your own words what the margin is and why a scale needs to be set.

Section 3. Hard vs. Soft SVM. What are the differences between hard and soft SVMs

Section 4. The Hinge Loss. Explain the terms in Eq. 12.31, while explicitly discussing the
role of the parameter C. What value does it take for a hard SVM? For a soft?

Section 5. Geometric View. Include a hand-made figure (scanned from a paper drawing or
tablet sketch) illustrating: margins, support vectors, slack variables, correctly classified
and wrongly classified points. Include a brief description of your figure.



In order to maximize our margin we need to minimize ||w|| where k = 1/||w||


### WHY SVMs?


### The Concept of the Margin
When we draw our decision boundary, we want the boundary to be drawn as far away from each class, we call this maximizing our margin, so that the
distance from the first class is equally as far from the boundary as the second class. The closest point(s) that can be defined as part of a class will be the "Support" that define our margin. 

However when we have outliers it becomes problematic because our decision boundary and margin will be redifined by this outlier from one class and the a point closest in the other class. If we can see how the a graph is drawn with an outlier we will see that our margin is drawn incorrectly and that maybe the point that is an outlier should be removed. This is where the concepts of Hard vs Soft margins come into play. 

### Hard vs. Soft SVM. 
Hard SVM's enforce a decision boundary that linearly seperates all the classes from one another. Where everything on one side of the boundary gets classified as plus 1 and everything on the other side gets classified as negative 1. In the real world, linear seperable data is very uncommon. 
When we have a

On the other hand, Soft SVM's are able to draw a decision boundary that seperate two classes while maximizing the margin, even with some noise and outliers. For every outlier we add a penalty, and depending on how far the outlier is from the boundary, we will increase the size of the penalty.
