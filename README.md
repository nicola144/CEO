# CEO
Repository for "Causal Entropy Optimization" , AISTATS 2023 


Contact nbranchini17@gmail.com for questions. 

Running the code requires installing Emukit (latest version should work), graphviz and pygraphviz. 
I recommend installing the above in that order, based on experience on Mac OS - Graphviz can be a pain.
Remaining packages are standard and any recent version should work. 

There is a lot of functionality inherited from DCBO (https://github.com/neildhir/DCBO), which the code was built upon, that is not used.

The directory `paper_results/` contains data for the results for the experiments in the paper, including random seeds corresponding to the initial interventional points used.

Note: the practical implementation of the acquisition function, CES, is inefficient and slow. It could at least be parallelised over acquisition points. 

Finally, this is research code. We are not professional software developers and are thus not equipped to make a (more) robust implementation.
Email the first author for questions on the code. 
