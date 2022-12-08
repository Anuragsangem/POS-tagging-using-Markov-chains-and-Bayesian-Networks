# ansangem-megavadh-nbheemr-a3










# PART 2:

# Image Recognition using Simple Bayes and HMM using Viterbi:

## Abstraction
### - Initial Probabilities: 
      #### The probabilities of each letter in the entire training data is given and stored in a list.
### - Emission Probabilities: The probabilities of each letter in the given test data as pixels is likely to be that letter in from the given train data pixels.
### - Transitional Probabilities: The probabilities of the next state being a particular letter after the given letter in the test data.These values are stored in a dictionary with the transition states as key and their probabilities as the values.
### - comb_list: The combinations of all possible next states required for transition is calculated using the n_combinations() function and is returned to comb_list.
### - pairs_count: The value(the count of those transitions) of all transitions from one state to the next state is calculated and the number and the transition state is stored as value,key pair in a dictionary and returned.
### - letter_counts: The dictionary stores the letters and the number of times it appeared in the training file as key,value pair respectively

### Simple Bayes: bayes_net() function returns the probabilities of the letter in the given image based on the emission probabilities 

### HMM_Viterbi: hmm_viterbi() function returns the probable sequence of the letter in the given image based on the transitional, emission and initial probabilities

