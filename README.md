# ansangem-megavadh-nbheemr-a3

#PART1



#PART1 : REPORT and ABSATRACTIONS : 


1)Abstractions used : 
        inital_prob_dict={}-> Initial probability of all the pos in the train set  eg : {noun:0.52,det:0.31} etc  
        possible_pos=[] -> This is the list of all possible pos obtained by iterating the train set used to calculate counts_pos
        counts_pos={} -> This dictionaty have counts of POS in the train set eg {noun:10,verb:2} etc
        pair_pos_list=[] ->This is the list of tuples of all the (word and pos) in the train set , we used this to calculate emmision probability 
        counts_pos_func={} -> This is a dictionary of counts of the tuples of pair_pos_list , this is used to calculate emmision probability
        pair_pos_list_dict_emmision={}-> We divide the tuple count of tuples of pair_pos_list with the count of the pos in the tuple to get the emmision probability 
        n_combinations_list = []-> This is the list of n combinations of a list , we used pair combinations of pos to calculate transition probabiity
        transition_matrix_for_train_set={}-> This dictionary has the transition probabilities of all the possible pairs of pos in train set
        output_simple=[]->This is the output of simple naive bayes model
        output_complex: this is the output of complex model


2)Ideology
i)Simple model : In simple model we use naive approach, we considered only the emmision probability of each pos multipled by the initial probability of the pos , 
                    We calculated all the possible words possible for each pos and return the max value per each pos

Disadvantages : This don't consider the intermediate dependencies of the words , but it only considers the max probability of a pos being a word as it has seen in the train set


ii)hmm_viterbi : To consider the intermediate dependencies of the words ,we use tranition probability to consider the probability of a pos transitioning to the next pos
                in all the train set pairs generated

            Workflow : 
            1)n_combinations function: We defined a function n_comninations : which returns the n_combinations of any input list , we obtained pairs inputting the 
                all the possible pos to the function
             2)pairs_counts function : This fucntion is used to get the counts of all the pairs of pos generated using the above function
             3)using the above 2 functions we calculate the transition probability of all the possible pairs of pos

    Using Viterbi algorithm we calculate the trnasition_prob of a pos to its next possible pos * the emmision prob of the  current pos and return the path which returns the max product
    Viterbi uses dynamic programming to efficiently deduce the unnecessary comparisions and computations of the sequences

Note: I referred the Viterbi code from this links as I'm not familiar with dynamic programming concepts: 
 https://en.wikipedia.org/wiki/Viterbi_algorithm
 https://github.com/chetan253/B551-Elements-of-AI/blob/f66851e1684ffb3a5a477e3fa8e310426b2d13b8/Part-of-Speech-Tagging/pos_solver.py#L134


iii)complex_mcmc : We use Gibbs sampler using Montecarlo methods to consider the dependencies of the words in a betetr way than hmm , as 
                    HMM cosiders only the one previos path , we cannot use viterbi algo here , we calculate the transition probabilities of 3 pos occuring in a sequence in the train set , to capture the dependencies in a betetr way than the hmm



                WorkFlow : 
                1)initially we have predicted all the words in an sentence to be 'nouns' (no specific reason but noun has the greatest initial probabilites among all the other pos in the train set)
                2)We fixed the first pos and change the next 2 pos and calculate all the transition probabilites of the possibilites , 
                in this way we were able to encorporate dependecies of 3 pos occuring in a sequence
                3)We choose 50 iterations , and we deduct first 10 iterations as our burning iterations
                4)we calculate the transition probabilities of states fixing one pos and return the max probability sequence

Referral links : 
https://www.youtube.com/watch?v=MNHIbOqH3sk&t=25s&ab_channel=ritvikmath
https://www.youtube.com/watch?v=rFSlqsxCD_g&ab_channel=JoshuaFrench


3)Output : 
This is the output we got on the bc.test file of 2000 sentences
So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       93.95%               47.70%
            2. HMM:       95.32%               56.35%
        3. Complex:       92.52%               40.60%


Conclusion : HMM performed better than the other two algorithms on the test set , but the difference is not that large  ,almost all the algorithms performed equally good on the test set

Future improvisations : 
We can try to calculate much more richer dependencies using gibbs sampling (which may take large computational power and more number of iterations) to get a much more comprehensive performance of our algorithm













# PART 2:

# Image Recognition using Simple Bayes and HMM using Viterbi:

## Abstraction
### - Initial Probabilities: The probabilities of each letter in the entire training data is given and stored in a list.
### - Emission Probabilities: The probabilities of each letter in the given test data as pixels is likely to be that letter in from the given train data pixels.
### - Transitional Probabilities: The probabilities of the next state being a particular letter after the given letter in the test data.These values are stored in a dictionary with the transition states as key and their probabilities as the values.
### - comb_list: The combinations of all possible next states required for transition is calculated using the n_combinations() function and is returned to comb_list.
### - pairs_count: The value(the count of those transitions) of all transitions from one state to the next state is calculated and the number and the transition state is stored as value,key pair in a dictionary and returned.
### - letter_counts: The dictionary stores the letters and the number of times it appeared in the training file as key,value pair respectively

### Simple Bayes: bayes_net() function returns the probabilities of the letter in the given image based on the emission probabilities 

### HMM_Viterbi: hmm_viterbi() function returns the probable sequence of the letter in the given image based on the transitional, emission and initial probabilities

