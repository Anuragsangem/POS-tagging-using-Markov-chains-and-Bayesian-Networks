###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids:[ansangem ,megavadh,nbheemr]
#
# (Based on skeleton code by D. Crandall)
#

#REPORT : 
"""
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



"""






import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.pair_pos_list_dict_emmision={}
        self.counts_pos={}
        self.possible_pos=[]
        self.pair_pos_list=[]
        self.counts_pos_func={}
        self.count_pos=[]
        self.pair_pos_list_dict={}
        self.inital_prob_dict={}
        self.n_combinations_list = []
        self.transition_probabilities={}
        self.transition_matrix_for_train_set={}
        self.train_pos_data=[]
        self.n_combinations_list=[]
        self.output_simple=[]
        
        

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            result=math.log1p(1) #using the log1p function in math library , to avoid exception of calculating log of 0 or less than 0
            for word_index in range(len(sentence)): #calculating the emmsion prob * initial_prob value as the posterior in Simple Model
                result+= self.pair_pos_list_dict_emmision.get((sentence[word_index], label[word_index]),0.00000000001)*self.inital_prob_dict.get(label[word_index],0.000000000001)
                result =math.log1p(result)
            return result

        elif model == "HMM":

            result_hmm=math.log1p(1)
            for word_index in range(1,len(sentence)):
                result_hmm+=math.log1p(self.pair_pos_list_dict_emmision.get((sentence[word_index], label[word_index]),0.00000000001)*self.inital_prob_dict.get(label[word_index],0.000000000001))
                for pos2 in self.counts_pos.keys():
                    transition_prob = self.transition_matrix_for_train_set.get((pos2,label[word_index]),0.0000000001) #returning a default value of 0.00000000001 if the given pos2,state does not exist in the transition table
                    result_hmm+=math.log1p(transition_prob)
            return result_hmm

        elif model == "Complex":
            result_complex=math.log1p(1)
            #for the first word we calculate only the emmision and initial probabilites
            for word_index in range(1,len(sentence)):
                if word_index<3:
                    result_complex+=math.log1p(self.pair_pos_list_dict_emmision.get((sentence[word_index], label[word_index]),0.00000000001)*self.inital_prob_dict.get(label[word_index],0.000000000001))
                    #we extend the transition probability to calculate the probability 3 parts of speect occuring in order
                    #and we add these to the result variable
                else:
                    result_complex += math.log1p(self.transition_matrix_for_train_set.get((sentence[word_index - 1], sentence[word_index]),0.00000000001))
            return result_complex

        else:
            print("Unknown algo!")
            
    def possible_pos_func(self,data):
        for i in range(len(data)):
            temp_len=len(data[i][1])
            for j in range(temp_len):
                self.possible_pos.append(data[i][1][j])
            temp_len=0
        return self.possible_pos
    
    
    def count_pos_func(self,possible_pos):  #gets the count of all the possible_pos
        for possib_pos in self.possible_pos:
            self.counts_pos[possib_pos] = self.counts_pos.get(possib_pos,0)+1
        return self.counts_pos
    
        
    def pair_pos_list_func(self,data): #used to get the data as pairs of (word,pos) from the train dataset and get their counts as well
        for i in range(len(data)):
            temp_store=len(data[i][0])
            for j in range(temp_store):
                a=(data[i][0][j],data[i][1][j])
                #print(a)
                self.pair_pos_list.append(a)
            a=()
            
        for ppl in self.pair_pos_list:
            self.pair_pos_list_dict[ppl] = self.pair_pos_list_dict.get(ppl,0)+1
            
        return self.pair_pos_list_dict
    
    
    def n_combinations(self,input_list, n):
        for j in input_list:
            for i in range(len(j)):
                self.n_combinations_list.append(tuple(j[i: i + n]))
        return self.n_combinations_list

    def pairs_counts(self,input_pos):
        pair_count={}
        for i in input_pos:
            if i in pair_count:
                pair_count[i] += 1
            else:
                pair_count[i] = 1
        return pair_count


    def transition_probabilty(self,input_pos):
        pairs_list = self.n_combinations(input_pos, 2) #creating pairs of all possible combinations of pos
        pair_count=self.pairs_counts(pairs_list) #counts the occurences of pairs of pos

        for pair in pairs_list:
            self.transition_probabilities[pair] = pair_count[pair] / self.counts_pos[pair[0]]
        return self.transition_probabilities    #pass the entire train dataset here and just save it to some variable
    
    


    
    def initial_prob(self,pos):
        for i in self.counts_pos.keys():
            self.inital_prob_dict[i]=self.counts_pos[i]/955797 #dividing the count of pos in train set by total length of the dataset
            
            
        return inital_prob_dict.get(pos,0.00000000001)

        
        
    def emission_prob(self,w1,pos):
        ip_pair = (w1,pos)
                                    #pre compute emmision, transition in training function
        emm_prob=self.pair_pos_list_dict.get(ip_pair,0.00000000001)/self.counts_pos.get(ip_pair[1],0.00000000001)
        return emm_prob
    
    
        

    # Do the training!
    #
    def train(self, data):
        #we calculate the initial , emmision , transition probability on the entire train set here and use it for algorithms
        
        possible_pos=self.possible_pos_func(data) #precomputing the total possible pos
        counts_pos=self.count_pos_func(self.possible_pos) #precomputing counts of each pos to further use it
        pair_pos_list_dict=self.pair_pos_list_func(data) #precomputing the pairs of (word,pos) as a dict keys and their counts in the dataset as their values
        
        for i in self.counts_pos.keys():
            self.inital_prob_dict[i]=self.counts_pos[i]/955797 #precomputing the initial prob of all the pos in train set
        

        for j in self.pair_pos_list_dict.keys():   #Precomputing all the emmision probabilities using the pair_pos_list tosave computing time
            self.pair_pos_list_dict_emmision[j]=self.pair_pos_list_dict.get(j,0.00000000001)/self.counts_pos.get(j[1],0.00000000001)
                
        for k in range(len(data)):
            self.train_pos_data.append(data[k][1])
        self.transition_matrix_for_train_set=self.transition_probabilty(self.train_pos_data)
        
        #return self.pair_pos_list_dict_emmision
        
        


    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, input_sentence):
        temp_storage_dict={}
        max_pos=''
        output = []

        for i in range(len(input_sentence)):
            if input_sentence[i] in "',?/!@#$%^&*()'":
                output.append('.')
            elif (input_sentence[i].isdigit()):
                output.append('num')
                
            else:
                prob=0
                for pos in self.counts_pos.keys():
                    prob_naive=self.pair_pos_list_dict_emmision.get((input_sentence[i].strip(),pos),0.0000000001)*self.inital_prob_dict.get(pos,0.000000000001)
                    if prob_naive>prob:
                        prob=prob_naive
                        max_pos=pos
                #print(prob)
                if max_pos==0:
                    output.append('x')
                output.append(max_pos) 
        return output


#I referred the Viterbi code from this links as I'm not familiar with dynamic programming concepts: 
# https://en.wikipedia.org/wiki/Viterbi_algorithm
# https://github.com/chetan253/B551-Elements-of-AI/blob/f66851e1684ffb3a5a477e3fa8e310426b2d13b8/Part-of-Speech-Tagging/pos_solver.py#L134

    def hmm_viterbi(self, sentence):
        seq = [] #stores the sequence which is the output
        v_find_list = [] #list to store the viterbi calculation
        viterbi_probs = {} #probs of viterbi
        probs_dict_viterbi = {}
        emission_prob_temp=0 #temp value to store the probability
        unique_pos=list(self.counts_pos.keys())
        for counter, word in enumerate(sentence):
            v_find_list.append({}) #append an initial empty dict , for the first word of the sequence
            for state in unique_pos:
                emission_prob_temp = self.pair_pos_list_dict_emmision.get((word,state),0.000000001)
                if counter == 0: #for the first word there is no transition probability hence calculating emmision*initial prob for the first word
                    v_find_list[counter][state] = {'prob' : emission_prob_temp * self.inital_prob_dict.get(state,0.00000000001), 'before': None} #appending a None value before the first word
                else:
                    tempDict = {}
                    for pos2 in unique_pos:
                        prev_state_prob = v_find_list[counter - 1][pos2]['prob']  #calculating the transition*emmision value at the previous word
                        transition_prob = self.transition_matrix_for_train_set.get((pos2,state),0.0000000001) #returning a default value of 0.00000000001 if the given pos2,state does not exist in the transition table
                        tempDict[pos2] = prev_state_prob * transition_prob
                    max_val_state= max(zip(tempDict.values(), tempDict.keys())) #saving the max of temp dict value, temp dict keys
                    max_value, max_state = max_val_state[0], max_val_state[1] 
                    v_find_list[counter][state] = {'prob' : emission_prob_temp * max_value, 'before' : max_state} #storing the max value pos in the v_find list and storing the max_state as the before node
                    
                    
        max_prob = max(value['prob'] for value in v_find_list[-1].values()) # getting the max value fo the last word
        previous = None
        for state, data in v_find_list[-1].items(): 
            if data["prob"] == max_prob:  #if the prob word has the max_prob then appending it to the sequence
                    seq.append(state)
                    previous = state
                    break

        for counter in range(len(v_find_list) - 2,-1,-1):   #backtracking the best sequence values in this loop 
            seq.insert(0, v_find_list[counter + 1][previous]["before"])
            previous = v_find_list[counter + 1][previous]["before"]

        return seq


    def complex_mcmc(self, sentence):
        #taking an initial sample of the prediction of pos as all nouns and then fixing initial value and changing the next pos values and saving thr probability

        words = list(sentence)
        sample = [ "noun" ] * len(sentence) #creating initial sample as all nouns as noun has highesty init prob
        number_of_iterations =20  #number of iterations
        output=[]
        burn_iterations = 5 #we burn first 5 iterations and count the iterations from later
        samples=[] #we add samples list here
        sample2=sample.copy() #creating a sample copy
        #print('new sample:',new_sample

        for i in range(len(sentence)):
            prob=0
            prob2=0
            for pos in self.counts_pos.keys():
                prob_complex=self.pair_pos_list_dict_emmision.get((sentence[i],pos),0.00001)*self.inital_prob_dict.get(pos,0.0001)

                for pos2 in self.counts_pos.keys():
                    transition_prob = self.transition_matrix_for_train_set.get((pos2,pos),0.000000001) #returning a default value of 0.00000000001 if the given pos2,state does not exist in the transition table
                    prob_2 = prob_complex * transition_prob

                if prob_complex>prob:
                    prob=prob_complex
                    max_pos=pos
            output.append(max_pos) 
        return output
 


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")
