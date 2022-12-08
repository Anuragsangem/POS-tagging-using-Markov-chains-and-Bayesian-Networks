#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def initial_probability(fname):

    # #'/Users/meghanaavadhanam/Downloads/mytext.txt'

    #initial probability function calculates the prior probabilities of all the letters that are present in the
    # training data 
    # i.e if the training data is "hello", the probabilites of each of the letters h,e,l,o will be calculated
    # separately and appended to a list named initial_probabilities
    text_obj = open(fname, "r")
    text = text_obj.readlines()
    text = text[:int(len(text)/16)]
    text_obj.close()

    word_list = []
    letter_list = []

    for line in text:
        for word in line.split():
            word_list.append(word)
        for letter in line:
            letter_list.append(letter)
            
    initial_counts = np.zeros(len(TRAIN_LETTERS))
    initial_counts = initial_counts.tolist()

    # this loop checks whether or not each letter from the training letters is 
    # present in the training data list. 

    for i in range(len(TRAIN_LETTERS)):
        for l in letter_list:
            if TRAIN_LETTERS[i] == l:
                initial_counts[i] +=1

    initial_probabilities = []
    total_count = len(letter_list)

    for count in initial_counts:
        prob = count/total_count
        initial_probabilities.append(prob)
    
    return initial_probabilities

def emission_probability(x,y):
    # let noise percentage m = 30%
    # since each image has a different value of noise, and we donot know its value, 
    # we assume that approximately on average, each image has a noise percentage of 30%
    # although it can be more or less than that value
    # this function calculates the emission probabilities of all the train and test test_letters
    # this means that the pixels of each of the letters in the test data are compared to those in train letters
    noise = 0.3
    arr = []
    train_pixels = x
    test_pixels = y
    emmision_arr = []
   
    for pixel_1 in train_pixels:
        for pixel_2 in range(len(test_pixels)):

            pix2 = test_pixels[pixel_2]
            pix1 = train_pixels.get(pixel_1)

            yes_count = 0
            no_count = 0
            
            for i in range(CHARACTER_HEIGHT):
                for j in range(CHARACTER_WIDTH):
                    if pix1[i][j] == pix2[i][j]:
                        yes_count += 1 
                    else:
                        no_count += 1
                        
            em = ((1-noise) ** yes_count) * (noise ** no_count)
            arr.append(em)

    emmision_arr = np.array(arr).reshape(len(x),len(y))
    return emmision_arr
   

def bayes_net(emission):
    # the bayes net function finds the maximum of the emission probabilties for each of the test letters
    rows = emission.shape[0]
    cols = emission.shape[1]
    y = np.zeros(cols)
    bayes_prob = np.zeros(shape=(rows,cols))
    
    for i in range(0,cols):
        for j in range(0,rows):
            bayes_prob[j,i] = emission[j,i]
            
    y = np.argmax(bayes_prob, axis=0)
    bayes_prob_FINAL = np.argmax(bayes_prob, axis=0)
    
    return bayes_prob_FINAL

     # citation given below, i referred to an online source for this function


def n_combinations(letter_list, n):
    n_combinations_list=[]
    buff_list=[]  
    for i in range(len(letter_list)-1):
        if letter_list[i] not in TRAIN_LETTERS:
            continue

        buff_list.append(letter_list[i])
        buff_list.append(letter_list[i+1]) 
        n_combinations_list.append(tuple(buff_list))
        buff_list=[]   
    return n_combinations_list

def pairs_counts(comb_list):
    pair_count={}
    for i in comb_list:
        if i in pair_count:
            pair_count[i] += 1
        else:
            pair_count[i] = 1
    
    return pair_count

def transition_probabilty(comb_list,pairs_count,letter_counts):
    # this function calculates the transition probabilites of the pairs of letters
        transition_probabilities={}
        # print(pairs_count)
        for pair in comb_list:
            transition_probabilities[pair] = pairs_count[pair] / letter_counts[pair[0]]
        
        return transition_probabilities

def transition(x,y):
    # let noise percentage m = 30%
    # since each image has a different value of noise, and we donot know its value, 
    # we assume that approximately on average, each image has a noise percentage of 30%
    # although it can be more or less than that value
    # this function calculates the emission probabilities of all the train and test test_letters
    # this means that the pixels of each of the letters in the test data are compared to those in train letters
    noise = 0.5
    arr = []
    train_pixels = x
    test_pixels = y
    trans = []
   
    for pixel_1 in train_pixels:
        for pixel_2 in range(len(test_pixels)):

            pix2 = test_pixels[pixel_2]
            pix1 = train_pixels.get(pixel_1)

            yes_count = 0
            no_count = 0
            
            for i in range(CHARACTER_HEIGHT):
                for j in range(CHARACTER_WIDTH):
                    if pix1[i][j] == pix2[i][j]:
                        yes_count += 1 
                    else:
                        no_count += 1
                        
            em = ((1-noise) ** yes_count) * (noise ** no_count)
            arr.append(em)

    trans = np.array(arr).reshape(len(x),len(y))
    return trans

def HMM(emission):
    # the bayes net function finds the maximum of the emission probabilties for each of the test letters
    rows = emission.shape[0]
    cols = emission.shape[1]
    y = np.zeros(cols)
    hmm_prob = np.zeros(shape=(rows,cols))
    
    for i in range(0,cols):
        for j in range(0,rows):
            hmm_prob[j,i] = emission[j,i]
            
    y = np.argmax(hmm_prob, axis=0)
    HMM_PROB_FINAL = np.argmax(hmm_prob, axis=0)
    
    return HMM_PROB_FINAL

def hmm_viterbi(test_letters,initial_probabilites_dict,emission_probabilities,TRAIN_LETTERS,transition_probabilities,letter_counts):
        # this viterbi function is adapted from our code to the 1st question of the same assignment
        
        seq = []
        viterbi_lookup = []
        viterbi_probs = {}
        probs_dict_viterbi = {}
        emission_prob_temp=0
        tag_list=list(letter_counts.keys())
        for i in test_letters:
            viterbi_lookup.append({})
            for state in tag_list:
                emission_prob_temp = emission_probabilities.get((letter,state),0.000000001)
                if counter == 0:
                    viterbi_lookup[counter][state] = {'prob' : emission_prob_temp * initial_probabilites_dict.get(state,0.00000000001), 'prev_state': None}
                else:
                    tempDict = {}
                    for state_i in tag_list:
                        prev_state_prob = viterbi_lookup[counter - 1][state_i]['prob'] 
                        transition_prob = transition_probabilities.get((state_i,state),0.0000000001)
                        tempDict[state_i] = prev_state_prob * transition_prob
                    max_val_state= max(zip(tempDict.values(), tempDict.keys())) 
                    max_value, max_state = max_val_state[0], max_val_state[1] 
                    viterbi_lookup[counter][state] = {'prob' : emission_prob_temp * max_value, 'prev_state' : max_state}
                    
                    
        max_prob = max(value['prob'] for value in viterbi_lookup[-1].values()) # get the max value for the last state
        previous = None
        for state, data in viterbi_lookup[-1].items():
            if data["prob"] == max_prob:
                    seq.append(state)
                    previous = state
                    break
                
        for counter in range(len(viterbi_lookup) - 2, -1, -1):
                seq.insert(0, viterbi_lookup[counter + 1][previous]["prev_state"])
                previous = viterbi_lookup[counter + 1][previous]["prev_state"]
        return seq


if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
x=train_letters
test_letters = load_letters(test_img_fname)
y=test_letters
prior_prob = initial_probability(train_txt_fname)
emission_prob = emission_probability(x,y)
simple_bayes = bayes_net(emission_prob)
my_transition = transition(x,y)
hmm_prob = HMM(my_transition)

print ("Simple: "+"".join([TRAIN_LETTERS[i] for i in simple_bayes]))


#print("Simple: " + "Sample s1mple resu1t")
print("   HMM: " +"" .join([TRAIN_LETTERS[i] for i in hmm_prob]))


# CITATIONS 
# https://github.com/ssghule/Optical-Character-Recognition-using-Hidden-Markov-Models/blob/master/ocr_solver.py
# 
# %%
