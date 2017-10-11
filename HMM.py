import numpy as np
from numpy import linalg
import sklearn.metrics
import matplotlib.pyplot as plt

def read_pos_regions(filename,string_length,chrnum):
    f = open(filename,"r")
    state_list = [0 for i in range(string_length)]
    for line in f:
        cols = line.split()
        if cols[1]=="chr"+chrnum:
            for i in range(int(cols[2]),int(cols[3])+1):
                state_list[i] = 1
    return state_list

def read_sequence(filename): 
    f = open(filename,"r")
    s = ""
    for line in f:
        if line[0]!='>':
            for c in line.rstrip("\n"):
                s+=c
    return s  
     
def nuc_to_int(nuc):
    if nuc=='a' or nuc=='A':
        return 0
    elif nuc=='c' or nuc=='C':
        return 1
    elif nuc=='g' or nuc=='G':
        return 2
    elif nuc=='t' or nuc=='T':
        return 3
    elif nuc=='n' or nuc=='N':
        return 4
    else:
        return 5

class HMM:
    def __init__(self):
        self.transition_matrix = None
        self.emission_prob = None
        self.initial_state = None

    def train_HMM(self,seq,state_list):
        #row number reps current state and column reps next state
        #rows are A-,C-,G-,T-,A+,C+,G+,T+
        #where - is normal regions and + is CpG regions
        self.transition_matrix = np.eye(8)
        self.emission_prob = np.vstack((np.eye(4),np.eye(4)))
        self.initial_state = np.array([0.0 for i in range(8)])
        for i,c in enumerate(seq[:-1]):
            if c == 'n' or c == 'N':
                if seq[i+1] == 'n' or seq[i+1] == 'N':
                    pass
                else:
                    next_state = nuc_to_int(seq[i+1]) + (0 if state_list[i+1]==0 else 4)
                    positive_model = (0 if state_list[i]==0 else 4)
                    for j in range(0+positive_model,4+positive_model):
                        self.transition_matrix[j][next_state]+=0.25
            else:
                current_state = nuc_to_int(c) + (0 if state_list[i]==0 else 4)
                if seq[i+1] == 'n' or seq[i+1] == 'N':
                    state = (0 if state_list[i+1]==0 else 4)
                    self.transition_matrix[current_state][0+state:4+state] += [0.25,0.25,0.25,0.25]
                else:
                    next_state = nuc_to_int(seq[i+1]) + (0 if state_list[i+1]==0 else 4)
                    self.transition_matrix[current_state][next_state] += 1
        print("pre-normalize: ",self.transition_matrix)
        #self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1).reshape(8,1)
        self.transition_matrix = (self.transition_matrix + 10*np.ones((8,8)) )/ (self.transition_matrix.sum(axis=1).reshape(8,1) + 80*np.ones((8,8)) )
        steady_state = linalg.matrix_power(self.transition_matrix,10000)
        print("steadystate: ",steady_state)
        self.initial_state = steady_state[0]
        return self.transition_matrix,self.emission_prob,self.initial_state
    
    def save_params(self,fname):
        f = open(fname,"w")
        num_states = self.transition_matrix.shape[0]
        f.write("{}\n".format(num_states))
        for i in range(self.transition_matrix.shape[0]):
            for j in range(self.transition_matrix.shape[0]):
                f.write("{} ".format(self.transition_matrix[i][j]))
            f.write("\n")
        num_emissions = self.emission_prob.shape[1]
        f.write("{}\n".format(num_emissions))
        for i in range(num_states):
            for j in range(num_emissions):
                f.write("{} ".format(self.emission_prob[i][j]))
            f.write("\n")                
        for i in range(num_states):
            f.write("{} ".format(self.initial_state[i]))
        f.write("\n")

    def load_params(self,fname):
        f = open(fname)
        num_states = int(f.readline())
        self.transition_matrix = np.eye(num_states)
        for i in range(num_states):
            line = f.readline()
            for j,prob in enumerate(line.split()):
                self.transition_matrix[i][j]=float(prob)
        num_emits = int(f.readline())
        self.emission_prob = np.ones((num_states,num_emits))
        for i in range(num_states):
            line = f.readline()
            for j,prob in enumerate(line.split()):
                self.emission_prob[i][j]=float(prob)
        self.initial_state = np.array([0.0 for i in range(num_states)])
        line = f.readline()
        for i,prob in enumerate(line.split()):
            self.initial_state[i] = float(prob)

def viterbi(s,hmm):
    log_transition = np.log(hmm.transition_matrix)
    num_states = hmm.transition_matrix.shape[0]
    table = np.zeros((len(s),num_states))
    table_log = np.zeros((len(s),num_states))
    pointers_back = -5*np.ones((len(s),num_states))
    pointers_back_log = -5*np.ones((len(s),num_states))
    for i in range(num_states):
        table[0][i] = hmm.initial_state[i]*(1 if s[0]=='n' or s[0]=='N' else hmm.emission_prob[i][nuc_to_int(s[0])])
        table_log[0][i] = np.log(hmm.initial_state[i]) + (0 if s[0]=='n' or s[0]=='N' else np.log(hmm.emission_prob[i][nuc_to_int(s[0])]) )
    for i in range(1,len(s)):
        for j in range(num_states):
            if s[i] == 'n' or s[i] == 'N':
                emit_prob = 1
            else:            
                emit_prob = hmm.emission_prob[j][nuc_to_int(s[i])]
            if np.isclose(emit_prob,0):
                table[i][j] = 0
                table_log[i,j] = -np.inf
                #pointers_back[i][j] = np.argmax( np.multiply(table[i-1], hmm.transition_matrix[:,j].flatten()) )
                #pointers_back_log[i,j] = np.argmax( table_log[i-1] + log_transition[:,j].flatten() )
            else:
                table[i][j] = emit_prob * np.max( np.multiply(table[i-1],hmm.transition_matrix[:,j].flatten()) )
                table_log[i,j] = np.log(emit_prob) + np.max( table_log[i-1] + log_transition[:,j].flatten() )
                pointers_back[i][j] = np.argmax( np.multiply(table[i-1], hmm.transition_matrix[:,j].flatten()) )
                pointers_back_log[i,j] = np.argmax( table_log[i-1] + log_transition[:,j].flatten() )
    print(pointers_back[0:5,:])
    print(pointers_back.shape)
    print(table_log[54400:55400,:])
    hidden_seq = [-1 for i in range(len(s))]
    hidden_seq[len(s)-1] = np.argmax(table[len(s)-1])
    prev_state = int(pointers_back_log[len(s)-1,hidden_seq[len(s)-1]])
    #print("prevstate",prev_state)
    for i in range(len(s)-2,0,-1):
        hidden_seq[i] = prev_state
        prev_state = int(pointers_back_log[i,prev_state])
        #print("prevstate",prev_state)
    hidden_seq[0] = prev_state
    hidden_seq_filtered = [0 if hidden_seq[i]<=3 else 1 for i in range(len(hidden_seq))]
    return hidden_seq,hidden_seq_filtered

def plot_regions(predicted,actual):
    predicted = [i if predicted[i]==1 else None for i in range(len(predicted))]
    actual = [i if actual[i]==1 else None for i in range(len(actual))]
    plt.plot(predicted,[[1]]*len(predicted))
    plt.plot(actual,[[0]]*len(actual))
    plt.show()

def main():
    
    s = read_sequence("chr21.fa")
    print(len(s))
    state_list = read_pos_regions("cpgIslandExt.txt",len(s),"21")
    print(len(state_list))
    
    hmm = HMM()
    transition_matrix,emission_prob,initial_state = hmm.train_HMM(s,state_list)
    hmm.save_params("HMM_params_21_pseudocount.txt")
    #hmm.load_params("HMM_params_21.txt")
    print(hmm.transition_matrix)
    print(hmm.emission_prob)
    print(hmm.initial_state)
    s22 = read_sequence("chr22.fa")
    print(len(s22))
    sry = s22[38000000:39000000]
    f = open("sry_seq.txt","w")
    f.write(sry)
    f.close()
    state_list_22 = read_pos_regions("cpgIslandExt.txt",len(s22),"22")
    print(len(state_list_22))
    state_list_sry = state_list_22[38000000:39000000]
    predicted_cpg_raw, predicted_cpg= viterbi(sry,hmm)
    f = open("predicted_cpg_sry.txt","w")
    f.write(str(predicted_cpg))
    f.close()
    f2 = open("actual_cpg_sry.txt","w")
    f2.write(str(state_list_sry))
    f2.close()
    print(sklearn.metrics.confusion_matrix(state_list_sry,predicted_cpg))
    plot_regions(predicted_cpg,state_list_sry)

main()
