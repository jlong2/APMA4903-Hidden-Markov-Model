import numpy as np

#row number reps current state and column reps next state
#rows are A-,C-,G-,T-,A+,C+,G+,T+
#where - is normal regions and + is CpG regions
transition_matrix = np.eye(8)
emission_prob = np.vstack((np.eye(4),np.eye(4)))
initial_state = np.array([0.0 for i in range(8)])

#pos_region_starts = []
#neg_region_starts = []
def read_pos_regions(filename):
    f = open(filename,"r")
    for line in f:
        cols = line.split()
        if cols[1]=="chr21":
            #pos_region_starts.append( int(cols[2]) )
            #neg_region_starts.append( int(cols[3])+1 )
            

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

def train_HMM(seq):
    """pos_regions = pos_region_starts.copy()
    neg_regions = neg_region_starts.copy()
    current_region = -1
    next_index = -1
    if pos_regions[0] != 0:
        current_region = 0
    else:
        pos_regions.pop()
        current_region = 1
    if pos_regions[0] < neg_regions[0]:
        next_region = 0
        next_index = pos_regions[0]
    else:
        next_region = 1
        next_index = neg_regions[0]
    """
    for i,c in enumerate(seq[:-1]):
        pass

def main():
    read_pos_regions("cpgIslandExt.txt")
    s = read_sequence("chr21.fa")
    #print(s)    
    print(len(s))    
    #train_HMM(s)
    #print(transition_matrix)
    #print(emission_prob)
    #print(initial_state)

main()
