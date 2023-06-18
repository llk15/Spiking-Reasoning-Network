import torch
import torch.nn as nn

ATTRIBUTE = {
    'shape': ['cube', 'cylinder', 'sphere'],
    'size': ['large', 'small'],
    'material': ['rubber', 'metal'],
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
}
NEURON_TYPE = [
    'scene', # 0 copy and start
    
    'shape', 'size', 'material', 'color', # 1 - 4 attribute types
    'cube', 'cylinder', 'sphere', # 5 - 7 shape
    'large', 'small', # 8 - 9 size
    'rubber', 'metal', # 10 - 11 metarial
    'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow', # 12 - 19 color
    
    # 20 - 34 attributes copies
    'cube_copy', 'cylinder_copy', 'sphere_copy',
    'large_copy', 'small_copy',
    'rubber_copy', 'metal_copy',
    'gray_copy', 'red_copy', 'blue_copy', 'green_copy', 'brown_copy', 'purple_copy', 'cyan_copy', 'yellow_copy',
    
    'count', 'count_copy', # 35 - 36 exist, count and count-compare
    'union', 'intersect', # 37 - 38 retrival
    'same', 'self_inh', # 39 - 40 same gate and self inhibition gate
    
    'front', 'behind', 'right', 'left', # 41 - 44 relations
    'object', # 45 + 7N, N-objects
    'object_copy', # 46 + 7N, N-object copies
    'object_front', # 47 + 7N, N-object front triggers
    'object_behind', # 48 + 7N, N-object behind triggers
    'object_right', # 49 + 7N, N-object right triggers
    'object_left', # 50 + 7N, N-object left triggers
    'object_inh', # 51 + 7N, N-object left triggers
]
NEURON = {} # from TYPE to INDEX
for i in range(len(NEURON_TYPE)):
    NEURON[ NEURON_TYPE[i] ] = i

THRESHOLD = 1.0
COPY = 0.5
SAME = 0.5
RELATION = 0.5
RETRIVAL = 0.3
epsilon = 0.05
    
__all__ = ['SRN', 'spike2answer', 'scene2connection', 'program2stimuli']

class SRN(nn.Module):
    def __init__(self, connection, decay = 0., threshold = 1.):
        super(SRN, self).__init__()
        self.src = nn.Parameter(connection) # sparse recurrent connection
        self.neuron_num = self.src.shape[0]
        self.decay = decay
        self.threshold = threshold
        
        if self.src.shape != torch.Size([self.neuron_num, self.neuron_num]):
            raise RuntimeError('connection shape should be [n,n]')
    
    def forward(self, x): # x.shape = [time_steps, vector_length = neuron_num, 1]
        mem = torch.zeros(x[0].shape, device = x.device) # membrane potential [neuron_num, 1]
        spike = torch.zeros(x[0].shape, device = x.device) # neuron firing [neuron_num, 1]

        for xt in x:
            mem = mem * self.decay * (1. - spike) + xt + \
                torch.mm(self.src, spike) # torch.spmm() is slow, torch.sparse.mm() is slower but trainable
            spike = mem.ge(self.threshold).float()     
            
        return mem, spike

    
    
    
def spike2answer(mem, spike, question): # decode SRN outputs into an answer
    
    answer = 'error'
    
    if question == 'exist':
        if spike[NEURON['count']]:
            answer = 'yes'
        else:
            answer = 'no'

    elif question == 'count':
        answer = str(mem[NEURON['count']].int().item())

    elif question == 'equal_integer':
        if mem[NEURON['count']] == mem[NEURON['count_copy']]:
            answer = 'yes'
        else:
            answer = 'no'

    elif question == 'greater_than': 
        if mem[NEURON['count']] > mem[NEURON['count_copy']]:
            answer = 'yes'
        else:
            answer = 'no'

    elif question == 'less_than': 
        if mem[NEURON['count']] < mem[NEURON['count_copy']]:
            answer = 'yes'
        else:
            answer = 'no'

    elif question.startswith('query'):
        for i in range(NEURON['cube'],NEURON['yellow']+1):
            if spike[i]:
                answer = NEURON_TYPE[i]
                break

    elif question.startswith('equal'):
        key = question.split('_')[1]
        count = 0
        for attr in ATTRIBUTE[key]:
            attr = attr + '_copy'
            count += spike[NEURON[attr]].int().item()
        if count == 1:
            answer = 'yes'
        else:
            answer = 'no'
    
    return answer
    
def program2stimuli(program, neuron_num): # encode a program into stimuli, as input of SRN
    stimuli = [[0.]*neuron_num]
    for token in program:
        # backup current objects and attributes, and restart a scene
        if token == 'scene': 
            stimuli[-1][NEURON['scene']] = THRESHOLD
            stimuli.append([0.]*neuron_num) # wait for propogation
            stimuli.append([0.]*neuron_num) # wait for propogation
            
        # retrive backup objects and combine with current ones    
        elif token == 'union': 
            stimulus = [0.]*neuron_num
            stimulus[NEURON['union']] = THRESHOLD
            stimuli.append(stimulus)
            stimuli.append([0.]*neuron_num) # wait for propogation
        
        # retrive backup objects and intersect with current ones
        elif token == 'intersect': 
            stimulus = [0.]*neuron_num
            stimulus[NEURON['intersect']] = THRESHOLD
            stimuli.append(stimulus)
            stimuli.append([0.]*neuron_num) # wait for propogation
            
        # filter current objects with certain attribute    
        elif token.startswith('filter'): 
            attr = token.split('[')[1][:-1]
            stimulus = [0.]*neuron_num
            stimulus[NEURON[attr]] = 11 * THRESHOLD
            stimuli.append(stimulus)
            stimuli.append([0.]*neuron_num) # wait for propogation
        
        # select objects according to current object and its relation    
        elif token.startswith('relate'):
            relation = token.split('[')[1][:-1]
            stimulus = [0.]*neuron_num
            stimulus[NEURON[relation]] = THRESHOLD
            stimuli.append(stimulus)
            stimuli.append([0.]*neuron_num) # wait for propogation
            stimuli.append([0.]*neuron_num) # wait for propogation
        
        # select objects according to current object and its attributes
        elif token.startswith('same'): 
            key = token.split('_')[1]
            stimulus = [0.]*neuron_num
            stimulus[NEURON[key]] = -THRESHOLD # extract corresponding attribute
            stimulus[NEURON['self_inh']] = THRESHOLD # trigger a delayed self inhibition
            stimuli.append(stimulus)
            
            stimulus = [0.]*neuron_num
            stimulus[NEURON[key]] = THRESHOLD # restore the inhibition attribute state
            stimulus[NEURON['same']] = THRESHOLD # trigger objects with same attribute
            stimuli.append(stimulus)
            stimuli.append([0.]*neuron_num) # wait for propogation
            
        # trigger attributes waiting for query
        elif token.startswith('query'): 
            key = token.split('_')[1]
            stimulus = [0.]*neuron_num
            stimulus[NEURON[key]] = -THRESHOLD # extract corresponding attribute
            stimuli.append(stimulus)
            
            stimulus = [0.]*neuron_num
            stimulus[NEURON[key]] = THRESHOLD # restore the inhibition attribute state
            stimuli.append(stimulus)
            
        elif token.startswith('unique'): 
            stimuli.append([0.]*neuron_num) # wait for propogation
            
        # combine backup attributes with current ones
        elif token.startswith('equal'):
            stimuli[-1][NEURON['scene']] = THRESHOLD
            stimuli.append([0.]*neuron_num) # wait for propogation
        
        elif token == 'exist' or token == 'count':
            stimuli.append([0.]*neuron_num) # wait for propogation
            
    return torch.tensor(stimuli)


def scene2connection(scene): # incode a scene into connect weight of SRN
# connection[][]
    object_num = len(scene)
    neuron_num = len(NEURON)-7 + 7*object_num
    connection = torch.zeros(neuron_num, neuron_num)
        
# attributes connections
    # common connections
    for key in ATTRIBUTE:
        connection[NEURON['scene']][NEURON[key]] = THRESHOLD # scene triggers
        connection[NEURON[key]][NEURON[key]] = THRESHOLD # type_neuron self-connections
        
        for attr in ATTRIBUTE[key]:
            connection[NEURON[key]][NEURON[attr]] = -10 * THRESHOLD # inhibitions from type_neurons
            connection[NEURON['scene']][NEURON[attr]+15] = THRESHOLD-COPY # copy gates
            connection[NEURON[attr]][NEURON[attr]+15] = COPY # copy channels
            connection[NEURON[attr]+15][NEURON[attr]+15] = THRESHOLD # copy self-connections
            
            for n in range(object_num):
                connection[NEURON[attr]][NEURON['object']+ 7*n] = -SAME # object inhibitions

# objects connections
    # common connections
    for n in range(object_num):
        connection[NEURON['scene']][NEURON['object']+ 7*n] = 10 * THRESHOLD # scene triggers
        connection[NEURON['object']+ 7*n][NEURON['object']+ 7*n] = THRESHOLD # self-connections
        connection[NEURON['object']+ 7*n][NEURON['count']] = THRESHOLD # count connections
        # copy
        connection[NEURON['scene']][NEURON['object_copy']+ 7*n] = THRESHOLD-COPY+epsilon # gates
        connection[NEURON['object']+ 7*n][NEURON['object_copy']+ 7*n] = COPY # channels
        connection[NEURON['object_copy']+ 7*n][NEURON['object_copy']+ 7*n] = THRESHOLD # self-connections
        connection[NEURON['object_copy']+ 7*n][NEURON['count_copy']] = THRESHOLD # count connections
        # retrival
        connection[NEURON['union']][NEURON['object']+ 7*n] = THRESHOLD-RETRIVAL+epsilon # gates 'union'
        connection[NEURON['intersect']][NEURON['object']+ 7*n] = -RETRIVAL+epsilon # gates 'intersect'
        connection[NEURON['object_copy']+ 7*n][NEURON['object']+ 7*n] = RETRIVAL # channels
        # self-inhibition
        connection[NEURON['self_inh']][NEURON['object_inh']+ 7*n] = THRESHOLD-SAME+epsilon # gates
        connection[NEURON['object']+ 7*n][NEURON['object_inh']+ 7*n] = SAME # triggers
        connection[NEURON['object_inh']+ 7*n][NEURON['object']+ 7*n] = -10 * THRESHOLD # connections
        # same attributes
        connection[NEURON['same']][NEURON['object']+ 7*n] = THRESHOLD-SAME+epsilon # attribute gates
        # relation gates
        connection[NEURON['front']][NEURON['object_front']+ 7*n] = THRESHOLD-RELATION+epsilon
        connection[NEURON['behind']][NEURON['object_behind']+ 7*n] = THRESHOLD-RELATION+epsilon
        connection[NEURON['right']][NEURON['object_right']+ 7*n] = THRESHOLD-RELATION+epsilon
        connection[NEURON['left']][NEURON['object_left']+ 7*n] = THRESHOLD-RELATION+epsilon
        # relation connections
        connection[NEURON['object']+ 7*n][NEURON['object_front']+ 7*n] = RELATION
        connection[NEURON['object']+ 7*n][NEURON['object_behind']+ 7*n] = RELATION
        connection[NEURON['object']+ 7*n][NEURON['object_right']+ 7*n] = RELATION
        connection[NEURON['object']+ 7*n][NEURON['object_left']+ 7*n] = RELATION
        # relation inhibitions
        connection[NEURON['object_front']+ 7*n][NEURON['object']+ 7*n] = -THRESHOLD
        connection[NEURON['object_behind']+ 7*n][NEURON['object']+ 7*n] = -THRESHOLD
        connection[NEURON['object_right']+ 7*n][NEURON['object']+ 7*n] = -THRESHOLD
        connection[NEURON['object_left']+ 7*n][NEURON['object']+ 7*n] = -THRESHOLD
        
    # special connections
        # according to object attributes
        for key in ATTRIBUTE:
            connection[NEURON['object']+ 7*n][NEURON[scene[n][key]]]  = THRESHOLD # to attributes
            connection[NEURON[scene[n][key]]][NEURON['object']+ 7*n]  = SAME # from attributes
        
        # according to relative positions
        for m in range(n):
            if scene[n]['position'][0] < scene[m]['position'][0]: # front and behind
                connection[NEURON['object_front']+ 7*n][NEURON['object']+ 7*m] = THRESHOLD
                connection[NEURON['object_behind']+ 7*m][NEURON['object']+ 7*n] = THRESHOLD
            else:
                connection[NEURON['object_front']+ 7*m][NEURON['object']+ 7*n] = THRESHOLD
                connection[NEURON['object_behind']+ 7*n][NEURON['object']+ 7*m] = THRESHOLD
                
            if scene[n]['position'][1] < scene[m]['position'][1]: # right and left
                connection[NEURON['object_right']+ 7*n][NEURON['object']+ 7*m] = THRESHOLD
                connection[NEURON['object_left']+ 7*m][NEURON['object']+ 7*n] = THRESHOLD
            else:
                connection[NEURON['object_right']+ 7*m][NEURON['object']+ 7*n] = THRESHOLD
                connection[NEURON['object_left']+ 7*n][NEURON['object']+ 7*m] = THRESHOLD
    
    return connection


