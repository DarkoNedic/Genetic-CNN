import random
import numpy as np

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from dag import DAG, DAGValidationError

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib.learn.python.learn.datasets import mnist
import gzip
from deap import algorithms
from deap.tools import crossover, selection

import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Hyperparameter as hp
import datetime

########
#overriding method from tensorflow.contrib.learn.python.learn.datasets.mnist.py
#to make custom amount of classes
########
def _extract_labels(f, one_hot=False, num_classes=10):
    global NUM_CLASSES
    num_classes = NUM_CLASSES
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
      magic = mnist._read32(bytestream)
      if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
      num_items = mnist._read32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
      if one_hot:
        return mnist.dense_to_one_hot(labels, num_classes)
      return labels

mnist.extract_labels = _extract_labels

#custom method for deap.tools.crossover.py
#now the crossover is performed per stage/gene.
def crossover(ind1, ind2, q_cpb):
    #print("CROSSOVER")
    global BITS_INDICES
    size = min(len(ind1), len(ind2))
    #print("Performing Crossover:")
    #print(ind1)
    #print(ind2)
    for x in BITS_INDICES:
        if (random.random() < q_cpb):
            #a, b = random.sample(range(size), 2)
            a, b = x[0], x[1]-1
            if a > b:
                a, b = b, a
            #print(a,b)

            holes1, holes2 = [True]*size, [True]*size
            for i in range(size):
                if i < a or i > b:
                    holes1[ind2[i]] = False
                    holes2[ind1[i]] = False
	
            # We must keep the original values somewhere before scrambling everything
            temp1, temp2 = ind1, ind2
            k1 , k2 = b + 1, b + 1
            for i in range(size):
                if not holes1[temp1[(i + b + 1) % size]]:
                    ind1[k1 % size] = temp1[(i + b + 1) % size]
                    k1 += 1
                if not holes2[temp2[(i + b + 1) % size]]:
                    ind2[k2 % size] = temp2[(i + b + 1) % size]
                    k2 += 1
            # Swap the content between a and b (included)
            for i in range(a, b + 1):
                ind1[i], ind2[i] = ind2[i], ind1[i]
    #print(ind1)
    #print(ind2)
    return ind1, ind2

#overriding method from deap.algorithms.py
def _varAnd(population, toolbox, cxpb, mutpb):
    q_cpb = hp.STAGE_CROSSOVER_PROB
    offspring = [toolbox.clone(ind) for ind in population]
    
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = crossover(offspring[i-1], offspring[i], q_cpb)
            del offspring[i-1].fitness.values, offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring

algorithms.varAnd = _varAnd


#new way for best individuals instead of deap.tools.selection.py selBest()
class AllIndividuals:
    def __init__(self):
        self.individuals = []
    def add_ind(self, ind, fitness):
        self.individuals.append((fitness, ind))
    def top(self, k = -1):
        if (k == -1):
            k = len(self.individuals)
        top = np.asarray(self.individuals)
        return sorted(self.individuals, reverse=True, key=lambda x: x[0])[:k]
#######

NUM_CLASSES = hp.CLASSES
INPUT_SIZE = hp.IMAGE_RESOLUTION # mnist is 28x28 pixel - tested on squared images only, sorry
mnist = input_data.read_data_sets(hp.IDX_DATA_FOLDER, one_hot=True, validation_size=hp.VALIDATION_SIZE)
train_imgs   = mnist.train.images[:hp.TRAIN_SIZE]
train_labels = mnist.train.labels[:hp.TRAIN_SIZE]
test_imgs    = mnist.test.images[:hp.TEST_SIZE]
test_labels  = mnist.test.labels[:hp.TEST_SIZE]
print(len(train_imgs))

train_imgs = np.reshape(train_imgs,[-1,INPUT_SIZE,INPUT_SIZE,1])
test_imgs = np.reshape(test_imgs,[-1,INPUT_SIZE,INPUT_SIZE,1])

STAGES = np.array(hp.STAGES) # S
NUM_NODES = np.array(hp.NUM_NODES_PER_STAGE) # K

# amount of filters per stage
NUM_FILTER = hp.NUM_FILTER#[4,8,12,12]
KERNEL_SIZE_PER_STAGE = hp.KERNEL_SIZE_PER_STAGE

# flag - do not change
first_stage = 0

# 0 to disable - adds a fully connected layer at the end
FLAG_FC_END = hp.END_FC

# size of fully connected layer  
FC_SIZE = hp.FC_SIZE

# flag to print extra stuff for debugging purposes
PRINT_TENSORS = 0
PRINT_DEBUG = 1

#useful for debugging
CUSTOM_IND = hp.CUSTOM_IND

L = 0  # genome length
BITS_INDICES = np.empty((0,2),dtype = np.int32)
start = 0
end = 0
for x in NUM_NODES:
    end = end + sum(range(x))
    BITS_INDICES = np.vstack([BITS_INDICES,[start, end]])
    start = end
L = end
print(BITS_INDICES)

TRAINING_EPOCHS = hp.TRAINING_EPOCHS
BATCH_SIZE = hp.BATCH_SIZE
TOTAL_BATCHES = train_imgs.shape[0] // BATCH_SIZE


def weight_variable(weight_name, weight_shape):
    return tf.Variable(tf.truncated_normal(weight_shape, stddev = 0.1),name = ''.join(["weight_", weight_name]))

def bias_variable(bias_name,bias_shape):
    return tf.Variable(tf.constant(0.01, shape = bias_shape),name = ''.join(["bias_", bias_name]))

def linear_layer(x,n_hidden_units,layer_name):
    n_input = int(x.get_shape()[1])
    weights = weight_variable(layer_name,[n_input, n_hidden_units])
    biases = bias_variable(layer_name,[n_hidden_units])
    return tf.add(tf.matmul(x,weights),biases)

def apply_convolution(x,kernel_height,kernel_width,num_channels,depth,layer_name):
    weights = weight_variable(layer_name,[kernel_height, kernel_width, num_channels, depth])
    biases = bias_variable(layer_name,[depth])
    global PRINT_TENSORS
    if (PRINT_TENSORS == 1):
        print(x)
    #return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights,[1,2,2,1],padding = "SAME"),biases)) 
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights,strides=[1,1,1,1],padding = "SAME"),biases))
    #1,1,1,1 to keep the 'spatial resolution' the same inside one stage as mentioned in paper

def apply_pool(x,kernel_height,kernel_width,stride_size):
    #return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], 
    #        strides=[1, 1, stride_size, 1], padding = "SAME")
    global PRINT_TENSORS
    if (PRINT_TENSORS == 1):
        print(x)
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, 2,2, 1], padding = "SAME")# to downsample by factor 2 as in paper

def apply_fully_connected(x, kernel_height, kernel_width, num_channels, depth, layer_name):
    weights = weight_variable(layer_name,[kernel_height, kernel_width, num_channels, depth])
    biases = bias_variable(layer_name,[depth])
    
    global FC_SIZE, NUM_FILTER
    flattened = tf.reshape(tf.nn.relu(tf.add(tf.nn.conv2d(x, weights,[1,1,1,1],padding = "SAME"), biases)), [-1, int(x.get_shape()[1]) * int(x.get_shape()[2]) * NUM_FILTER[len(NUM_FILTER)-1]])
    wd1 = tf.Variable(tf.truncated_normal([int(x.get_shape()[1]) * int(x.get_shape()[2]) * NUM_FILTER[len(NUM_FILTER)-1], FC_SIZE], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([FC_SIZE], stddev=0.01), name='bd1')
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
    if (PRINT_TENSORS == 1 or PRINT_DEBUG == 1):
        print("XX:",x)
        print(str(layer_name)+":", "Input("+str(num_channels)+"),", "Output("+str(FC_SIZE)+")")
        print(dense_layer1)
    return dense_layer1

def add_node(node_name, connector_node_name, h = 5, w = 5, nc = 1, d = 1):
    global NUM_FILTER, first_stage, PRINT_DEBUG
    stagenumber = int(re.findall("\d+", node_name)[0])
    if (first_stage == 0):
        first_stage = 1
        nc = 1 # change to 3 for RGB images - not tested
        d = NUM_FILTER[0]
    else:
        nc = NUM_FILTER[stagenumber-1]
        d = NUM_FILTER[stagenumber-1]
    h = KERNEL_SIZE_PER_STAGE[stagenumber-1]
    w = KERNEL_SIZE_PER_STAGE[stagenumber-1]
    if ("input" in str(node_name) and not "s1" in str(node_name)):
        nc = NUM_FILTER[stagenumber-2]
        d = NUM_FILTER[stagenumber-1]
    if (PRINT_DEBUG == 1):
        print("Kerel size:", (str(h) + "x" + str(w)))
        print(str(node_name)+":", "Input("+str(nc)+"),", "Output("+str(d)+")")
    global FLAG_FC_END
    with tf.name_scope(node_name) as scope:
        if (FLAG_FC_END == 1 and ''.join(["conv_",node_name]) == "conv_"+STAGES[len(STAGES)-1]+"_output"):
        #if (FLAG_FC_END == 1 and int((tf.get_default_graph().get_tensor_by_name(connector_node_name)).get_shape()[1]) == 1):
            dense = apply_fully_connected(tf.get_default_graph().get_tensor_by_name(connector_node_name), kernel_height = h, kernel_width = w, num_channels = nc , depth = d, layer_name = ''.join(["fc_",node_name]))
            #if (PRINT_DEBUG == 1):
                #print(dense)
        else:
            conv = apply_convolution(tf.get_default_graph().get_tensor_by_name(connector_node_name), kernel_height = h, kernel_width = w, num_channels = nc , depth = d, layer_name = ''.join(["conv_",node_name]))
            if (PRINT_DEBUG == 1):
                print(conv)

def sum_tensors(tensor_a,tensor_b,activation_function_pattern):
    if not tensor_a.startswith("Add"):
        tensor_a = ''.join([tensor_a,activation_function_pattern])
        
    return tf.add(tf.get_default_graph().get_tensor_by_name(tensor_a),
                 tf.get_default_graph().get_tensor_by_name(''.join([tensor_b,activation_function_pattern])))

def has_same_elements(x):
    return len(set(x)) <= 1

'''This method will come handy to first generate DAG independent of Tensorflow, 
    afterwards generated graph can be used to generate Tensorflow graph'''
def generate_dag(optimal_indvidual,stage_name,num_nodes):
    # create nodes for the graph
    nodes = np.empty((0), dtype = np.str)
    for n in range(1,(num_nodes + 1)):
        nodes = np.append(nodes,''.join([stage_name,"_",str(n)]))
    
    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    # split best indvidual found via GA to identify vertices connections and connect them in DAG 
    edges = np.split(optimal_indvidual,np.cumsum(range(num_nodes - 1)))[1:]
    v2 = 2
    for e in edges:
        v1 = 1
        for i in e:
            if i:
                dag.add_edge(''.join([stage_name,"_",str(v1)]),''.join([stage_name,"_",str(v2)])) 
            v1 += 1
        v2 += 1

    # delete nodes not connected to anyother node from DAG
    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])
    
    return dag, nodes

def generate_tensorflow_graph(individual,stages,num_nodes,bits_indices):
    print(datetime.datetime.now())
    global NUM_CLASSES, INPUT_SIZE, FLAG_FC_END, PRINT_DEBUG
    activation_function_pattern = "/Relu:0"
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = [None,INPUT_SIZE,INPUT_SIZE,1], name = "X")
    Y = tf.placeholder(tf.float32,[None,NUM_CLASSES],name = "Y")
        
    d_node = X
    for stage_name,num_node,bpi in zip(stages,num_nodes,bits_indices):
        indv = individual[bpi[0]:bpi[1]]
        if (PRINT_DEBUG == 1):
            print(indv)

        add_node(''.join([stage_name,"_input"]),d_node.name)
        pooling_layer_name = ''.join([stage_name,"_input",activation_function_pattern])

        if True or not has_same_elements(indv):
            # ------------------- Temporary DAG to hold all connections implied by GA solution ------------- #  

            # get DAG and nodes in the graph
            dag, nodes = generate_dag(indv,stage_name,num_node) 
            # get nodes without any predecessor, these will be connected to input node
            without_predecessors = dag.ind_nodes() 
            # get nodes without any successor, these will be connected to output node
            without_successors = dag.all_leaves()

            # ----------------------------------------------------------------------------------------------- #

            # --------------------------- Initialize tensforflow graph based on DAG ------------------------- #

            for wop in without_predecessors:
                add_node(wop,''.join([stage_name,"_input",activation_function_pattern]))

            for n in nodes:
                predecessors = dag.predecessors(n)
                if len(predecessors) == 0:
                    continue
                elif len(predecessors) > 1:
                    first_predecessor = predecessors[0]
                    for prd in range(1,len(predecessors)):
                        t = sum_tensors(first_predecessor,predecessors[prd],activation_function_pattern)
                        first_predecessor = t.name
                    add_node(n,first_predecessor)
                elif predecessors:
                    add_node(n,''.join([predecessors[0],activation_function_pattern]))

            if len(without_successors) > 1:
                first_successor = without_successors[0]
                for suc in range(1,len(without_successors)):
                    t = sum_tensors(first_successor,without_successors[suc],activation_function_pattern)
                    first_successor = t.name
                add_node(''.join([stage_name,"_output"]),first_successor) 
                pooling_layer_name = ''.join([stage_name,"_output",activation_function_pattern])
            elif len(without_successors) == 1:
                add_node(''.join([stage_name,"_output"]),''.join([without_successors[0],activation_function_pattern])) 
                pooling_layer_name = ''.join([stage_name,"_output",activation_function_pattern])
            # ------------------------------------------------------------------------------------------ #
        if (FLAG_FC_END == 1 and not pooling_layer_name == STAGES[len(STAGES)-1]+"_output/Relu:0"):
            d_node =  apply_pool(tf.get_default_graph().get_tensor_by_name(pooling_layer_name), kernel_height = 16, kernel_width = 16,stride_size = 1)
            if (PRINT_DEBUG == 1):
                print(d_node)
        if (FLAG_FC_END == 0):# and "_output/Relu:0" in pooling_layer_name):
            d_node =  apply_pool(tf.get_default_graph().get_tensor_by_name(pooling_layer_name), kernel_height = 16, kernel_width = 16,stride_size = 1)
            if (PRINT_DEBUG == 1):
                print(d_node)

    shape = d_node.get_shape().as_list()
    flat = tf.reshape(d_node, [-1, shape[1] * shape[2] * shape[3]])
    logits = linear_layer(flat,NUM_CLASSES,"logits")
    
    xentropy =  tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
    loss_function = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss_function) 
    accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(tf.nn.softmax(logits),1), tf.argmax(Y,1)), tf.float32))
    
    return  X, Y, optimizer, loss_function, accuracy

def evaluateModel(individual):
    global CUSTOM_IND, BI
    if (not len(CUSTOM_IND) == 0):
        individual = CUSTOM_IND
    print("Individual:", individual)
    score = 0.0
    X, Y, optimizer, loss_function, accuracy = generate_tensorflow_graph(individual,STAGES,NUM_NODES,BITS_INDICES)
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for epoch in range(TRAINING_EPOCHS):
            for b in range(TOTAL_BATCHES):
                offset = (epoch * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
                batch_x = train_imgs[offset:(offset + BATCH_SIZE), :, :, :]
                batch_y = train_labels[offset:(offset + BATCH_SIZE), :]
                _, c = session.run([optimizer, loss_function],feed_dict={X: batch_x, Y : batch_y})
                
        score = session.run(accuracy, feed_dict={X: test_imgs, Y: test_labels})
        print(datetime.datetime.now())
        print("\nIndividual:", individual)
        print('Accuracy:',score,"\n\n")
        print("########################################\n\n")
        BI.add_ind(individual, score)
    global first_stage
    first_stage = 0
    return score,

population_size = hp.POPULATION_SIZE
num_generations = hp.NUM_GENERATIONS

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("binary", bernoulli.rvs, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n = L)
toolbox.register("population", tools.initRepeat, list , toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
#toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 0.8)
toolbox.register("mutate", tools.mutFlipBit, indpb = hp.FLIP_BIT_PROB)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluateModel)

popl = toolbox.population(n = population_size)
BI = AllIndividuals()
#print(popl, "\n")
result = algorithms.eaSimple(popl, toolbox, cxpb = hp.CROSSOVER_PROBABILITY, mutpb = hp.MUTATION_PROBABILITY, ngen = num_generations, verbose = True)


# print top-3 optimal solutions 
'''
best_individuals = selBest(popl, k = 3)
for bi in best_individuals:
    print(bi)
'''
toplist = BI.top()
print()
for x in toplist:
    print(x[1])
    print("Fitness score:", x[0], "\n")
