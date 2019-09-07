########################
'''Number of classes'''#
########################
CLASSES = 2

#######################################
'''Directory to the Data and       '''#
'''image size (only squared images)'''#
#######################################
IDX_DATA_FOLDER = "ball_data/"
IMAGE_RESOLUTION = 16

#########################
'''Dataset parameters'''#
#########################
VALIDATION_SIZE = 0
TRAIN_SIZE = None
TEST_SIZE = None

##########################
'''Training parameters'''#
##########################
TRAINING_EPOCHS = 20
BATCH_SIZE = 50
POPULATION_SIZE = 20
NUM_GENERATIONS = 30

####################################
'''Number of stages             '''#
'''Number of nodes per stage    '''#
'''Number of filter for each    '''#
'''convolutional layer per stage'''#
'''Size of kernels per stage    '''#
'''(squared only. e.g. 3 = 3x3) '''#
####################################
STAGES = ["s1","s2","s3","s4"]
NUM_NODES_PER_STAGE = [3,3,4,5]
NUM_FILTER = [16,16,32,32]
KERNEL_SIZE_PER_STAGE =[5,3,3,2]


#####################################
'''Adds a fully-connected layer  '''#
'''at the end + the size of nodes'''#
#####################################
END_FC = 1
FC_SIZE = 1000

###############################################
'''Possibility to go with custom individual'''#
###############################################
CUSTOM_IND = []#[0, 0, 0, 1, 0, 0, 1, 1, 1]

######################################
'''Probability parameters         '''#
'''p_c, q_c, p_m, p_c respectively'''#
######################################
CROSSOVER_PROBABILITY=0.2
STAGE_CROSSOVER_PROB=0.3
MUTATION_PROBABILITY=0.8
FLIP_BIT_PROB=0.1

