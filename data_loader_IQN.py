import numpy as np
import pandas as pd
import os

def get_batch(x, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return batch_x

def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy array
    t = np.array(df[target].to_numpy().reshape(-1, 1))
    #where scaler_t is a StandardScaler() object, which has the .transorm method
    x = np.array(df[source])
    t = t.reshape(-1,)
    return t, x

X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

FIELDS  = {'RecoDatam' : {'inputs': X, 
                        'xlabel': r'$p_T$ (GeV)', 
                        'xmin': 0, 
                        'xmax':80},
        
        'RecoDatapT': {'inputs': ['RecoDatam']+X, 
                        'xlabel': r'$\eta$', 
                        'xmin'  : -8, 
                        'xmax'  :  8},
        
        'RecoDataeta': {'inputs': ['RecoDatam','RecoDatapT'] + X, 
                        'xlabel': r'$\phi$',
                        'xmin'  : -4,
                        'xmax'  :  4},
        
        'RecoDataphi'  : {'inputs': ['RecoDatam', 'RecodatapT', 'RecoDataeta']+X,
                        'xlabel': r'$m$ (GeV)',
                        'xmin'  : 0, 
                        'xmax'  :20}
        }



def get_data_set():
    # os.environ["DATA_DIR"]="/home/ali/Desktop/Pulled_Github_Repositorie/IQN_HEP/Davidson/data"
    DATA_DIR=os.environ["DATA_DIR"]
    # DATA_DIR = "~/Desktop/Pulled_Github_Repositorie/IQN_HEP/Davidson/data"
    target = 'RecoDatam'
    source  = FIELDS[target]
    features= source['inputs']
    ########
    print('USING NEW DATASET')
    train_data=pd.read_csv(DATA_DIR + '/train_data_10M_2.csv' )
    print('TRAINING FEATURES\n', train_data[features].head())

    test_data=pd.read_csv(DATA_DIR+'/test_data_10M_2.csv')
    valid_data=pd.read_csv(DATA_DIR+'/validation_data_10M_2.csv')

    print('train set shape:',  train_data.shape)
    print('validation set shape:', valid_data.shape)
    print('test set shape:  ', test_data.shape)    
    n_examples= int(8e6)
    batchsize=10
    N_batches = 10
    #N_batches=n_examples/batch_size
    train_t, train_x = split_t_x(train_data, target, features)
    
    test_t,  test_x  = split_t_x(test_data,  target, features)

	# for i in range(N_batches):
		# batch= #sample unifrm data in range (0,1) of size 2x2
		# examples.append(array_2d)
		#its generating a 2x2 array for each example, but when training, this will be flattened as a 1d aray for each example

    def training_set():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch(train_x, batchsize)
            #print('batch_x', batch_x)
            #index of one of the items in our examples
            yield batch_x

    def evaluation_set():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch(test_x,batchsize)
            #index of one of the items in our examples
            yield batch_x

    return training_set, evaluation_set







if __name__ == '__main__':
    train_generator, eval_generator=get_data_set()
    sample=next(train_generator())
    print('sample', sample)

