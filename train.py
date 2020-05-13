import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import shutil
import os
from tqdm import tqdm
import random

''' Define architecture '''
HIDDEN_LAYER_DIMS = [500, 500, 2000]
OUTPUT_DIM = 2


def nn_model(n_features, dropout_rate, l2_lambda):
    """
    Returns a tf.keras.Model for cell-type classification with the specifications
    listed above.

    Arguments:
      n_features: the number of features of the datapoints used
        as input to the model (used to determine the input shape)
      dropout_rate: the dropout rate of the dropout layers
      l2_lambda: the weight of the L2 regularization penalty on the
        weights (but not the biases) of the model

    Returns:
      model: a tf.keras.Model for tSNE with the specifications
        listed above
    """
    model = K.Sequential()

    l2_reg = tf.keras.regularizers.l2(l=l2_lambda)

    model.add(tf.keras.layers.Dense(HIDDEN_LAYER_DIMS[0], batch_input_shape=(None, n_features),
                                    activation='relu', kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(HIDDEN_LAYER_DIMS[1], batch_input_shape=(None, n_features),
                                    activation='relu', kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(HIDDEN_LAYER_DIMS[2], batch_input_shape=(None, n_features),
                                    activation='relu', kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(OUTPUT_DIM, batch_input_shape=(None, n_features),
                                    activation=tf.nn.softmax, kernel_regularizer=l2_reg))


    return model



def sample_shuffle_data(arrays, n_samples=None):
    """subsamples examples from a list of datasets

    samples `n_samples` without replacement from along the first dimension
    of each array in `arrays`. The same first-dimension slices are
    selected for each array in `arrays`.

    Arguments:
      arrays: the arrays to be sliced, all must have the same size along
        their first dimension.
      n_samples: (None) the number of samples to be selected, `n_samples` must
        be less than or equal to the length of the arrays. If n_samples it not
        passed or is `None`. Then each array in `arrays` will be
        shuffled in the same way and returned.

    Returns:
      sampled: a `tuple` of len the same as `len(arrays)` where each
        element is an array of len `n_samples`
    """
    batch_len = arrays[0].shape[0]
    n_samples = batch_len if n_samples is None else n_samples

    err_msg = 'all arrays must have the same size along their first dimension'
    assert all(batch_len == x.shape[0] for x in arrays), err_msg
    err_msg = 'n cannot be greater then the length of the arrays'
    assert n_samples <= batch_len, err_msg

    sampling_idxs = tf.random.shuffle(tf.range(batch_len))[:n_samples]
    sampled = tuple(tf.gather(x, sampling_idxs, axis=0) for x in arrays)

    return sampled




def train_step(model, loss, optimizer, x_batch, y_batch):
    """
    Performs one training step on a model given a loss, optimizer, inputs,
    and labels.

    Arguments:
      model: the model on which the pass will be performed
      loss: the loss function to be evaluated, from which the gradients will be
        computed
      optimizer: a `tf.optimizers` object defining the optimization scheme
      x_batch: model training inputs
      y_batch: model training labels

    Returns:
      loss_value: the computed loss for the forward training pass
    """
    #     print('X_batch : {} \n y_batch : {}'.format(x_batch, y_batch))
    with tf.GradientTape() as tape:
        #         print(f'x_batch: {x_batch}')
        #         print(f'y_batch: {y_batch}')

        y_batch_pred = model(x_batch, training=True)
        loss_value = loss(y_batch, y_batch_pred)
        #         print(f"model losses type: {model.losses}")
        loss_value += sum(model.losses)
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss_value


def training(x_train, y_train, x_val, y_val, hyperparam_config, num_epochs, batch_size,
             stop_count=10, save_model=True, model_dir='models/best_model'):
    """
    Train a fully-connected network to classify foreground or background

    Arguments:
      x_train: input training set
      y_train: label training set
      x_val: input validation set
      y_val: label validation set
      hyperparam_config: a dictionary that stores a hyperparameter configuration,
                         including:
                           - "dropout_rate": dropout rate (1 - keep probability),
                           - "l2": coefficient lambda for L2 regularization,
                           - "lr": learning rate for RMSProp optimizer
      num_epochs: number of epochs to train
      batch_size: training mini-batch size (must be same as the batch size for pairwise P calculation)
      save_model: whether or not to save the best model based on the validation loss
      model_dir: location where model will be saved

    Returns:
      best_loss: best validation loss
    """
    
    # wrapping train_step to speed up training
    train_fn = tf.function(train_step)
    #     train_fn = train_step

    # get num_batches
    num_samples, num_features = x_train.shape
    num_batches = np.floor(num_samples) // batch_size

    # initalize model, loss, and optimizers
    dropout_rate = hyperparam_config['dropout_rate']
    l2_lambda = hyperparam_config['l2_lambda']
    lr = hyperparam_config['lr']
    model = nn_model(n_features=num_features, dropout_rate=dropout_rate, l2_lambda=l2_lambda)
    loss_fn = tf.keras.losses.BinaryCrossentropy()  # todo - make sure this is correct
    optimizer = K.optimizers.SGD(learning_rate=lr)  # TODO - double check this optimizer

    # Get initial loss for comparison
    best_loss = loss_fn(y_val, model(x_val))

    # init progress bars
#     epoch_pbar = tqdm(total=num_epochs, desc="Training Epochs")
#     batch_pbar = tqdm(desc="Training Steps")

    n = x_train.shape[0]
    # for each epoch
    # start training loop
    losses = [best_loss]
    prev_loss = best_loss
    early_stop_count = 0
    for epoch in range(num_epochs):
        if early_stop_count >= stop_count: break
        # shuffle data
        if epoch >= 1: x_train, y_train = sample_shuffle_data([x_train, y_train], num_samples)

#         batch_pbar.reset(num_batches)
        for step in range(int(num_batches)):
            # getting indices of batches to train on
            range_begin = (step * batch_size) % (x_train.shape[0] - batch_size)  # taking mod to prevent ix errors
            range_end = range_begin + batch_size
            batch_x = x_train[range_begin:range_end, :]
            batch_y = y_train[range_begin:range_end, :]
            epoch_loss = train_fn(model, loss_fn, optimizer,
                                  batch_x, batch_y)

#             batch_pbar.update()

        # compute and print loss on validation data
        val_loss = loss_fn(y_val, model(x_val))  # note - don't need reg_coeff defined because already defined using functools.partial
        
        # check for early stopping
        if epoch_loss >= prev_loss:
            early_stop_count += 1
        else:
            early_stop_count = 0
        prev_loss = epoch_loss

#         if epoch % 5 == 0:
        tf.print("epoch: {:02d}, loss: {:5.3f}".format(epoch, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                # if directory hasn't been created, create it
                if not os.path.isdir('models'):
                    os.mkdir('models')
                # if model has already been saved, remove folder and save again
                if os.path.isdir(model_dir):
                    shutil.rmtree(model_dir)
                # make directory again and save
                #                 !mkdir -p models/best_loss
                model.save(model_dir)
#         batch_pbar.refresh()
#         epoch_pbar.update()


    return best_loss



def train_val(x_val, y_val, hyperparam_config, num_epochs, batch_size):
    """
    Train a fully-connected network to classify foreground or background

    Arguments:
      x_val: input validation set
      y_val: label validation set
      hyperparam_config: a dictionary that stores a hyperparameter configuration,
                         including:
                           - "dropout_rate": dropout rate (1 - keep probability),
                           - "l2": coefficient lambda for L2 regularization,
                           - "lr": learning rate for RMSProp optimizer
      num_epochs: number of epochs to train
      batch_size: training mini-batch size (must be same as the batch size for pairwise P calculation)

    Returns:
      best_loss: best validation loss
    """
    
    # wrapping train_step to speed up training
    train_fn = tf.function(train_step)
    #     train_fn = train_step

    # get num_batches
    num_samples, num_features = x_val.shape
    num_batches = np.floor(num_samples) // batch_size

    # initalize model, loss, and optimizers
    dropout_rate = hyperparam_config['dropout_rate']
    l2_lambda = hyperparam_config['l2_lambda']
    lr = hyperparam_config['lr']
    model = nn_model(n_features=num_features, dropout_rate=dropout_rate, l2_lambda=l2_lambda)
    loss_fn = tf.keras.losses.BinaryCrossentropy()  # todo - make sure this is correct
    optimizer = K.optimizers.SGD(learning_rate=lr)  # TODO - double check this optimizer

    best_loss = loss_fn(y_val, model(x_val))


    n = x_val.shape[0]
    # for each epoch
    # start training loop
    for epoch in range(num_epochs):
        # shuffle data
        if epoch >= 1: x_val, y_val = sample_shuffle_data([x_val, y_val], num_samples)

        for step in range(int(num_batches)):
            # getting indices of batches to train on
            range_begin = (step * batch_size) % (x_val.shape[0] - batch_size)  # taking mod to prevent ix errors
            range_end = range_begin + batch_size
            batch_x = x_val[range_begin:range_end, :]
            batch_y = y_val[range_begin:range_end, :]
            epoch_loss = train_fn(model, loss_fn, optimizer,
                                  batch_x, batch_y)


        # compute and print loss on validation data
        val_loss = loss_fn(y_val, model(x_val))  # note - don't need reg_coeff defined because already defined using functools.partial

        tf.print("epoch: {:02d}, loss: {:5.3f}".format(epoch, val_loss))


    return best_loss




def grid_search(x_train, y_train, dropout_rates, l2_lambdas, learning_rates, num_epochs=40, batch_size=300):
    """
    Perform grid search for the best tSNE hyperparameters

    Arguments:
      x_train: input training set
      y_train: label training set
      dropout_rates: dropout rates to try
      l2_lambdas: L2 lambda coefficients to try
      learning_rates: learning rates to try
      num_epochs: number of epochs to train
      batch_size: training mini-batch size

    Returns:
      losses: list losses for configurations tested where
        losses[i] = [dropout_rate, l2_lambda, learning_rate, best_loss, best_kl_divgergence]
    """
    losses = []

    pbar = tqdm(total=len(dropout_rates) * len(l2_lambdas) * len(learning_rates))
    for dropout_rate in dropout_rates:
        for l2_lambda in l2_lambdas:
            for learning_rate in learning_rates:
                print("training with dropout:{} l2:{} lr:{}".format(dropout_rate, l2_lambda, learning_rate))
                # DO NOT shuffle your validation/train set because the pairwise label are calculated by batch
                # Use the last batch in train set as the validation set
                subset_x_train, subset_y_train = (x_train[0:-batch_size], y_train[0:-batch_size])
                subset_x_val, subset_y_val = (x_train[-batch_size:], y_train[-batch_size:])
                hyperparam_config = {'dropout_rate': dropout_rate,
                                     'l2_lambda': l2_lambda,
                                     'lr': learning_rate}

                best_loss = training(subset_x_train, subset_y_train,
                                     subset_x_val, subset_y_val,
                                     hyperparam_config,
                                     num_epochs,
                                     batch_size,
                                     save_model=False)

                losses.append([dropout_rate, l2_lambda, learning_rate, best_loss])
                pbar.update(1)
    pbar.close()
    return losses



def random_search(x_val, y_val, param_grid, max_iter=12, num_epochs=10, batch_size=300):
    """

    Arguments:
      x_train: input training set
      y_train: label training set
      param_grid is a dictionary where keys are hyperparameter names and values are the search space

    Returns:
      losses: list losses for configurations tested where
        losses[i] = [dropout_rate, l2_lambda, learning_rate, best_loss, best_kl_divgergence]
    """
    losses = []

    pbar = tqdm(total=max_iter)
    for _ in range(max_iter):
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        print(f"training with params: {random_params}")
        # DO NOT shuffle your validation/train set because the pairwise label are calculated by batch


        best_loss = train_val(x_val, y_val,
                                    random_params,
                                    num_epochs,
                                    batch_size)
        losses.append([random_params, best_loss])
        pbar.update(1)
    pbar.close()
    return losses


## Evaluate model performance

def predict_one_hot(x, model_dir):
    """
    Load a trained model and predict class

    Arguments:
      x: input data of size [n_datapoints, n_features]
      model_dir: location of saved model

    Returns:
      y_output: one_hot representation of model output where argmax
                is just chosen as class
    """

    model = K.models.load_model(model_dir)
    y_pred = model.predict(x)

    # convert to one-hot representation
    y_output = tf.one_hot(tf.nn.top_k(y_pred).indices, tf.shape(y_pred)[1])

    return tf.squeeze(y_output)


def predict(x, model_dir):
    """
    Load a trained model and predict probability for each class

    Arguments:
      x: input data of size [n_datapoints, n_features]
      model_dir: location of saved model

    Returns:
        y_output: softmax rep of class labels
    """

    model = K.models.load_model(model_dir)
    y_pred = model.predict(x)

    return y_pred


def get_labels_from_one_hot(one_hot_labels):
    ''' Get labels from one-hot encoding '''
    return np.argmax(one_hot_labels.numpy(), axis=1)


def accuracy(y_true, y_pred):
    ''' Computes accuracy '''
    acc = tf.keras.metrics.Accuracy()
    acc.update_state(y_true, y_pred)
    return acc.result().numpy()


def precision(y_true, y_pred):
    ''' Computes precision '''
    prec = tf.keras.metrics.Precision()
    prec.update_state(y_true, y_pred)
    return prec.result().numpy()


def recall(y_true, y_pred):
    ''' Computes recall'''
    rec = tf.keras.metrics.Recall()
    rec.update_state(y_true, y_pred)
    return rec.result().numpy()


def TPR(y_true, y_pred, thresholds=None):
    ''' Computes True positive rate '''
    tp = tf.keras.metrics.TruePositives()
    tp.update_state(y_true, y_pred)
    return tp.result().numpy()


def FPR(y_true, y_pred, thresholds=None):
    ''' Computes false positive rate '''
    fp = tf.keras.metrics.FalsePositives()
    fp.update_state(y_true, y_pred)
    return fp.result().numpy()



def compute_ROC_curve(y_true, y_pred):
    pass


def make_confusion_matrix_DEX(y_test, y_pred, cmap='Blues', subtitle='INSERT SUBTITLE'):
    y_true_labeled = get_labels_from_one_hot(y_test)
    y_pred_labeled = get_labels_from_one_hot(y_pred)
    conf_matrix = tf.math.confusion_matrix(y_true_labeled, y_pred_labeled, dtype=tf.dtypes.int64).numpy()
    fig, axs = plt.subplots(1)
    sns.heatmap(conf_matrix, annot=True, cmap=cmap,
                xticklabels=['0 hr', '1 hr', '3 hrs'],
                yticklabels=['0 hr', '1 hr', '3 hrs'],
                ax=axs);
    subtitle = subtitle
    axs.set_title(f'Confusion Matrix\n{subtitle}')
