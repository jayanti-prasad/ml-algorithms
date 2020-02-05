import os
from tensorflow.compat.v1.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

def get_utils (cfg):

    model_checkpoint = ModelCheckpoint( cfg.workspace  + os.sep + 'model.hdf5',
         monitor='val_loss', save_best_only=True, period=1)

    csv_logger = CSVLogger(cfg.workspace + os.sep + 'history.log')

    tensorboard = TensorBoard(log_dir = cfg.workspace + os.sep + 'tensorboard',
        histogram_freq = 10,
        batch_size = cfg.batch_size,
        write_graph = True,
        write_grads = False,
        write_images = False,
        embeddings_freq = 0,
        embeddings_layer_names = None,
        embeddings_metadata = None,
        embeddings_data = None)

    return model_checkpoint, csv_logger, tensorboard 
 


