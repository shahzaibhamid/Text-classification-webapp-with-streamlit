import tensorflow as tf

def get_mlp(units=128, num_layers=3, max_seq_len=128, embedding_matrix=None, num_words=5120):
    sequence_input = tf.keras.layers.Input(shape=(max_seq_len,), dtype="int32")
    
    if embedding_matrix is not None:
        num_words, embedding_dim = embedding_matrix.shape
        embedding_layer = tf.keras.layers.Embedding(input_dim=num_words,output_dim=embedding_dim,weights=[embedding_matrix],
                                                    input_length=max_seq_len,mask_zero=True,trainable=True)
    else:
        embedding_layer = tf.keras.layers.Embedding(input_dim=num_words, output_dim=units, input_length=max_seq_len, mask_zero=True, trainable=True)
        
    x = embedding_layer(sequence_input)   
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    
    l_drop = tf.keras.layers.Dropout(rate=0.1)(x)
    l_pool = tf.keras.layers.GlobalAveragePooling1D()(l_drop)
    preds = tf.keras.layers.Dense(units=2)(l_pool)
    
    model = tf.keras.models.Model(sequence_input, preds)
    return model

def get_cnn(units=128, num_layers=3, max_seq_len=128, embedding_matrix=None, num_words=5120):
    sequence_input = tf.keras.layers.Input(shape=(max_seq_len,), dtype="int32")
    
    if embedding_matrix is not None:
        num_words, embedding_dim = embedding_matrix.shape
        embedding_layer = tf.keras.layers.Embedding(input_dim=num_words,output_dim=embedding_dim,weights=[embedding_matrix],
                                                    input_length=max_seq_len,mask_zero=True,trainable=True)
    else:
        embedding_layer = tf.keras.layers.Embedding(input_dim=num_words,output_dim=units, input_length=max_seq_len,mask_zero=True,trainable=True)
    x = embedding_layer(sequence_input)
    for _ in range(num_layers):
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu")(x)
    
    l_drop = tf.keras.layers.Dropout(rate=0.1)(x)
    l_pool = tf.keras.layers.GlobalAveragePooling1D()(l_drop)
    preds = tf.keras.layers.Dense(units=2)(l_pool)
    
    model = tf.keras.models.Model(sequence_input, preds)
    return model

def get_lstm(units=128, num_layers=3, bidirectional=True, max_seq_len=128, embedding_matrix=None, num_words=5120):
    sequence_input = tf.keras.layers.Input(shape=(max_seq_len,), dtype="int32")
    
    if embedding_matrix is not None:
        num_words, embedding_dim = embedding_matrix.shape
        embedding_layer = tf.keras.layers.Embedding(input_dim=num_words,
                                                    output_dim=embedding_dim,
                                                     weights=[embedding_matrix],
                                                     input_length=max_seq_len,
                                                    mask_zero=True,
                                                     trainable=True)
    else:
         embedding_layer = tf.keras.layers.Embedding(input_dim=num_words,
                                                     output_dim=units,
                                                     input_length=max_seq_len,
                                                     mask_zero=True,
                                                     trainable=True)
 
    x = embedding_layer(sequence_input)
    
    
    
    for _ in range(num_layers-1):
        if bidirectional:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)
        else:
            x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
    
    if bidirectional:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=False))(x)
    else:
        x = tf.keras.layers.LSTM(units, return_sequences=False)(x)
       
    
    l_drop = tf.keras.layers.Dropout(rate=0.1)(x)
    preds = tf.keras.layers.Dense(units=2)(l_drop)
    
    
    model = tf.keras.models.Model(sequence_input, preds)
    return model
