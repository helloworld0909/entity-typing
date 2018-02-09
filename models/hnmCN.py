import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

left_length = 25
entity_length = 12
right_length = 70

def hnmCN(corpus):
    global left_length, entity_length, right_length

    left_input = Input((left_length,), name='left_input')
    entity_input = Input((entity_length,), name='entity_input')
    right_input = Input((right_length,), name='right_input')

    left_word = Embedding(
        input_dim=corpus.vocabSize,
        output_dim=100,
        input_length=left_length,
        weights=[corpus.wordEmbedding],
        trainable=False,
        name='left_embedding'
    )(left_input)

    entity_word = Embedding(
        input_dim=corpus.vocabSize,
        output_dim=100,
        input_length=entity_length,
        weights=[corpus.wordEmbedding],
        trainable=False,
        name='entity_embedding'
    )(entity_input)

    right_word = Embedding(
        input_dim=corpus.vocabSize,
        output_dim=100,
        input_length=right_length,
        weights=[corpus.wordEmbedding],
        trainable=False,
        name='right_embedding'
    )(right_input)

    left = Reshape((left_length * 100,), input_shape=(left_length, 100))(left_word)

    left_hidden = Dense(1000)(left)
    left_hidden = Dense(100)(left_hidden)

    entity_rnn = SimpleRNN(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='entity_rnn')(
        entity_word)

    right = Reshape((right_length * 100,), input_shape=(right_length, 100))(right_word)
    right_hidden = Dense(1000)(right)
    right_hidden = Dense(100)(right_hidden)

    merge_layer = concatenate([left_hidden, entity_rnn, right_hidden])

    hidden_1 = Dense(200, activation='relu', name='hidden_1')(merge_layer)
    hidden_2 = Dense(100, activation='relu', name='hidden_2')(hidden_1)
    output = Dense(corpus.labelDim, activation='sigmoid', name='output')(hidden_2)

    model = Model(inputs=[left_input, entity_input, right_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

