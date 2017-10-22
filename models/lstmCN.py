import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

left_length = 25
entity_length = 12
right_length = 70


def lstmCN(corpus):
    global left_length, entity_length, right_length

    left_input = Input((left_length,), name='left_input')
    entity_input = Input((entity_length,), name='entity_input')
    right_input = Input((right_length,), name='right_input')

    left_word = Embedding(
        input_dim=corpus.vocabSize,
        output_dim=100,
        input_length=left_length,
        weights=[corpus.wordEmbedding],
        trainable=True,
        name='left_embedding'
    )(left_input)

    entity_word = Embedding(
        input_dim=corpus.vocabSize,
        output_dim=100,
        input_length=entity_length,
        weights=[corpus.wordEmbedding],
        trainable=True,
        name='entity_embedding'
    )(entity_input)

    right_word = Embedding(
        input_dim=corpus.vocabSize,
        output_dim=100,
        input_length=right_length,
        weights=[corpus.wordEmbedding],
        trainable=True,
        name='right_embedding'
    )(right_input)

    left_lstm = Bidirectional(
        LSTM(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='left_lstm'))(left_word)
    entity_lstm = Bidirectional(
        LSTM(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='entity_lstm'))(entity_word)
    right_lstm = Bidirectional(
        LSTM(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='right_lstm'))(right_word)

    merge_layer = concatenate([left_lstm, entity_lstm, right_lstm])
    hidden_1 = Dense(200, activation='relu', name='hidden_1')(merge_layer)
    hidden_2 = Dense(100, activation='relu', name='hidden_2')(hidden_1)
    output = Dense(corpus.labelDim, activation='sigmoid', name='output')(hidden_2)

    model = Model(inputs=[left_input, entity_input, right_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model