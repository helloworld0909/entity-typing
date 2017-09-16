import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

left_length = 130
entity_length = 6
right_length = 130
charEmbeddingDim = 30

def lstm(corpus):
    global left_length, entity_length, right_length

    left_input = Input((left_length, ), name='left_input')
    entity_input = Input((entity_length, ), name='entity_input')
    right_input = Input((right_length, ), name='right_input')

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

    left_char_input = Embedding(
        input_dim=corpus.tokenIdx2charVector.shape[0],
        output_dim=corpus.tokenIdx2charVector.shape[1],
        input_length=left_length,
        weights=[corpus.tokenIdx2charVector],
        trainable=False,
        name='left_char_input'
    )(left_input)

    entity_char_input = Embedding(
        input_dim=corpus.tokenIdx2charVector.shape[0],
        output_dim=corpus.tokenIdx2charVector.shape[1],
        input_length=entity_length,
        weights=[corpus.tokenIdx2charVector],
        trainable=False,
        name='entity_char_input'
    )(entity_input)

    right_char_input = Embedding(
        input_dim=corpus.tokenIdx2charVector.shape[0],
        output_dim=corpus.tokenIdx2charVector.shape[1],
        input_length=right_length,
        weights=[corpus.tokenIdx2charVector],
        trainable=False,
        name='right_char_input'
    )(right_input)

    left_char = TimeDistributed(Embedding(
        input_dim=len(corpus.char2idx),
        output_dim=charEmbeddingDim,
        weights=[corpus.charEmbedding],
        trainable=True,
    ),
        name='left_char'
    )(left_char_input)

    entity_char = TimeDistributed(Embedding(
        input_dim=len(corpus.char2idx),
        output_dim=charEmbeddingDim,
        weights=[corpus.charEmbedding],
        trainable=True,
    ),
        name='entity_char'
    )(entity_char_input)

    right_char = TimeDistributed(Embedding(
        input_dim=len(corpus.char2idx),
        output_dim=charEmbeddingDim,
        weights=[corpus.charEmbedding],
        trainable=True,
    ),
        name='right_char'
    )(right_char_input)

    left_char = TimeDistributed(LSTM(30, return_sequences=False), name='left_charLSTM')(left_char)
    entity_char = TimeDistributed(LSTM(30, return_sequences=False), name='entity_charLSTM')(entity_char)
    right_char = TimeDistributed(LSTM(30, return_sequences=False), name='right_charLSTM')(right_char)


    left = concatenate([left_word, left_char])
    entity = concatenate([entity_word, entity_char])
    right = concatenate([right_word, right_char])

    # TODO: 人工特征

    left_lstm = Bidirectional(LSTM(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='left_lstm'))(left)
    entity_lstm = Bidirectional(LSTM(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='entity_lstm'))(entity)
    right_lstm = Bidirectional(LSTM(100, return_sequences=False, recurrent_dropout=0.25, dropout=0.25, name='right_lstm'))(right)

    # TODO: 模型：分开的模型和合并的sentence的模型（多加一位0/1指明mention的位置）

    merge_layer = concatenate([left_lstm, entity_lstm, right_lstm])
    hidden_1 = Dense(200, activation='relu', name='hidden_1')(merge_layer)
    hidden_2 = Dense(100, activation='relu', name='hidden_2')(hidden_1)
    output = Dense(corpus.labelDim, activation='sigmoid', name='output')(hidden_2)

    model = Model(inputs=[left_input, entity_input, right_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model