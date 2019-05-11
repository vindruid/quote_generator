## Quote Generator Using Recurrent Neural Network 

![](https://images.unsplash.com/photo-1498435999018-6803de1f1c1f?ixlib=rb-0.3.5&s=e5fecd9f3003ad1f6b5eb51d69f4e930&auto=format&fit=crop&w=500&q=80)

Look at my [Medium Article](https://medium.com/data-folks-indonesia/build-your-own-quotes-generator-3a23e9cbcff3)

Dataset come from [Alvations in Kaggle.](https://www.kaggle.com/alvations/quotables/data)
It consist of of 36,165 quotes with 878,450 words from 2,297 famous people (Author, singer, politician, sportsman, scientist, etc.).

Basically the model is in character level, it take sequence of character and match with the next character.
Lower case and upper case is matter in this case. 
```
Example if we only consider sequence of 10 character
quote: The serve, I think, is the most difficult
input: 
('The serve,' , ' ') 
('he serve, ' , 'I')
('e serve, I' , ' ') 
(' serve, I ' , 't') 
....
```

The architecture is using Bidirectional LSTM then LSTM again then Dense layers. 
```python
input_sequences = Input((maxlen, len(chars)) , name="input_sequences")
lstm = Bidirectional(LSTM(256, return_sequences= True, input_shape=(maxlen, len(chars))), name = 'bidirectional')(input_sequences)
lstm = Dropout(0.1, name = 'dropout_bidirectional_lstm')(lstm)
lstm = LSTM(64, input_shape=(maxlen, len(chars)), name = 'lstm')(lstm)
lstm = Dropout(0.1,  name = 'drop_out_lstm')(lstm)

dense = Dense(15 * len(chars), name = 'first_dense')(lstm)
dense = Dropout(0.1,  name = 'drop_out_first_dense')(dense)
dense = Dense(5 * len(chars), name = 'second_dense')(dense)
dense = Dropout(0.1,  name = 'drop_out_second_dense')(dense)
dense = Dense(len(chars), name = 'last_dense')(dense)

next_char = Activation('softmax', name = 'activation')(dense)

model = Model([input_sequences], next_char)
model.compile(optimizer='adam', loss='categorical_crossentropy')
``` 
I trained it about 6 hours
After the network is trained, it is time to make the network produce full sentence (inspirational one hopefully). The steps is described as: 
1. Start with 10 initial character as out sequence, it can be randomly pick 1 or 2 words.
2. Insert the last 10 character in the sequence to RNN then we get the probabilities of next candidate characters.
3. We sample the next candidate with weight is the probabilities from RNN.
4. Repeat step 2 and 3 until we get the end of sentence.

```
Before Training:
I confessXvhYjyijV0OGBO!kX CLGfzfRwTbu'xiqaIs3&A5K?Uem"kYnlN0Y58w1??zWOk2xe'jxQSlDQCp3WbJIXdc"1ycDq&JFNSltR3sv0SNG?vtO?Sx,H6bHwGD,Bweqpojw7/J8bz'eSbvf
After Training:
The legal songs on the same credit the long time, and you love a man without love, understanding.
```

You can put initial words or chracter as you want.
If the input of function is empty, itu will pick 2 words randomly.

There are so many improvement you can to in quote generator
+ Use word level instead of character level.
+ Use larger network, combine bidirectional LSTM and then LSTM again sounds cool! 
+ Implement beam search method in generating part. 
+ Personalize quote with current mood (?). 

Original code from [keras example](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py) to generate text from Nietzche's writings. 

I change the network architecture, the data and make prediction function so we can have fun

## Cheers! 
