# %%
import random, glob
import numpy as np

# %%


# %%
music_files = [a for a in glob.glob("dataset/*/*")]
print("A random song", random.sample(music_files, 1))

# %%
from music21 import midi
def play_midi_file(midi_file_name):
    mf = midi.MidiFile()

    mf.open(midi_file_name) # path='abc.midi'
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    s.show('midi')

# %%
len(music_files)

# %%
from music21 import converter,corpus, chord

# %%



# %%


# %%



# %%


# %%


# %%


# %%
def get_score(music_files):
    chords = []
    for file_no, filename in enumerate(music_files):
        try:
            chords.append(converter.parse(filename))
            print(f'Happening of {file_no}', "filename = ", filename)
            
        except:
            print(f'Happening of {file_no}', "filename = ", filename)
            print("file failed!!!!!")
            continue
        
    
    return chords

# %%
music_files_5 = music_files[:5]
#get_chords(music_files[:10])

# %%
score_of_all_musics = get_score(music_files_5)

# %%
score_of_all_musics

# %%


# %%


# %%
from music21 import chord, duration

# %% [markdown]
# Classifying the music into different modes
# - firstly, let's code for major mode

# %%
from music21 import *

# %%


# %%


# %%
def round_chord_durations(number):
    if(number>1.25):
        return 2
    if(number>.30):
        return 1.25
    if(number>.10):
        return 0.3
    return 0.1

# %% [markdown]
# For chord and duration of a single song(music file)

# %%


# %%
def get_chord_and_duration_data(individual_score):
    chord_duration_data = []
    note_and_chord_sequence =[]
    
    note_and_chord_duration =[]   ## not for now:
    all_tempo= []
    
    #to see the number of tracks:
    print(len(individual_score.parts))
    
    for element in individual_score.flat:
        print("element = ", element, type(element))
        
        if isinstance(element, chord.Chord):
            note_and_chord_sequence.append('<SOC>') ## Start of Chord
            [note_and_chord_sequence.append(pitch.nameWithOctave) for pitch in element.pitches]
            note_and_chord_sequence.append('<EOC>') ## End of Chord
            
            chord_duration =str(round_chord_durations(element.duration.quarterLength))
            print (type(element.duration.quarterLength))
            note_and_chord_sequence.append(chord_duration)
            #print(chord_name, chord_duration)
            
        elif isinstance(element, note.Note):
            [note_and_chord_sequence.append(pitch.nameWithOctave) for pitch in element.pitches]
    
            note_duration = str(round_chord_durations(element.duration.quarterLength))
            note_and_chord_sequence.append(note_duration)
            #print(note_name, note_duration)
        
        elif isinstance(element, note.Rest):
            rest_note_name = element.name
            #print(rest_note_name)
            
        elif isinstance(element, tempo.MetronomeMark):
            tempo_bpm = element.getQuarterBPM()
            all_tempo.append(tempo_bpm)
            #print(tempo_bpm)
        else:
            print(element, type(element))
            
    
    print("tempo ko lagi = " ,np.quantile(all_tempo, .25), np.quantile(all_tempo, .50), np.quantile(all_tempo, .75) )
    print(note_and_chord_sequence)
    return note_and_chord_sequence

# %%
#get_chord_and_duration_data(score_of_all_musics[0])

# %%
chords_and_duration_data_all_music= []
for one_score in score_of_all_musics:
   chords_and_duration_data_all_music.append((get_chord_and_duration_data(one_score)))

# %%
# this is the main data:
len(chords_and_duration_data_all_music[0])
#np.asarray(chords_and_duration_data_all_music).shape

# %%
print("Generating music from our processed chords...")
proccessed_chords_to_midi_sample = get_music_midi_filename_from_chords(chords_and_duration_data_all_music[0][4:50])
print(proccessed_chords_to_midi_sample)
play_midi_file(proccessed_chords_to_midi_sample)

# %%


# %% [markdown]
# from Tonic_mode_all, separating major and minor songs 

# %%
tonic_mode_all[0][0], tonic_mode_all[0][-1]

# %%


# %%
all_major_songs_chords_and_duration = []
all_minor_songs_chords_and_duration = []
for i in range(len(tonic_mode_all)):
    if(tonic_mode_all[i][-1] == 'major'):
        try:
            all_major_songs_chords_and_duration.append(chords_and_duration_data_all_music[i])
        except:
            all_major_songs_chords_and_duration.append('NANNNNNN')
    else:
        try:
            all_minor_songs_chords_and_duration.append(chords_and_duration_data_all_music[i])
        except:
             all_minor_songs_chords_and_duration.append('NANNNNNN')

# %%
len(all_minor_songs_chords_and_duration), #minor_songs_chords_with_duration, len(minor_songs_chords_with_duration)

# %%
major_dataset = all_major_songs_chords_and_duration 
minor_dataset = all_minor_songs_chords_and_duration

# %%
#just a reference code
#major_dataset
#y=np.array(major_dataset)
#unique = set(major_dataset_all)
#unique2 = set(minor_dataset_all)

# %%
chords_and_duration_data_all_music[0][0]

# %%
main_dataset = chords_and_duration_data_all_music

# %%
len(main_dataset[0])

# %% [markdown]
# * Data preparation stage

# %%
#  !jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

# %%
no_of_timesteps = 32
x = []
y = []

## CD stands for chord and duration.

for each_music_with_CD in main_dataset:
    for each_CD in range(0, len(each_music_with_CD) - no_of_timesteps,  1):
        
        ## preparing input and output sequences:
        input_ = each_music_with_CD[each_CD:each_CD + no_of_timesteps]
        output = each_music_with_CD[each_CD + no_of_timesteps]
        #print(input_)
        
       
        
        x.append(input_)
        y.append(output)
        
x=np.array(x)
y=np.array(y)

# %%
x.shape

# %%


# %%
#assigning unique integer to every chords_and_duration

unique_x_CD = list(set(np.concatenate(x)))
unique_x_CD_to_int = dict((chord_and_duration, number) for number, chord_and_duration in enumerate(unique_x_CD))
unique_x_CD_to_int

# %%
#preparing input sequences::

x_seq=[]
for each_row in x:
    temp=[]
    for each_piece in each_row:
        #assigning unique integer to every note
        temp.append(unique_x_CD_to_int[each_piece])
    x_seq.append(temp)
    
x_seq = np.array(x_seq)
x_seq.shape

# %%
# preparing th output sequences as well::

unique_y_CD = list(set(y))
unique_y_CD_to_int = dict((chord_and_duration, number) for number, chord_and_duration in enumerate(unique_y_CD)) 
unique_y_CD_to_int

# %%
y_seq=np.array([unique_y_CD_to_int[i] for i in y])
y_seq.shape

# %%
# preserving 80% of the data for training and the rest 20% for the evaluation:

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_seq,y_seq,test_size=0.2,random_state=0)

# %%
len(x_train)

# %%
x_train.shape

# %% [markdown]
# # model building phase:

# %%
import tensorflow as tf

# %%
tf.config.experimental.list_physical_devices()

# %%
tf.__version__

# %%
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K
from tensorflow import keras

K.clear_session()
model = Sequential()
    
#embedding layer
model.add(Embedding(len(unique_x_CD), 100, input_length=32,trainable=True)) 

model.add(Conv1D(64,3, padding='causal',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
    
model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))

model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
          
#model.add(Conv1D(256,5,activation='relu'))    
model.add(GlobalMaxPool1D())
    
model.add(Dense(256, activation='relu'))
model.add(Dense(len(unique_y_CD), activation='softmax'))
    
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.summary()

# %%
def lstm():
    K.clear_session()
    model = Sequential()
    #embedding layer
    model.add(Input(shape= (None,)))
    model.add(Embedding(len(unique_x_CD), 100,trainable=True)) 
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(len(unique_y_CD), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

# %%
model = lstm()
model.summary()

# %%
!pip install pydot

# %%
keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

# %%
[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
#[print(l.name, l.input_shape, l.dtype) for l in model.layers]

# %%


# %%
!pip install h5py

# %%
import h5py

# %%
#defining call back to save the best model during training>
mc=ModelCheckpoint('my_best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)

# %%


# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
num_words= 71
#x_train = np.random.randint(num_words, size=(6771, 10))
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, padding="post"
)
padded_inputs_val = tf.keras.preprocessing.sequence.pad_sequences(
    x_val, padding="post"
)

# %%
len(padded_inputs[0])

# %%
len(padded_inputs_val[0])

# %%
#actual training
history = model.fit(np.array(x_train),np.array(y_train),batch_size=128,epochs=10, 
                   validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc])

# %%


# %%


# %%
#loading best model (Previously trained modle)
from keras.models import load_model
model = load_model('my_best_model.h5')


# %%
random_music = x_val[12]

# %%
random_music = random_music.reshape(1, 32 )
random_music

# %%


# %%
prob1 = model.predict(random_music)[0]
prob1

# %%
y_pred1= np.argmax(prob1,axis=0)
y_pred1

# %%
import numpy as np
import random
ind = np.random.randint(0,len(x_val)-1)
random_music = x_val[4]
random_music

# %%


# %%


# %%


# %%


# %%
no_of_timesteps = 32
predictions=[]
for i in range(10):

    random_music = random_music.reshape(1,no_of_timesteps)
    print("random music = ", random_music)
    

    prob  = model.predict(random_music)[0]
    y_pred= np.argmax(prob,axis=0)
    predictions.append(y_pred)

    random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
    random_music = random_music[1:]
    
print(predictions)

# %%
#intergers back to notes
unique_x_int_to_CD = dict((number, note_) for number, note_ in enumerate(unique_x_CD)) 
unique_x_int_to_CD

# %%
predicted_CD = [unique_x_int_to_CD[i] for i in predictions]
predicted_CD

# %%
predicted_CD_split = []
for each_outcome in predicted_CD:
    temp_list = []
    temp_list = each_outcome.split("@")
    temp_list[1] =float(temp_list[1])
    predicted_CD_split.append(tuple(temp_list))
    

predicted_CD_split

# %%
type(float('fds1.5'))

# %%
print("Generating music from our processed chords...")
proccessed_chords_to_midi_sample = get_music_midi_from_chords_and_duration(predicted_CD_split)
print(proccessed_chords_to_midi_sample)
play_midi_file(proccessed_chords_to_midi_sample)

# %%
def pred_out_to_midi(pred_output):
    
    #generate new score                  
    midi_score = stream.Score()
    
    
    for i in range(0, len(pred_output)):
        print(pred_output[i])
        
        if pred_output[i] == '<SOC>':
            while(pred_output[i] != '<EOC>'):
                print(pred_output[i])
                if (not pred_output[i][0].isdigit): 
                    i = i+1
                    new_chord = []
                    new_chord.append(pred_output[i])
            #out of while loop i.e end of one chord:
            midi_score.append(midi_score.flat.chord.Chord(new_chord))
        
        if (not pred_output[i][0].isdigit):
            temp_note = midi_score.flat.note.Note(pred_output[i])
            midi_score.append(temp_note)
            i =i+1
            midi_score.append(temp_note.duration.quarterLength(pred_output[i]))
        
        else : #else it is digit
            print(type(float(pred_output[i])))
           # midi_score.append(midi_score.flat.duration.quarterLength(1.5) )
            
    return converted_score  

# %%
p =pred_out_to_midi(predicted_CD)
p

# %%
#generate new stream                  
midi_stream = stream.Stream()

midi_stream
#converter.parse(midi_stream)


# %%


# %%


# %%


# %%


# %%
import random, glob
import numpy as np

# %%
music_files = [a for a in glob.glob("dataset/*/*")]
print("A random song", random.sample(music_files, 1))

# %%
len(music_files)

# %%
from music21 import midi
def play_midi_file(midi_file_name):
    mf = midi.MidiFile()

    mf.open(midi_file_name) # path='abc.midi'
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    s.show('midi')

# %%
# for seeing output from chords and given duration:
def get_music_midi_from_chords_and_duration(input_chords):
    midi_stream = stream.Stream()

    for note_pattern, duration_pattern in input_chords:
        notes_in_chord = note_pattern.split('.')
        
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(current_note)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = instrument.Violoncello()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        
        midi_stream.append(new_chord)

        new_tempo = tempo.MetronomeMark(number=50)
            
        midi_stream.append(new_tempo)

    midi_stream = midi_stream.chordify()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_file = 'output-' + timestr + '.mid'
    return midi_stream.write('midi', fp=new_file)

# %%



