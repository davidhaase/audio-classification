import os
import re
import pickle

from audioprocessor import Processor

prediction_dir = 'pickles/predictions/'
pattern = re.compile(r'test_prediction_(\d+)\.pkl')
settings_file = 'settings/setting_1.pkl'
data = pickle.load( open( settings_file, 'rb' ) )

Sounds = Processor(data)
Sounds.process_training()
Sounds.run(300)

Sounds.show_accuracy()
Sounds.show_loss()

Sounds.prep_x_test()
Sounds.predict()


start = True
max_value = 0
for f in os.listdir(prediction_dir):
    match = pattern.search(f)
    if (match):
        start = False
        file_count = int(match.group(1))
        if file_count > max_value:
            max_value = file_count

if start:
    max_value == 0
else:
    max_value += 1

filename = 'test_prediction_' + str(max_value) + '.pkl'


Sounds.test_df.to_pickle(prediction_dir + filename)


if os.path.isfile(data['history_file']):
    history = pickle.load(open(data['history_file'], 'rb'))
else:
    history = []

trial = {}
trial['history'] = Sounds.history.history
trial['settings'] = data

history.append(trial)
pickle.dump( history, open( data['history_file'], 'wb') )
