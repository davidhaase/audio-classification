import os
import pickle

from audioprocessor import Processor

settings_file = 'settings/setting_1.pkl'
data = pickle.load( open( settings_file, 'rb' ) )

Sounds = Processor(data)
Sounds.process_training()
Sounds.run(100)

Sounds.show_accuracy()
Sounds.show_loss()

Sounds.prep_x_test()
Sounds.predict()


if os.path.isfile(data['history_file']):
    history = pickle.load(open(data['history_file'], 'rb'))
else:
    history = []

trial = {}
trial['history'] = Sounds.history.history
trial['settings'] = data

history.append(trial)
pickle.dump( history, open( data['history_file'], 'wb') )
