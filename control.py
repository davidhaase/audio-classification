import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential

from audioprocessor import Processor

data_dir = '../../Datasets/urban-sound-classification/'
train_csv = 'train.csv'
test_csv = 'test.csv'

Sounds = Processor(data_dir, train_csv, test_csv)
Sounds.process_training()
acc, val_acc, loss, val_loss = Sounds.run(800)

Sounds.show_accuracy()
Sounds.show_loss()

Sounds.prep_x_test()
Sounds.predict()
