from DataLoader.classification_data_Loader import generate_data,Classification_DATASET
from torch.utils.data import DataLoader



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_data,testing_data=generate_data("Data")

train_loader = DataLoader(Classification_DATASET(training_data), batch_size=2, shuffle=False)
test_loader = DataLoader(Classification_DATASET(testing_data), batch_size=2, shuffle=True)

#define NN
