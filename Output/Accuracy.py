from Model import Dataset, Preprocess, FinanceNetwork

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

fig = plt.figure(figsize=(12,6))
plt.title('Accuracy over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(training_accuracy, label="train")
plt.plot(validation_accuracy, label="validation")
plt.legend()
plt.show()
