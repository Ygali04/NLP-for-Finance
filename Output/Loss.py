from Model import Dataset, Preprocess, FinanceNetwork

training_loss = history.history['loss']
validation_loss = history.history["val_loss"]

fig = plt.figure(figsize=(12,6))
plt.title('Loss over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(training_loss, label="train")
plt.plot(validation_loss, label="validation")
plt.legend()
plt.show()
