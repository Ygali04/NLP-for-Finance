import Dataset
import Preprocess
import FinanceNetwork
import Loss
import Accuracy

fig = plt.figure(figsize=(12,6))
plt.title('Loss over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(training_loss, label="train")
plt.plot(validation_loss, label="validation")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12,6))
plt.title('Accuracy over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(training_accuracy, label="train")
plt.plot(validation_accuracy, label="validation")
plt.legend()
plt.show()
