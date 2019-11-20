import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import datasets, models, transforms
from skimage import io, transform


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(17 * 17 * 32, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, _x):
        _x = self.pool(F.relu(self.conv1(_x)))
        _x = self.pool(F.relu(self.conv2(_x)))
        _x = _x.view(_x.size(0), -1)
        _x = F.relu(self.fc1(_x))
        _x = F.relu(self.fc2(_x))
        _x = self.fc3(_x)
        return _x


input_dir = r'C:\Users\bunny\Desktop\ships-in-satellite-imagery'
print(os.listdir(input_dir))

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "alexnet"

# Number of classes in the dataset
classes = ('no', 'yes')
num_classes = len(classes)

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

data = pd.read_json(os.path.join(input_dir, 'shipsnet.json'))
print(data.head())

img_count = 4000

x = []
for d in data['data'][:img_count]:
    d = np.array(d)
    orig_img = d.reshape((3, 80 * 80)).T.reshape((80, 80, 3))
    resized_img = transform.resize(orig_img, (224, 224), anti_aliasing=False)
    x.append(resized_img)
plt.imshow(x[2])
plt.show()
x = np.transpose(np.array(x), (0, 3, 1, 2))

y = np.array(data['labels'])[:img_count]
print(x.shape)
print(y.shape)

# splitting the data into training ans test sets
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.20)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)  # 0.50)
# Normalizing the data
scalar = MinMaxScaler()
scalar.fit(x_train.reshape(x_train.shape[0], -1))

x_train = scalar.transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
x_val = scalar.transform(x_val.reshape(x_val.shape[0], -1)).reshape(x_val.shape)
x_test = scalar.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print_freq = 20
    print('epoch [%d] loss: %.3f' % (epoch + 1, running_loss / (img_count / batch_size)))
    whole_model_PATH = f'./model_ship_{epoch+1}.pth'
    torch.save(net, whole_model_PATH)

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

images = images.to(device, dtype=torch.float)
labels = labels.to(device, dtype=torch.long)

# print images

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

net = torch.load(f'./model_ship_{num_epochs}.pth')
net.eval()

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.long)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 400 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

savepath = r'C:\Users\bunny\Desktop\ship_test'
for i in [0, 1]:
    for j in [0, 1]:
        make_dir(r'{}\{}{}'.format(savepath, i, j))

with torch.no_grad():
    count = 0
    for data in testloader:
        images, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.long)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        # save error images
        for i in range(len(predicted.cpu().numpy())):
            pred = predicted.cpu().numpy()[i]
            io.imsave(r'{0}\{1}{2}\{1}{2}_{3}.png'.format(savepath, pred, labels.cpu().numpy()[i], '%s' % count),
                      np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
            count += 1

for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
