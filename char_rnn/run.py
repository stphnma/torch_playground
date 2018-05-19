from __future__ import unicode_literals, print_function, division
import torch
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn


## Turning names into Tensors
def letterToIndex(letter):
    return all_letters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line),1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExample(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn, criterion, category_tensor, line_tensor):

    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":

    from extract import extract
    all_letters, n_letters, category_lines, all_categories = extract("../names/names/*.txt")
    n_categories = len(all_categories)
    n_hidden = 128

    from model import RNN
    rnn = RNN(n_letters, n_hidden, n_categories)
    criterion = nn.NLLLoss()
    learning_rate = 0.005

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for i in range(1, n_iters+1):

        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output, loss = train(rnn, criterion, category_tensor, line_tensor)
        current_loss += loss

        if i % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if i % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()
