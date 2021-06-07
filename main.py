import sys
import argparse
from torch import nn, optim
import torch
import matplotlib.pyplot as plt

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr= float(args.lr))
        epochs=5
        print_every = 40
        running_loss = 0
        steps = 0
        tLoss= []




        for e in range(epochs):
            # Model in training mode, dropout is on
            model.train()
            for images, labels in train_set:
                steps += 1



                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        accuracy = 0
                        test_loss = 0
                        for images, labels in train_set:


                            output = model.forward(images)
                            test_loss += criterion(output, labels).item()

                            ## Calculating the accuracy
                            # Model's output is log-softmax, take exponential to get the probabilities
                            ps = torch.exp(output)
                            # Class with highest probability is our predicted class, compare with true label
                            equality = (labels.data == ps.max(1)[1])
                            # Accuracy is number of correct predictions divided by all predictions, just take the mean
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss / len(train_set)),
                          "Test Accuracy: {:.3f}".format(accuracy / len(train_set)))
                    tLoss.append(running_loss / steps)
                    running_loss = 0

                    # Make sure dropout and grads are on for training
                    model.train()

        torch.save(model.state_dict(), 'checkpoint.pth')

        plt.plot(tLoss, label='Training Loss')

        plt.legend()
        plt.show()




        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Evaluating arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = MyAwesomeModel()
            model.load_state_dict(torch.load(args.load_model_from))
            model.eval()

            accuracy = 0


            _, test_set = mnist()

           # with torch.no_grad():
            for itr, (images, labels) in enumerate(test_set):
                output = model.forward(images)

                ps = torch.exp(output)

                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            print("Test Accuracy: {:.3f}".format(accuracy / len(test_set)))

        else:
            print("There is no model to evaluate please train first...")




if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    