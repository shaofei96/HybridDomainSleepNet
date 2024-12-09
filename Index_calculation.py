#coding=utf-8
import numpy as np

class testclass:

    def acc(self,x_lable, y_lable):
        b = []
        for i in range(len(x_lable)):
            if x_lable[i] == y_lable[i]:
                b.append(1)
        train_acc = len(b)
        return train_acc

    def len(self,X, batch_size):
        if(X%batch_size == 0):
            len = X
        else:
            len = (X - (X % batch_size))

        return len

    def nums(self,data):
        nums = 0
        for i in range(len(data)):
            for j in range(len(data[0])):
                if(data[i][j] <= 0.35):
                    nums = nums+1
        return nums