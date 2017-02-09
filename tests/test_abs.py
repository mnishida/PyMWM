#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc


class Friend(metaclass=abc.ABCMeta):

    def __init__(self, name):
        self.name = self.uppercase(name)

    @classmethod
    @abc.abstractmethod
    def uppercase(cls, name: str):
        pass

    def explain(self):
        print(self.name)

    @abc.abstractmethod
    def hello(self):
        pass


class BoyFriend(Friend):

    def uppercase(cls, name: str):
        return name.upper()

    def hello(self):
        print("Hello, Mr. {}".format(self.uppercase(self.name)))



if __name__ == '__main__':
    bob = BoyFriend('Bob')
    bob.hello()

