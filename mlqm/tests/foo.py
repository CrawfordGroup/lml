#!/usr/bin/python3


import atexit

def ext() :
    print("Exiting")

if __name__ == "__main__" :
    atexit.register(ext)
    print("running")
