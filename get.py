#!/usr/bin/env python

if __name__ == "__main__":
    with open("ciao_trusts.txt", "a") as file:
        for i in range(1,17615):
            file.write(str(i) + ',1,0\n')