import os


def FileReader(filename):
    file = open(filename)

    data = []

    while True:

        eachline = file.readline()
        if len(eachline) == 0:
            break
        data.append(eachline)

    file.close()

    return data


def FileWriter(filename, data, style='w'):
    file = open(filename, style)

    for i in range(len(data)):
        file.write(data[i] + '\n')

    file.close()
    return