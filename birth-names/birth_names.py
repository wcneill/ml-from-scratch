import tkinter as tk
import os
import csv
import numpy as np


def parseFileYear(file_name):
    """

    :param file_name: file name in the format yobXXXX.csv where XXXX is the four digit year.
    :return: the year
    """
    return file_name[3:7]


def printNames():
    """
    Get and print all names from CSV file,
    :return:
    """

    # start the engine, withdraw root window because we don't need a GUI
    root = tk.Tk()
    root.withdraw()

    file_path = tk.filedialog.askopenfilename()
    with open(file_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            print(row[0])


def totalScore(path):
    """
    Reads the CSV file at location 'path' and prints out the number of boys and girls born
    along with the number of unique boy names and unique girl names.

    :param path: The CSV file containing birth name data
    :return:
    """
    boys = 0
    girls = 0
    boyNames = 0
    girlNames = 0

    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if row[1] == "M":
                boys += int(row[2])
                boyNames += 1
            else:
                girls += int(row[2])
                girlNames += 1

    totalNames = boyNames + girlNames
    print("{} boys were born with {} unique names".format(boys, boyNames))
    print("{} girls were born with {} unique names".format(girls, girlNames))
    print("Total unique names: ", totalNames, '\n')


def getID(year, rank, gender):
    """
    Returns the name with the given year, rank and gender. If there is a tie between two names, the first
    instance is returned.

    Finds the name
    :param year:
    :param rank:
    :param gender:
    :return:
    """

    file = "yob{}.csv".format(str(year))
    current_path = os.path.dirname(__file__)
    file_path = current_path + "/us_babynames_by_year/{}".format(file)
    name = "NO NAME"

    curr_rank = 0
    last_count = 0

    with open(file_path, 'r') as csvFile:
        for row in csv.reader(csvFile):
            if row[1] == gender:
                curr_count = row[2]
                if not curr_count == last_count:
                    curr_rank += 1
                    last_count = curr_count

                if curr_rank == rank:
                    name = row[0]
                    break
    return name


def getScore(year, name, gender):
    file_name = "yob{}.csv".format(year)
    curr_path = os.path.dirname(__file__)
    abs_path = curr_path + "/us_babynames_by_year/{}".format(file_name)

    with open(abs_path, 'r') as CSVdat:
        for row in csv.reader(CSVdat):
            if row[1] == gender and row[0] == name:
                return row[2]

    return -1


def getRank(year, name, gender):
    """
    This method takes a year, name and gender and returns the rank of that name in the given year. If the name does not
    exist, the method returns -1
    :param year:
    :param name:
    :param gender:
    :return:
    """
    file_name = "yob{}.csv".format(year)
    curr_path = os.path.dirname(__file__)
    abs_path = curr_path + "/us_babynames_by_year/{}".format(file_name)

    last_count = 0
    rank = 0
    name_found = False
    with open(abs_path, 'r') as CSVdat:
        for row in csv.reader(CSVdat):
            if row[1] == gender:
                name_count = int(row[2])

                if name_count == last_count:
                    last_count = name_count
                else:
                    rank += 1
                    last_count = name_count
                if row[0] == name:
                    name_found = True
                    break
    if not name_found:
        return -1
    return rank


def getNewID(year, new_year, name, gender):
    """
    This method takes a name and a year, and gets the popularity of that name in that year. It then gets the name
    with equal popularity in a different year and returns that name. Returns "NO NAME" if no match in popularity is
    found. This edge case typically occurs if, for example, the name "Joe" comes in 5th place in one year, but there is
    no 5th place in the following year because of multiple ties.

    :param year:
    :param new_year:
    :param name:
    :param gender:
    :return:
    """
    rank = getRank(year, name, gender)
    new_name = getID(new_year, rank, gender)
    return new_name


def getAverageRank(name, gender, select=True, filez=None):
    """
    return the average rank of a given name over the selected files. If using
    this method in conjunction with others that are analyzing multiple CSV
    files, it is prudent to use select=False, and pass an iterable containing
    the absolute paths of the files being looked at. This will save you from
    dealing with multiple pop-up file selection dialogues.

    :param name:
    :param gender:
    :param select: default True. If True, pop up file selection dialogue.
    If false, the files argument must not be empty.
    :param files: default None. An iterable containing the absolute paths
    to the CSV files being analyzed. This parameter is required if param
    select is False.
    :return:
    """
    if select:
        files = getPaths()
    else:
        files = filez

    name_found = False
    rank_sum = 0
    count = len(files)
    for file in files:
        file_name = os.path.basename(file)
        year = parseFileYear(file_name)
        rank = getRank(year, name, gender)
        if rank != -1:
            name_found = True
            rank_sum += rank

    if name_found:
        return rank_sum / count
    return -1


def yearOfHighestRank(name, gender, select=True, filez=None):
    """
    This method finds the year of highest popularity of a given name. If there is a tie over multiple years,
    the method returns the first year at which the highest popularity occurred. If the name is not found in any
    of the selected years, -1 is returned.

    :param name:
    :param gender:
    :return:
    """
    if select:
        files = getPaths()
    else:
        files = filez

    highest_rank = None
    highest_year = None

    for file in files:
        file_name = os.path.basename(file)
        year = parseFileYear(file_name)
        curr_rank = getRank(year, name, gender)
        if highest_rank is None:
            highest_rank = curr_rank
            highest_year = year
        elif curr_rank != -1 and curr_rank < highest_rank:
            highest_rank = curr_rank
            highest_year = year

    if highest_rank == -1:
        return -1
    return highest_year


def birthsRankedHigher(year, name, gender):
    """
    Returns the total number of births with names ranked higher than the one given.

    :param year:
    :param name:
    :param gender:
    :return:
    """
    curr_dir = os.path.dirname(__file__)
    file_path = curr_dir + "/us_babynames_by_year/yob{}.csv".format(year)

    rank = getRank(year, name, gender)
    births = 0
    with open(file_path) as CSVdat:
        for row in csv.reader(CSVdat):
            if row[1] == gender:
                curr_name = row[0]
                curr_rank = getRank(year, curr_name, gender)
                if curr_rank != -1 and curr_rank < rank:
                    births += int(row[2])
                else:
                    break

    return births

def getAllRanks(name, gender, files):
    """
    Returns a numpy 2-D array containing the year and rank pairs. Useful for plotting visualizations

    :param name:
    :param gender:
    :param files:
    :return:
    """
    dim = len(files)
    year_data = np.zeros(dim, dtype=int)
    rank_data = np.zeros(dim, dtype=int)
    for i, file in enumerate(files):
        file_name = os.path.basename(file)
        year = parseFileYear(file_name)
        rank = getRank(year, name, gender)
        if rank == -1: # clean -1 from data (case of name not found in file)
            rank = 0
        year_data[i] = year
        rank_data[i] = rank

    return year_data, rank_data


def getPaths():
    root = tk.Tk()
    root.withdraw()
    return tk.filedialog.askopenfilenames()


def getAllScores(name, gender, files):
    dim = len(files)
    year_data = np.zeros(dim, dtype=int)
    score_data = np.zeros(dim, dtype=int)
    for i, file in enumerate(files):
        year = parseFileYear(os.path.basename(file))
        score = getScore(year, name, gender)
        year_data[i] = year
        score_data[i] = score

    return year_data, score_data
