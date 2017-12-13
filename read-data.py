import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

targetCountries = [
    "Sierra Leone",
    #"Guinea",
    #"Liberia"
]

# Takes a date string, returns a datetime.date object
# Returns None if invalid date string
def ParseDate(dateStr):
    dateStrArray = dateStr.split("-")
    if len(dateStrArray) != 3:
        return None

    year = int(dateStrArray[0])
    month = int(dateStrArray[1])
    day = int(dateStrArray[2])
    return datetime.date(year, month, day)

def ReadData(dataFilePath, targetCountries, plot = False):
    N_TIMESERIES = -1
    data = {
        # "country": [ time series ]
    }

    SUSPECTED = 0
    PROBABLE = 1
    CONFIRMED = 2
    suspectedStr = "Cumulative number of suspected Ebola cases"
    probableStr = "Cumulative number of probable Ebola cases"
    confirmedStr = "Cumulative number of confirmed Ebola cases"
    dataCategorized = {
        # sample entry:
        # "country": [
        #   [ time series for suspected ],
        #   [ for probable ],
        #   [ for confirmed ]
        # ]
    }

    with open(dataFilePath, "r") as csvFile:
        reader = csv.reader(csvFile)
        rows = []
        minDate = datetime.date.max
        maxDate = datetime.date.min
        for row in reader:
            date = ParseDate(row[2])
            if date == None:
                continue

            rows.append(row)
            if date < minDate:
                minDate = date
            if date > maxDate:
                maxDate = date
        
        #print("Minimum date: " + str(minDate))
        #print("Maximum date: " + str(maxDate))
        print("Start date: " + str(minDate))
        print("End date:   " + str(maxDate))
        #print("Maximum index: " + str((maxDate - minDate).days))
        maxDelta = maxDate - minDate
        N_TIMESERIES = maxDelta.days + 1

        for row in rows:
            category = -1
            if row[0] == suspectedStr:
                category = SUSPECTED
            elif row[0] == probableStr:
                category = PROBABLE
            elif row[0] == confirmedStr:
                category = CONFIRMED

            if category == -1:
                continue

            country = row[1]
            if country not in targetCountries:
                continue

            if country not in dataCategorized:
                dataCategorized[country] = [
                    [-1] * N_TIMESERIES,
                    [-1] * N_TIMESERIES,
                    [-1] * N_TIMESERIES
                ]

            date = ParseDate(row[2])
            number = int(float(row[3]))
            dayIndex = date - minDate
            dayIndex = dayIndex.days
            dataCategorized[country][category][dayIndex] = number

    for country, countryData in dataCategorized.items():
        for categoryData in countryData:
            lastValue = -1
            for i in range(len(categoryData)):
                if categoryData[i] == -1:
                    if lastValue != -1:
                        categoryData[i] = lastValue
                    else:
                        categoryData[i] = 0
                else:
                    lastValue = categoryData[i]

    for country, countryData in dataCategorized.items():
        data[country] = np.array(countryData[CONFIRMED])
        data[country] -= data[country][0]

    #exit()
    # plotting
    for country, infected in data.items():
        print("Initial infected: " + str(infected[0]))
        window = 10
        infected = np.convolve(infected, np.ones(window,) / window, mode="valid")
        plt.plot(infected)
        
        plt.title(country)
        plt.show()