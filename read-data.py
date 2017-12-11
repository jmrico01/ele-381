import csv
import datetime
import matplotlib.pyplot as plt

targetCountries = [
    "Guinea",
    "Sierra Leone",
    "Liberia",
    "Mali"
]

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

useCategories = [
    SUSPECTED,
    PROBABLE,
    CONFIRMED
]

data = {
    # "country": [ time series ]
}

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


with open("ebola_data_db_format.csv", "r") as csvFile:
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
    #print("Maximum index: " + str((maxDate - minDate).days))
    timeSize = maxDate - minDate
    timeSize = timeSize.days + 1

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
            dataCategorized[country] = [[0] * timeSize,
                [0] * timeSize,
                [0] * timeSize,
                [0] * timeSize]

        date = ParseDate(row[2])
        number = int(float(row[3]))
        dayIndex = date - minDate
        dayIndex = dayIndex.days
        dataCategorized[country][category][dayIndex] = number

for country, countryData in dataCategorized.items():
    for categoryData in countryData:
        lastNonZero = -1
        for i in range(len(categoryData)):
            if categoryData[i] == 0:
                if lastNonZero != -1:
                    categoryData[i] = lastNonZero
            else:
                lastNonZero = categoryData[i]

for country, countryData in dataCategorized.items():
    pass