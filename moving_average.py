def computeSimpleMovingAverage(dayClosePrices, movingAveragePeriod):
    periodSum = 0.0
    dayIndex = 0
    result = []

    for closePrice in dayClosePrices:
        periodSum += closePrice
        if dayIndex >= movingAveragePeriod:
            periodSum -= dayClosePrices[dayIndex-movingAveragePeriod]
            result.append(periodSum/float(movingAveragePeriod))
        dayIndex += 1

    return result

def computeSimpleAndExponentialMovingAverage(dayClosePrices, movingAveragePeriod):
    sma = computeSimpleMovingAverage(dayClosePrices, movingAveragePeriod)
    multiplier = 2.0 / (float(movingAveragePeriod) + 1.0)
    ema = [sma[0]]
    for dayIndex in range(movingAveragePeriod + 1, len(dayClosePrices), 1):
        prevEMA = ema[dayIndex - 1 - movingAveragePeriod]
        closePrice = dayClosePrices[dayIndex - movingAveragePeriod]
        curEMA = (closePrice - prevEMA) * multiplier + prevEMA
        ema.append(curEMA)

    return sma, ema
