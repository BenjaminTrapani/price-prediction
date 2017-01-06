def should_buy(probabilities, buyThreshold):
    totalIndices = 0.0
    totalWeight = 0.0

    for step, prob in enumerate(probabilities):
        if prob > 0:
            totalIndices += prob * step
            totalWeight += prob

    averageIndex = totalIndices / totalWeight
    return averageIndex > buyThreshold


def fetch_final_value(initialCash, probabilities, prices, window_size, daysIntoFuture, buyThreshold):
    totalYield = 0
    buyCount = 0
    for yIndex in range(len(probabilities)):
        lastDayPrice = prices[yIndex + window_size - 1]
        futurePrice = prices[yIndex + daysIntoFuture + window_size - 1]
        priceChange = futurePrice - lastDayPrice
        if should_buy(probabilities[yIndex], buyThreshold):
            print 'buy: ' + str(probabilities[yIndex]) + ', price change: ' + str(priceChange)
            priceFrac = futurePrice / lastDayPrice
            totalYield = totalYield + priceFrac
            buyCount = buyCount + 1
        else:
            print 'sell: ' + str(probabilities[yIndex]) + ', price change: ' + str(priceChange)

    if buyCount == 0:
        average_yield = 1
    else:
        average_yield = totalYield / float(buyCount)

    return initialCash * average_yield

