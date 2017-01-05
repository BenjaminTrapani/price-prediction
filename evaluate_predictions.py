def shouldBuy(probabilities, buyThreshold):
    totalIndices = 0.0
    totalWeight = 0.0

    for step, prob in enumerate(probabilities):
        if prob > 0:
            totalIndices += prob * step
            totalWeight += prob

    averageIndex = totalIndices / totalWeight
    return averageIndex > buyThreshold

def fetch_final_value(initialCash, probabilities, prices, daysIntoFuture, buyThreshold):
    totalYield = 0
    buyCount = 0
    for yIndex in range(len(probabilities)):
        lastDayPrice = prices[yIndex]
        futurePrice = prices[yIndex + daysIntoFuture]
        priceChange = futurePrice - lastDayPrice
        if shouldBuy(probabilities[yIndex], buyThreshold):
            print 'buy: ' + str(probabilities[yIndex]) + ', price change: ' + str(priceChange)
            priceFrac = futurePrice / lastDayPrice
            totalYield = totalYield + priceFrac
            buyCount = buyCount + 1
        else:
            print 'sell: ' + str(probabilities[yIndex]) + ', price change: ' + str(priceChange)

    averageYield = totalYield / float(buyCount)
    return initialCash * averageYield

