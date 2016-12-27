def shouldBuy(probabilities, buyThreshold):
    totalIndices = 0.0
    totalWeight = 0.0

    for step, prob in enumerate(probabilities):
        if prob > 0:
            totalIndices += prob * step
            totalWeight += prob

    averageIndex = totalIndices / totalWeight
    return averageIndex > buyThreshold

def fetch_final_value(initialCash, probabilities, prices, buyThreshold):
    lastDayPrice = prices[0]
    for yIndex in range(len(probabilities)):
        priceChange = prices[yIndex] - lastDayPrice
        if shouldBuy(probabilities[yIndex], buyThreshold):
            print 'buy: ' + str(probabilities[yIndex]) + ', price change: ' + str(priceChange)
            oldInitialCash = initialCash
            priceFrac = prices[yIndex] / lastDayPrice
            initialCash = initialCash * priceFrac
            print 'initial cash change %d to %d' % (oldInitialCash, initialCash)
        else:
            print 'sell: ' + str(probabilities[yIndex]) + ', price change: ' + str(priceChange)
        lastDayPrice = prices[yIndex]


    return initialCash

