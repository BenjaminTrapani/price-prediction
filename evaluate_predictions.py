def shouldBuy(probabilities, buyThreshold):
    leftWeight = 0
    rightWeight = 0
    for index in range(len(probabilities)):
        curVal = probabilities[index]
        if index < len(probabilities) / 2:
            leftWeight = leftWeight + curVal
        else:
            rightWeight = rightWeight + curVal

    if leftWeight == 0:
        return rightWeight > 0

    weightFrac = rightWeight / leftWeight
    return weightFrac > buyThreshold

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
