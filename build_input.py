import numpy as np
import targets


def _add_details_for_price(technicals, closePriceIndex):
    row = np.array([], dtype=np.float32)

    smamap, emamap = technicals.getMovingAverages()
    movingAveragePeriods = technicals.movingAveragePeriods
    usefulClosePrices = technicals.getUsefulClosePrices()
    volumes = technicals.getVolumes()

    row = np.append(row, usefulClosePrices[closePriceIndex])
    row = np.append(row, volumes[closePriceIndex])
    for movingAveragePeriod in movingAveragePeriods:
        row = np.append(row, smamap[movingAveragePeriod][closePriceIndex])
        row = np.append(row, emamap[movingAveragePeriod][closePriceIndex])

    return row


# returns a 2D array of inputs: [len(usefulClosePrices), num_steps]
def _fetch_raw_input(technicals, daysIntoTheFuture):
    result = None
    usefulClosePriceLen = len(technicals.getUsefulClosePrices())
    for closePriceIndex in range(usefulClosePriceLen - daysIntoTheFuture):
        row = _add_details_for_price(technicals, closePriceIndex)
        if result is None:
            result = np.empty(shape=[0, len(row)], dtype=np.float32)

        result = np.append(result, row.reshape([1, len(row)]), axis=0)

    return result


class InvalidEpochSize(Exception):
    pass


def get_inputs_targets_epochsize_numsteps(technicals, simulationParams):
    input_data = _fetch_raw_input(technicals, simulationParams.daysIntoTheFuture)

    if len(input_data) == 0:
        raise ValueError("len(npInput) = 0, input building failed")

    num_steps = len(input_data[0]) * simulationParams.windowSizeInPrices
    batch_size = simulationParams.batchSize
    data_len = input_data.size
    batch_len = (data_len // (batch_size * num_steps)) * num_steps

    print "data_len = %d" % data_len
    print "batch_size = %d" % batch_size
    print "batch_len = %d" % batch_len
    print "num_steps = %d" % num_steps

    epoch_size = (batch_len - num_steps - 1) // len(input_data[0])
    print "epoch_size = %d" % (epoch_size)
    if epoch_size <= 0:
        raise InvalidEpochSize("epoch_size <= 0, decrease batch_size or num_steps")

    def batch_data(raw_data, i_technicals_per_price):
        temp_batch_len = (len(raw_data) // (batch_size * i_technicals_per_price)) * i_technicals_per_price
        print 'temp_batch_len = %d' % temp_batch_len
        data = np.zeros([batch_size, temp_batch_len], dtype=np.float32)
        for i in range(batch_size):
            data[i] = raw_data[temp_batch_len * i:temp_batch_len * (i + 1)]
        return data

    target_array = targets.build_targets(num_steps, simulationParams.priceChangeScale,
                                         simulationParams.daysIntoTheFuture, simulationParams.windowSizeInPrices,
                                         technicals)
    batched_targets = batch_data(target_array, 1)

    reshaped_input = input_data.reshape([-1])
    batched_input = batch_data(reshaped_input, num_steps)

    return batched_input, batched_targets, epoch_size, num_steps, batch_size


def check_inputs_targets(inputs, target_chunk, offset, num_steps, days_into_future, close_prices, window_size):
    print '---'
    for index, input in enumerate(inputs):
        if index is not 0:
            return
        target = target_chunk[index]
        initial_price = input[len(input) - 8]
        expected_final_price = (target * 2.0 / num_steps) * initial_price
        actual_final_price = close_prices[offset + days_into_future + window_size - 1] #close price index incorrect, fix.
        tolerated_diff = (initial_price / num_steps) * 2.0
        if abs(expected_final_price - actual_final_price) > tolerated_diff:
            print 'expected final price %f to be within %f of actual %f' % (expected_final_price, tolerated_diff,
                                                                            actual_final_price)


#returns two iterators: input, target
def get_iterators(technicals, simulationParams):
    input_data, batched_targets, epoch_size, num_steps, _ = get_inputs_targets_epochsize_numsteps(technicals, simulationParams)

    advance_per_chunk = num_steps / simulationParams.windowSizeInPrices
    cur_index = 0

    for i in range(epoch_size):
        x_begin_index = cur_index
        x_end_index = x_begin_index + num_steps
        inputs = input_data[:, x_begin_index:x_end_index]
        target_chunk = batched_targets[:, i]

        check_inputs_targets(inputs, target_chunk, i, num_steps, simulationParams.daysIntoTheFuture,
                             technicals.getUsefulClosePrices(), simulationParams.windowSizeInPrices)

        #print 'epoch %d' % i
        #print 'inputs ' + str(inputs)
        #print 'outputs ' + str(target_chunk)

        cur_index += advance_per_chunk

        yield (inputs, target_chunk)
