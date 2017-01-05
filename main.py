import tensorflow as tf
import simulation_params
import security_technicals
import stock_lstm
import build_input
import evaluate_predictions
import numpy as np
from datetime import datetime, timedelta

def run_epoch(session, m, inputTechnicals, inputParams, eval_op, verbose=False, lastState=None):
        costs = 0.0
        iters = 0
        state = lastState
        concatenatedResults = None

        for step, (x, y) in enumerate(build_input.get_iterators(inputTechnicals, inputParams)):
            capturedParams = {m.input_data: x, m.targets: y}
            if state is not None:
                capturedParams[m.initial_state] = state

            cost, state, targets, input, output, _ = session.run([m.cost, m.final_state, m.targets, m.input_data, m.output, eval_op], capturedParams)
            costs += cost
            iters += m.num_steps

            if concatenatedResults is None:
                concatenatedResults = output
            else:
                concatenatedResults = np.concatenate((concatenatedResults, output))

            if verbose:
                print 'input: ' + str(input)
                print 'output: ' + str(output)
                print 'target: ' + str(targets)

        return costs / (iters if iters > 0 else 1), state, concatenatedResults


def main(_):
    np.set_printoptions(threshold=np.nan)

    beginDate = '2010-01-01'
    endDate = '2014-01-01'
    params = simulation_params.SimulationParams(startDate=beginDate, endDate=endDate,
                                                securityTicker='AMD',
                                                movingAveragePeriods=[20, 50, 100],
                                                batchSize=28,
                                                hiddenSize=20,
                                                numLayers=2,
                                                keepProb=0.9,
                                                maxGradNorm=1,
                                                numEpochs=1,
                                                lrDecay=0.97,
                                                learningRate=0.08,
                                                initScale=0.5,
                                                priceChangeScale=1.0,
                                                windowSizeInPrices=15,
                                                daysIntoTheFuture=30,
                                                numPredictionDays=600)


    technicals = security_technicals.load_or_fetch_technicals(params)

    with tf.Graph().as_default():
        with tf.Session() as session:
            initializer = tf.random_uniform_initializer(-params.initScale, params.initScale)
            inputs, targets, _, numSteps, batch_size = build_input.get_inputs_targets_epochsize_numsteps(technicals, params)
            print 'Raw inputs: ' + str(inputs)
            print 'Raw targets: ' + str(targets)

            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = stock_lstm.StockLSTM(is_training=True, simulationParams=params, num_steps=numSteps, batch_size=batch_size)

            tf.global_variables_initializer().run()

            cachedStateBetweenEpochs = None
            for epoch in xrange(params.numEpochs):
                lr_decay = params.lrDecay ** epoch
                print 'lr_decay = ' + str(lr_decay)
                m.assign_lr(session, params.learningRate * lr_decay)
                cur_lr = session.run(m.lr)

                mse, cachedStateBetweenEpochs, _ = run_epoch(session, m, technicals, params, m.train_op, verbose=False, lastState=cachedStateBetweenEpochs)
                m.is_training = False
                vmse, _, _ = run_epoch(session, m, technicals, params, tf.no_op(), lastState=cachedStateBetweenEpochs)
                m.is_training = True
                print("Epoch: %d - learning rate: %.3f - train mse: %.3f - test mse: %.3f" %
                      (epoch, cur_lr, mse, vmse))

            startDatetime = datetime.strptime(params.startDate, "%Y-%m-%d")
            startDatetime += timedelta(days=params.numPredictionDays)
            endDatetime = datetime.strptime(params.endDate, "%Y-%m-%d")
            endDatetime += timedelta(days=params.numPredictionDays)
            params.endDate = endDatetime.strftime("%Y-%m-%d")
            params.startDate = startDatetime.strftime("%Y-%m-%d")
            if startDatetime > datetime.now():
                params.daysIntoTheFuture = 0

            technicals = security_technicals.load_or_fetch_technicals(params)
            usefulPrices = technicals.getUsefulClosePrices()
            print('Close prices used in prediction = ' + str(usefulPrices))
            print('Useful close price len = %d' % len(usefulPrices))
            m.is_training = False
            tmse, _, outputs = run_epoch(session, m, technicals, params, tf.no_op(), verbose=True, lastState=cachedStateBetweenEpochs)

            print("Test mse: %.3f" % tmse)
            print("Number of output predictions = %d" % len(outputs))

            initialValue = 1000
            outputsForUnknown = outputs[len(outputs) - params.numPredictionDays:]
            finalValue = evaluate_predictions.fetch_final_value(initialValue, outputsForUnknown,
                                                                technicals.getUsefulClosePrices(),
                                                                params.daysIntoTheFuture, float(numSteps) / 2)
            print 'Final value of %f initial is %f' % (initialValue, finalValue)
            naiveFinalValue = (usefulPrices[-1] / usefulPrices[0]) * initialValue
            print 'Naive final value is %f' % (naiveFinalValue)

    return 0

if __name__ == "__main__":
    tf.app.run()