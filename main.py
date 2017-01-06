import tensorflow as tf
import simulation_params
import security_technicals
import stock_lstm
import build_input
import evaluate_predictions
import numpy as np
from datetime import datetime, timedelta


def run_epoch(session, m, input_technicals, input_params, eval_op, verbose=False, last_state=None):
        costs = 0.0
        iters = 0
        state = last_state
        concatenated_results = None

        for step, (x, y) in enumerate(build_input.get_iterators(input_technicals, input_params)):
            captured_params = {m.input_data: x, m.targets: y}
            if state is not None:
                captured_params[m.initial_state] = state

            cost, state, targets, inputs, output, _ = session.run([m.cost, m.final_state, m.targets, m.input_data, m.output, eval_op], captured_params)
            costs += cost
            iters += m.num_steps

            if concatenated_results is None:
                concatenated_results = output
            else:
                concatenated_results = np.concatenate((concatenated_results, output))

            if verbose:
                print 'input: ' + str(inputs)
                print 'output: ' + str(output)
                print 'target: ' + str(targets)

        return costs / (iters if iters > 0 else 1), state, concatenated_results


def main(_):
    np.set_printoptions(threshold=np.nan)

    beginDate = '2006-04-10'
    endDate = '2014-01-01'
    params = simulation_params.SimulationParams(startDate=beginDate, endDate=endDate,
                                                securityTicker='USO',
                                                movingAveragePeriods=[20, 50, 100],
                                                batchSize=28,
                                                hiddenSize=20,
                                                numLayers=2,
                                                keepProb=0.9,
                                                maxGradNorm=1,
                                                numEpochs=3,
                                                lrDecay=0.97,
                                                learningRate=0.08,
                                                initScale=0.5,
                                                priceChangeScale=1.0,
                                                windowSizeInPrices=30,
                                                daysIntoTheFuture=15,
                                                numPredictionDays=100)

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

            cached_state_between_epochs = None
            for epoch in xrange(params.numEpochs):
                lr_decay = params.lrDecay ** epoch
                print 'lr_decay = ' + str(lr_decay)
                m.assign_lr(session, params.learningRate * lr_decay)
                cur_lr = session.run(m.lr)

                mse, cached_state_between_epochs, _ = run_epoch(session, m, technicals, params, m.train_op,
                                                                verbose=False, last_state=cached_state_between_epochs)
                m.is_training = False
                vmse, _, _ = run_epoch(session, m, technicals, params, tf.no_op(),
                                       last_state=cached_state_between_epochs)
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
            tmse, _, outputs = run_epoch(session, m, technicals, params, tf.no_op(), verbose=True,
                                         last_state=cached_state_between_epochs)

            print("Test mse: %.3f" % tmse)
            initial_value = 1000
            outputs_for_unknown = outputs[len(outputs) - params.numPredictionDays:]
            final_value = evaluate_predictions.fetch_final_value(initial_value, outputs_for_unknown,
                                                                technicals.getUsefulClosePrices(),
                                                                params.windowSizeInPrices,
                                                                params.daysIntoTheFuture, float(numSteps) / 2)
            print 'Final value of %f initial is %f' % (initial_value, final_value)
            first_predicted_price = usefulPrices[len(usefulPrices) - params.numPredictionDays - 1]
            naive_final_value = (usefulPrices[-1] / first_predicted_price) * initial_value
            print 'Naive final value is %f' % naive_final_value

    return 0

if __name__ == "__main__":
    tf.app.run()