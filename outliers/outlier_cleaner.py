#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here
    import operator as operator
    stats = [(ages[i], net_worths[i], (predictions[i] - net_worths[i])**2) for i in xrange(len(predictions))]
    stats = sorted(stats, key=operator.itemgetter(2))
    num_clean_elems = int(len(predictions) * 0.9)
    return stats[0:num_clean_elems]

