import pandas as pd

class SampleStore():
    '''
    Wrapper class to work on data sets that have already been sampled.
    '''
    def __init__(self, df: pd.DataFrame, **kwargs):
        self.df = df

    def symmetric_sample(self, columns, size):
        '''
        This sampling strategy extracts pairs of configuratins that differ exactly in the
        set of options specified. Therefore, one can estimate the effect of one or more options
        by quantifying the pair-wise difference with respect to non-functional properties.
        '''
        pass

    def check_uniformity(self, columns):
        '''
        Run Kolmogorov-Smirnov test to do something
        '''
        pass
