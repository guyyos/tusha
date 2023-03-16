


COUNTER_FACTUAL_NUM_POINTS = 100
COUNTER_FACTUAL_SMOOTH_NUM_POINTS = 55  # note that COUNTER_FACTUAL_SMOOTH_NUM_POINTS<COUNTER_FACTUAL_NUM_POINTS

class PredictionSummary:
    def __init__(self,predictor,target_pred,mu_pred,target_hdi,mu_hdi,mu_mean,predictor_vals,cat_codes):
        self.predictor = predictor
        self.target_pred = target_pred
        self.mu_pred = mu_pred
        self.target_hdi = target_hdi
        self.mu_hdi = mu_hdi
        self.mu_mean = mu_mean
        self.predictor_vals = predictor_vals
        self.cat_codes = cat_codes
