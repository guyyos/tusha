# -*- coding: utf-8 -*-

from .context import sample

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True


    def def test_create_num_to_features_model():
        num_items = 100
        df = pd.DataFrame(np.random.randint(0,4,size=(num_items, 4)), columns=list('ABCD'))
        df['A'] = np.random.randint(0,3,size=(num_items, 3))
        df['C'] = np.random.randint(0,3,size=(num_items, 5))
        df['target']=np.random.randn(1,num_items)[0]
        df['X']=np.random.randn(1,num_items)[0]
        df['Y']=np.random.randn(1,num_items)[0]
        df['W']=np.random.randn(1,num_items)[0]
        df['Z']=np.random.randn(1,num_items)[0]

        target = 'target'
        categorical_predictors = ['A','B','C','D']#['A','B'] #list('ABCD')
        numerical_predictors = ['X','Y','W','Z']

        cat_num_map = {'A':['X'],'B':['Y']}

        cat_factors = {col:factorize(df[col]) for col in categorical_predictors}

        model,idata = create_num_to_features_model(df,target,categorical_predictors,numerical_predictors,cat_num_map,cat_factors)

        post = az.extract(idata)
        plot = pm.model_to_graphviz(model)

        return model,idata,post,plot



if __name__ == '__main__':
    unittest.main()