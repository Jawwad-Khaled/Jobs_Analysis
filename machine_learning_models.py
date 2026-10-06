import numpy as np

class Ridge_Regression():
    def __init__(self):
        self.parameters = np.array([])
        self.y_mean = 0
        self.X_mean = 0

    def fit(self, X, y, alpha):
        self.y_mean = np.mean(y)
        self.X_mean = np.mean(X, axis=0)
        X_ = (X - self.X_mean).copy()
        y_ = (y - self.y_mean).reshape(-1, 1).copy()
        self.parameters = np.linalg.inv(X_.T @ X_ + alpha * np.identity((X_.T @ X_).shape[0])) @ X_.T @ y_

    def predict(self, X):
        y_hat = X @ self.parameters + self.y_mean - self.parameters.T @ self.X_mean.reshape(-1, 1).copy()
        y_hat = y_hat.reshape(-1)
        return y_hat

class Logistic_Regression():
    def __init__(self, max_iterations=1000, record_losses=False):
        self.parameters = np.array([])
        self.first_moment_r = np.array([])
        self.second_moment_r = np.array([])
        self.first_moment_ur = 0
        self.second_moment_ur = 0
        self.bias = 0
        self.bias_vector = np.array([])
        self.learning_rate = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.max_iterations = max_iterations
        self.record_losses = record_losses
        self.losses = []
        self.iterations = []

    def fit(self, X, y, alpha):
        X_ = X.copy()
        y_ = y.reshape(-1, 1).copy()
        self.parameters = np.zeros(X_.shape[1]).reshape(-1, 1)
        self.first_moment_r = np.zeros(X_.shape[1]).reshape(-1, 1)
        self.second_moment_r = np.zeros(X_.shape[1]).reshape(-1, 1)
        self.first_moment_ur = 0
        self.second_moment_ur = 0
        self.bias_vector =  np.full(shape=y_.shape[0], fill_value=self.bias).reshape(-1, 1)
        for i in range(0, self.max_iterations):
            previous_parameters = self.parameters.copy()
            p = 1/(1+np.exp(-(X_ @ self.parameters + self.bias_vector)))
            grad_w = -(np.transpose(X_) @ (y_ - p) -2 * X_.shape[0] * alpha * self.parameters) * (1/X_.shape[0])
            grad_b = -(np.mean(y_ - p))
            self.first_moment_r = self.beta_1 * self.first_moment_r + (1-self.beta_1) * grad_w 
            self.second_moment_r = self.beta_2 * self.second_moment_r + (1-self.beta_2) * np.square(grad_w)
            self.first_moment_ur = self.beta_1 * self.first_moment_ur + (1-self.beta_1) * grad_b
            self.second_moment_ur = self.beta_2 * self.second_moment_ur + (1-self.beta_2) * np.square(grad_b)
            m_hat_w = self.first_moment_r / (1 - self.beta_1 ** (i+1))
            v_hat_w = self.second_moment_r / (1 - self.beta_2 ** (i+1))
            m_hat_b = self.first_moment_ur / (1 - self.beta_1 ** (i+1))
            v_hat_b = self.second_moment_ur / (1 - self.beta_2 ** (i+1))
            self.parameters = self.parameters - self.learning_rate * (m_hat_w / (np.sqrt(v_hat_w) + 1e-8))
            self.bias = self.bias - self.learning_rate * (m_hat_b / (np.sqrt(v_hat_b) + 1e-8))
            self.bias_vector = np.full(shape=y_.shape[0], fill_value=self.bias).reshape(-1, 1)
            if self.record_losses:
                loss = (-1/y_.shape[0]) * np.sum(y_ * np.log(p) + (np.ones_like(y_) - y_) * np.log(np.ones_like(p) - p))
                self.losses.append(loss)
                self.iterations.append(i+1)
            if np.max(np.abs(self.parameters - previous_parameters)) < 1e-12:
                break

    def predict(self, X):
        bias_vector_prediction = np.full(shape=X.shape[0], fill_value=self.bias).reshape(-1, 1)
        y_hat = 1/(1+np.exp(-(X @ self.parameters + bias_vector_prediction)))
        y_hat = y_hat.reshape(-1)
        return y_hat    

class Node():
    def __init__(self, feature=None, threshold=0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Regression_Tree():
    def __init__(self, number_of_random_features, minimum_samples_split=250, max_depth=100, features=0, hessian=None):
        self.minimum_samples_split = minimum_samples_split
        self.max_depth = max_depth
        self.features = features
        self.number_of_random_features = number_of_random_features
        self.hessian = hessian
        self.root = None

    def choose_best_split(self, X, y):
        self.features = np.random.choice(X.shape[1], self.number_of_random_features, replace=False)
        choices = []
        for i in self.features:
            sorted_array = np.unique(X[:, i])
            previous_element = np.inf
            midpoints = []
            for element in sorted_array:
                if previous_element == np.inf:
                    previous_element = element
                else:
                    midpoints.append(0.5 * (previous_element + element))
                    previous_element = element
            midpoints = np.array(midpoints)
            if len(midpoints) == 0:
                continue
            sse_values = {}
            for midpoint in midpoints:
                sse = 0
                left_index = np.argwhere(X[:, i] <= midpoint).flatten()
                right_index = np.argwhere(X[:, i] > midpoint).flatten()
                left_mean = np.mean(y[left_index])
                sse += np.sum((left_mean - y[left_index])**2)
                right_mean = np.mean(y[right_index])
                sse += np.sum((right_mean - y[right_index])**2)
                sse_values.update({midpoint:sse})
            best_midpoint = min(sse_values, key=sse_values.get)
            best_sse = sse_values[best_midpoint]
            choices.append([i, best_midpoint, best_sse])
        if not choices:
            return None, None, False
        best_predictor = 0
        best_midpoint = 0
        best_sse = np.inf
        for choice in choices:
            if choice[2] < best_sse:
                best_sse = choice[2]
                best_predictor = choice[0]
                best_midpoint = choice[1]
        return best_predictor, best_midpoint, True
    
    def fit(self, X, y):
        if self.hessian is None:
            self.hessian = np.ones(y.shape)
        self.features = X.shape[1]
        self.root = self.grow_tree(X, y, self.hessian)

    def grow_tree(self, X, y, hessian, depth=0):
        if depth >= self.max_depth or len(y) < self.minimum_samples_split:
            prediction = np.sum(y) / np.sum(hessian)
            return Node(value=prediction)
        best_predictor, best_midpoint, valid_split = self.choose_best_split(X, y)
        if not valid_split:
            prediction = np.sum(y) / np.sum(hessian)
            return Node(value=prediction)
        left_mask = X[:, best_predictor] <= best_midpoint
        right_mask = ~left_mask
        left = self.grow_tree(X[left_mask], y[left_mask], hessian[left_mask], depth+1)
        right = self.grow_tree(X[right_mask], y[right_mask], hessian[right_mask], depth+1)
        return Node(best_predictor, best_midpoint, left, right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

class Classification_Tree():
    def __init__(self, number_of_random_features, minimum_samples_split=250, max_depth=100, features=0):
        self.minimum_samples_split = minimum_samples_split
        self.max_depth = max_depth
        self.features = features
        self.number_of_random_features = number_of_random_features
        self.root = None

    def choose_best_split(self, X, y):
        self.features = np.random.choice(X.shape[1], self.number_of_random_features, replace=False)
        choices = []
        for i in self.features:
            sorted_array = np.unique(X[:, i])
            previous_element = np.inf
            midpoints = []
            for element in sorted_array:
                if previous_element == np.inf:
                    previous_element = element
                else:
                    midpoints.append(0.5 * (previous_element + element))
                    previous_element = element
            midpoints = np.array(midpoints)
            if len(midpoints) == 0:
                continue
            gini_impurities = {}
            for midpoint in midpoints:
                impurity = 0
                left_index = np.argwhere(X[:, i] <= midpoint).flatten()
                right_index = np.argwhere(X[:, i] > midpoint).flatten()
                left_impurity = 1 - (np.sum(y[left_index] == 0) / len(y[left_index])) ** 2 - (np.sum(y[left_index] == 1) / len(y[left_index])) ** 2
                right_impurity = 1 - (np.sum(y[right_index] == 0) / len(y[right_index])) ** 2 - (np.sum(y[right_index] == 1) / len(y[right_index])) ** 2
                impurity = (len(y[left_index]) / len(y)) * left_impurity + (len(y[right_index]) / len(y)) * right_impurity
                gini_impurities.update({midpoint:impurity})
            best_midpoint = min(gini_impurities, key=gini_impurities.get)
            best_impurity = gini_impurities[best_midpoint]
            choices.append([i, best_midpoint, best_impurity])
        if not choices:
            return None, None, False
        best_predictor = 0
        best_midpoint = 0
        best_impurity = np.inf
        for choice in choices:
            if choice[2] < best_impurity:
                best_impurity = choice[2]
                best_predictor = choice[0]
                best_midpoint = choice[1]
        return best_predictor, best_midpoint, True
    
    def fit(self, X, y):
        self.features = X.shape[1]
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.minimum_samples_split or len(np.unique(y)) == 1:
            prediction = np.argmax(np.bincount(y))
            return Node(value=prediction)
        best_predictor, best_midpoint, valid_split = self.choose_best_split(X, y)
        if not valid_split:
            prediction = np.argmax(np.bincount(y))
            return Node(value=prediction)
        left_mask = X[:, best_predictor] <= best_midpoint
        right_mask = ~left_mask
        left = self.grow_tree(X[left_mask], y[left_mask], depth+1)
        right = self.grow_tree(X[right_mask], y[right_mask], depth+1)
        return Node(best_predictor, best_midpoint, left, right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

class Regression_Random_Forest():
    def __init__(self, number_of_trees=100, number_of_random_features=1, max_depth=100, minimum_samples_split=25):
        self.number_of_trees = number_of_trees
        self.number_of_random_features = number_of_random_features
        self.max_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
        self.trees = []

    def fit(self, X, y):
        for i in range(0, self.number_of_trees):
            indices = self.select_samples(X, y)
            regression_tree = Regression_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=self.number_of_random_features)
            regression_tree.fit(X[indices], y[indices])
            self.trees.append(regression_tree)
            
    def select_samples(self, X, y):
        return np.random.choice(y.shape[0], y.shape[0], True)
    
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

class Classification_Random_Forest():
    def __init__(self, number_of_trees=100, number_of_random_features=1, max_depth=100, minimum_samples_split=25):
            self.number_of_trees = number_of_trees
            self.number_of_random_features = number_of_random_features
            self.max_depth = max_depth
            self.minimum_samples_split = minimum_samples_split
            self.trees = []

    def fit(self, X, y):
        for i in range(0, self.number_of_trees):
            indices = self.select_samples(X, y)
            classification_tree = Classification_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=self.number_of_random_features)
            classification_tree.fit(X[indices], y[indices])
            self.trees.append(classification_tree)

    def select_samples(self, X, y):
        return np.random.choice(y.shape[0], y.shape[0], True)
    
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return  np.mean(predictions, axis=0)

class Regression_Gradient_Boosting():
    def __init__(self, number_of_trees=100, max_depth=5, minimum_samples_split=25, learning_rate=0.1):
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
        self.learning_rate = learning_rate
        self.initial_prediction = 0
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(shape=y.shape, fill_value=self.initial_prediction)
        for i in range(0, self.number_of_trees):
            residuals = y - current_prediction
            regression_tree = Regression_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=X.shape[1])
            regression_tree.fit(X, residuals)
            current_prediction = current_prediction + self.learning_rate * regression_tree.predict(X)
            self.trees.append(regression_tree)

    def predict(self, X):
        current_prediction = np.full(shape=X.shape[0], fill_value=self.initial_prediction)
        for tree in self.trees:
            current_prediction = current_prediction + self.learning_rate * tree.predict(X)
        return current_prediction

class Classification_Gradient_Boosting():
    def __init__(self, number_of_trees=100, max_depth=5, minimum_samples_split=25, learning_rate=0.1):
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
        self.learning_rate = learning_rate
        self.initial_prediction = 0
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.log((np.sum(y == 1) + 1e-8) / (np.sum(y == 0) + 1e-8))
        current_prediction = np.full(shape=y.shape, fill_value=self.initial_prediction)
        for i in range(0, self.number_of_trees):
            p = 1 / (1 + np.exp(-current_prediction))
            residuals = y - p
            regression_tree = Regression_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=X.shape[1], hessian=p*(1-p))
            regression_tree.fit(X, residuals)
            current_prediction = current_prediction + self.learning_rate * regression_tree.predict(X)
            self.trees.append(regression_tree)
            
    def predict(self, X):
        current_prediction = np.full(shape=X.shape[0], fill_value=self.initial_prediction)
        for tree in self.trees:
            current_prediction = current_prediction + self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-current_prediction))

class Regression_Neural_Network():
    def __init__(self, epochs=1000, learning_rate=0.001):
        self.w1 = np.array([])
        self.b1 = np.array([])
        self.w2 = np.array([])
        self.b2 = np.array([])
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999

    def fit(self, X, y, hidden_size):
        self.w1 = np.random.randn(X.shape[1], hidden_size) * np.sqrt(2 / X.shape[1])
        self.b1 = np.random.randn(1, hidden_size) * np.sqrt(2 / hidden_size)
        self.w2 = np.random.randn(hidden_size, 1)
        self.b2 = np.random.randn(1, 1)
        first_moment_w1 = np.zeros_like(self.w1)
        second_moment_w1 = np.zeros_like(self.w1)
        first_moment_b1 = np.zeros_like(self.b1)
        second_moment_b1 = np.zeros_like(self.b1)
        first_moment_w2 = np.zeros_like(self.w2)
        second_moment_w2 = np.zeros_like(self.w2)
        first_moment_b2 = np.zeros_like(self.b2)
        second_moment_b2 = np.zeros_like(self.b2)

        for i in range(0, self.epochs):
            z1 = X @ self.w1 + self.b1
            a1 = np.where(z1>0, z1, 0.01*z1)
            z2 = a1 @ self.w2 + self.b2
            a2 = z2
            y_hat = a2
            
            grad_w2 = (np.transpose(a1) @ (-2 * (y-y_hat))) / X.shape[0]
            grad_b2 = (np.sum(-2 * (y-y_hat), axis=0, keepdims=True)) / X.shape[0]
            grad_w1 = (np.transpose(X) @ ((-2 * (y-y_hat)) @ np.transpose(self.w2) * np.where(z1>0, 1, 0.01))) / X.shape[0]
            grad_b1 = (np.sum((-2 * (y-y_hat)) @ np.transpose(self.w2) * np.where(z1>0, 1, 0.01), axis=0, keepdims=True)) / X.shape[0]

            first_moment_w2, second_moment_w2, m_hat_w2, v_hat_w2 = self.adam_optimizing(grad_w2, first_moment_w2, second_moment_w2, i)
            first_moment_b2, second_moment_b2, m_hat_b2, v_hat_b2 = self.adam_optimizing(grad_b2, first_moment_b2, second_moment_b2, i)
            first_moment_w1, second_moment_w1, m_hat_w1, v_hat_w1 = self.adam_optimizing(grad_w1, first_moment_w1, second_moment_w1, i)
            first_moment_b1, second_moment_b1, m_hat_b1, v_hat_b1 = self.adam_optimizing(grad_b1, first_moment_b1, second_moment_b1, i)

            self.w2 = self.w2 - self.learning_rate * (m_hat_w2 / (np.sqrt(v_hat_w2) + 1e-8))
            self.b2 = self.b2 - self.learning_rate * (m_hat_b2 / (np.sqrt(v_hat_b2) + 1e-8))
            self.w1 = self.w1 -self.learning_rate * (m_hat_w1 / (np.sqrt(v_hat_w1) + 1e-8))
            self.b1 = self.b1 - self.learning_rate * (m_hat_b1 / (np.sqrt(v_hat_b1) + 1e-8))

    def adam_optimizing(self, grad, first_moment, second_moment, index):
        first_moment = self.beta_1 * first_moment + (1 - self.beta_1) * grad 
        second_moment = self.beta_2 * second_moment + (1 - self.beta_2) * np.square(grad)
        m_hat = first_moment / (1 - self.beta_1 ** (index+1))
        v_hat = second_moment / (1 - self.beta_2 ** (index+1))
        return first_moment, second_moment, m_hat, v_hat
    
    def predict(self, X):
        z1 = X @ self.w1 + self.b1
        a1 = np.where(z1>0, z1, 0.01*z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = z2
        y_hat = a2
        return y_hat

class Classification_Neural_Network():
    def __init__(self, epochs=1000, learning_rate=0.001):
        self.w1 = np.array([])
        self.b1 = np.array([])
        self.w2 = np.array([])
        self.b2 = np.array([])
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999

    def fit(self, X, y, hidden_size):
        self.w1 = np.random.randn(X.shape[1], hidden_size) * np.sqrt(2 / X.shape[1])
        self.b1 = np.random.randn(1, hidden_size) * np.sqrt(2 / hidden_size)
        self.w2 = np.random.randn(hidden_size, 1)
        self.b2 = np.random.randn(1, 1)
        first_moment_w1 = np.zeros_like(self.w1)
        second_moment_w1 = np.zeros_like(self.w1)
        first_moment_b1 = np.zeros_like(self.b1)
        second_moment_b1 = np.zeros_like(self.b1)
        first_moment_w2 = np.zeros_like(self.w2)
        second_moment_w2 = np.zeros_like(self.w2)
        first_moment_b2 = np.zeros_like(self.b2)
        second_moment_b2 = np.zeros_like(self.b2)

        for i in range(0, self.epochs):
            z1 = X @ self.w1 + self.b1
            a1 = 1 / (1 + np.exp(-z1))
            z2 = a1 @ self.w2 + self.b2
            a2 = 1 / (1 + np.exp(-z2))
            y_hat = a2
            
            grad_w2 = (np.transpose(a1) @ -(y-y_hat)) / X.shape[0] 
            grad_b2 = (np.sum(-(y-y_hat), axis=0, keepdims=True)) / X.shape[0]
            grad_w1 = (np.transpose(X) @ ((-(y-y_hat) @ np.transpose(self.w2)) * a1 * (np.ones_like(a1)-a1))) / X.shape[0]
            grad_b1 = (np.sum(-(y-y_hat) @ np.transpose(self.w2) * a1 * (np.ones_like(a1)-a1), axis=0, keepdims=True)) / X.shape[0]

            first_moment_w2, second_moment_w2, m_hat_w2, v_hat_w2 = self.adam_optimizing(grad_w2, first_moment_w2, second_moment_w2, i)
            first_moment_b2, second_moment_b2, m_hat_b2, v_hat_b2 = self.adam_optimizing(grad_b2, first_moment_b2, second_moment_b2, i)
            first_moment_w1, second_moment_w1, m_hat_w1, v_hat_w1 = self.adam_optimizing(grad_w1, first_moment_w1, second_moment_w1, i)
            first_moment_b1, second_moment_b1, m_hat_b1, v_hat_b1 = self.adam_optimizing(grad_b1, first_moment_b1, second_moment_b1, i)

            self.w2 = self.w2 - self.learning_rate * (m_hat_w2 / (np.sqrt(v_hat_w2) + 1e-8))
            self.b2 = self.b2 - self.learning_rate * (m_hat_b2 / (np.sqrt(v_hat_b2) + 1e-8))
            self.w1 = self.w1 -self.learning_rate * (m_hat_w1 / (np.sqrt(v_hat_w1) + 1e-8))
            self.b1 = self.b1 - self.learning_rate * (m_hat_b1 / (np.sqrt(v_hat_b1) + 1e-8))

    def adam_optimizing(self, grad, first_moment, second_moment, index):
        first_moment = self.beta_1 * first_moment + (1 - self.beta_1) * grad 
        second_moment = self.beta_2 * second_moment + (1 - self.beta_2) * np.square(grad)
        m_hat = first_moment / (1 - self.beta_1 ** (index+1))
        v_hat = second_moment / (1 - self.beta_2 ** (index+1))
        return first_moment, second_moment, m_hat, v_hat
    
    def predict(self, X):
        z1 = X @ self.w1 + self.b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = a1 @ self.w2 + self.b2
        a2 = 1 / (1 + np.exp(-z2))
        y_hat = a2
        return y_hat

def train_test_split(number_of_elements, testing_size):
    training_index = [i for i in range(0, number_of_elements)]
    testing_index = []
    for i in range(1, testing_size + 1):
        np.random.seed(43)
        choice = np.random.choice(training_index, replace=False)
        training_index.remove(choice)
        testing_index.append(choice)
    return training_index, testing_index

def kfold_cross_validation(number_of_folds, data_size):
    lists = []
    for i in range(0, number_of_folds):
        lists.append([])
    for j in range(0, data_size):
        list_index = j % number_of_folds
        lists[list_index].append(j)
    return lists