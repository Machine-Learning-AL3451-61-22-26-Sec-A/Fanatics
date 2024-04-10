import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # Attribute used for splitting
        self.label = label  # Class label if it's a leaf node
        self.children = {}  # Dictionary to store child nodes

def entropy(data):
    # Calculate entropy of a dataset
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    entropy = 0
    for count in label_counts.values():
        probability = count / len(labels)
        entropy -= probability * np.log2(probability)
    return entropy

def information_gain(data, attribute_index):
    # Calculate information gain of splitting the dataset on a particular attribute
    total_entropy = entropy(data)
    attribute_values = set([row[attribute_index] for row in data])
    weighted_entropy = 0
    for value in attribute_values:
        subset = [row for row in data if row[attribute_index] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

def majority_label(data):
    # Return the majority class label in a dataset
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

def id3(data, attributes):
    # Base cases
    if len(set([row[-1] for row in data])) == 1:
        # If all instances have the same class label, return a leaf node
        return Node(label=data[0][-1])
    if len(attributes) == 0:
        # If there are no more attributes to split on, return a leaf node with majority label
        return Node(label=majority_label(data))
    
    # Find the best attribute to split on
    gains = [information_gain(data, i) for i in range(len(attributes))]
    best_attribute_index = np.argmax(gains)
    best_attribute = attributes[best_attribute_index]
    
    # Create a new node with the best attribute
    node = Node(attribute=best_attribute)
    
    # Recursively construct subtree for each value of the best attribute
    attribute_values = set([row[best_attribute_index] for row in data])
    for value in attribute_values:
        subset = [row[:-1] for row in data if row[best_attribute_index] == value]
        child_attributes = attributes[:best_attribute_index] + attributes[best_attribute_index+1:]
        child_node = id3(subset, child_attributes)
        node.children[value] = child_node
    
    return node

def predict(tree, instance):
    # Predict the class label for a single instance using the decision tree
    if tree.label is not None:
        # If the node is a leaf node, return the class label
        return tree.label
    attribute_value = instance[tree.attribute]
    if attribute_value not in tree.children:
        # If the attribute value is not seen during training, return majority label
        return majority_label([instance])
    child_node = tree.children[attribute_value]
    return predict(child_node, instance)

# Example usage:
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

attributes = [0, 1, 2, 3]  # Indices of attributes
tree = id3(data, attributes)

# Predicting a new instance
new_instance = ['Sunny', 'Mild', 'High', 'Weak']
prediction = predict(tree, new_instance)
print("Prediction:", prediction)
