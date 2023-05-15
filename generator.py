import yaml
import io
import random
import csv
from numpy import random as nprand
import numpy as np

class Cluster:

    def __init__(self, numberOfPoints, dimensions):
        self.numberOfPoints = numberOfPoints
        self.dimensions = dimensions
        
    def generate(self):
        generatedValues = []
        for dimension in self.dimensions:
            generatedDimensionValues = nprand.normal(loc=dimension['center'], scale=dimension['deviation'], size=(1, self.numberOfPoints))
            generatedValues.append(generatedDimensionValues[0])
            
        points = []
        for pointIndex in range(0, self.numberOfPoints):
            pointValues = []
            for generatedValue in generatedValues:
            
                pointValues.append(generatedValue[pointIndex])
            points.append(pointValues)
        return points

def createClusterFromDefault(defaultCluster, numberOfDimensions, numberOfPoints):
    dimensions = []
    for i in range(0, numberOfDimensions):
        dimensions.append({'center' : random.randint(defaultCluster['center_min'], defaultCluster['center_max']), 'deviation' : random.randint(defaultCluster['deviation_min'], defaultCluster['deviation_max'])})
    return Cluster(numberOfPoints, dimensions)


def createClusterFromDefined(definedCluster, defaultCluster, numberOfDimensions, numberOfPoints):
    dimensions = []
    definedDimensions = [] if definedCluster.get('dimensions') == None else definedCluster['dimensions']
    numberOfDefinedDimensions = len(definedDimensions)
    centerMin = defaultCluster['center_min'] if definedCluster.get('center_min') == None else definedCluster['center_min']
    centerMax = defaultCluster['center_max'] if definedCluster.get('center_max') == None else definedCluster['center_max']
    deviationMin = defaultCluster['deviation_min'] if definedCluster.get('deviation_min') == None else definedCluster['deviation_min']
    deviationMax = defaultCluster['deviation_max'] if definedCluster.get('deviation_max') == None else definedCluster['deviation_max']
    for i in range(0, numberOfDimensions):
        if numberOfDefinedDimensions > i:
            dimensions.append({'center' : definedDimensions[i]['center'], 'deviation' : definedDimensions[i]['deviation']})
        else:
            dimensions.append({'center' : random.randint(centerMin, centerMax), 'deviation' : random.randint(deviationMin, deviationMax)})
    return Cluster(numberOfPoints, dimensions)

with open('generator_config.yaml', 'r') as stream:
    configuration = yaml.safe_load(stream)

numberOfDatapoints = configuration['number_of_datapoints']
numberOfDimensions = configuration['number_of_dimensions']
numberOfClusters = configuration['number_of_clusters']
defaultCluster = configuration['default_cluster']
definedClusters = [] if configuration.get('clusters') == None else configuration['clusters']

if configuration.get('seed') != None:
    random.seed(configuration['seed'])
    np.random.seed(configuration['seed'])

clusters = []
numberOfDefinedClusters = len(definedClusters)
definedShareSum = 0
definedShareCount = 0

for i in range (0, numberOfDefinedClusters):
    if definedClusters[i].get('share') != None:
        definedShareCount = definedShareCount + 1
        definedShareSum = definedShareSum + definedClusters[i].get('share')

defaultShare = (1 - definedShareSum) / (numberOfClusters - definedShareCount)

for i in range (0, numberOfClusters):
    if numberOfDefinedClusters > i:
        clusterShare = defaultShare if definedClusters[i].get('share') == None else definedClusters[i]['share']
        cluster = createClusterFromDefined(definedClusters[i], defaultCluster, numberOfDimensions, int(numberOfDatapoints * clusterShare))
    else:
        cluster = createClusterFromDefault(defaultCluster, numberOfDimensions, int(numberOfDatapoints * defaultShare))
    clusters.append(cluster)
allPoints = []
for i, cluster in enumerate(clusters):
    points = cluster.generate()
    allPoints = allPoints + points

np.savetxt(f'generated_data/points_{str(numberOfDatapoints)}_{str(numberOfDimensions)}.csv', allPoints, delimiter =',')